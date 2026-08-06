"""Mamba Terminal — web HUD backend (`mamba-web`).

A small FastAPI app that serves the neon dashboard and feeds it live data:

    GET  /                     the HUD page
    GET  /api/config           mode / account / demo flags
    GET  /api/symbols          default watchlist
    GET  /api/state?symbol=SPY regime, matrix, signal, greed/fear, chart series
    GET  /api/portfolio        equity, positions, open orders (broker)
    GET  /api/accounts         list account profiles
    POST /api/accounts         add a profile {name,key,secret,paper}
    POST /api/accounts/active  switch active profile {name}
    DEL  /api/accounts/{name}  remove a profile
    POST /api/orders           submit any order style (full taxonomy)
    POST /api/orders/cancel_all cancel every open order

Reuses the existing engine/regime/orders/broker/accounts modules, so the web UI
and the terminal share one brain. `main()` launches uvicorn and opens a browser.

NOTE: this module deliberately does NOT use `from __future__ import annotations`.
FastAPI resolves endpoint parameter types from real annotation objects; stringized
annotations (from that future import) break body/Request detection here.
"""

import os
import time
from dataclasses import replace

from .accounts import AccountStore
from .broker import ReadOnlyError, make_broker
from .config import Mode, Settings, load_settings
from .market_data import get_history, synthetic_close
from .markov2 import Strategy
from .news import fetch_news
from .orders import OrderTicket, OrderValidationError
from .webstate import market_state, quote_state

STATIC_DIR = os.path.join(os.path.dirname(__file__), "web_static")
DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT", "BTC-USD"]

# A small bundled universe so the search box has suggestions offline. Typing any
# ticker not in this list still works — it's used verbatim.
SYMBOL_UNIVERSE = sorted(set(DEFAULT_SYMBOLS + [
    "AMZN", "GOOGL", "GOOG", "META", "NFLX", "AMD", "INTC", "MU", "AVGO", "QCOM",
    "CRM", "ORCL", "ADBE", "CSCO", "IBM", "TXN", "NOW", "SHOP", "UBER", "ABNB",
    "PLTR", "SNOW", "COIN", "SQ", "PYPL", "V", "MA", "JPM", "BAC", "WFC", "GS",
    "MS", "C", "BRK.B", "BX", "SCHW", "KO", "PEP", "MCD", "SBUX", "NKE", "DIS",
    "WMT", "COST", "TGT", "HD", "LOW", "PG", "JNJ", "PFE", "MRK", "ABBV", "LLY",
    "UNH", "CVS", "XOM", "CVX", "COP", "OXY", "BA", "CAT", "GE", "F", "GM",
    "T", "VZ", "TMUS", "DAL", "AAL", "UAL", "RIVN", "LCID", "NIO", "SOFI",
    "DKNG", "ROKU", "ZM", "DOCU", "TWLO", "NET", "DDOG", "CRWD", "ZS", "PANW",
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "ARKK", "XLF", "XLK", "XLE", "GLD",
    "SLV", "USO", "TLT", "HYG", "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD",
]))


def _seed(symbol: str) -> int:
    return sum(ord(ch) for ch in symbol) % 997


class AppState:
    """Mutable app state for the single local user."""

    # Recomputing the walk-forward backtest per symbol is the slow part; cache
    # the finished payload briefly so flipping between assets is instant.
    CACHE_TTL = 3600.0  # demo data is deterministic; refreshed by version bumps

    def __init__(self, settings: Settings, demo: bool = False):
        self.settings = settings
        self.demo = demo
        self.accounts = AccountStore()
        self.broker = None if demo else make_broker(settings)
        self._state_cache: dict[str, tuple[float, dict]] = {}
        self.ttl = self.CACHE_TTL if demo else 60.0  # live data goes stale sooner

    def reconnect(self) -> None:
        self.broker = None if self.demo else make_broker(self.settings)
        self._state_cache.clear()  # a new account may change the data source

    def close_for(self, symbol: str):
        """Close series for a symbol, always returning something renderable."""
        if self.demo:
            return synthetic_close(seed=_seed(symbol)), "synthetic (demo)"
        try:
            return get_history(replace(self.settings, ticker=symbol)), "live"
        except Exception:  # noqa: BLE001 — never leave the HUD blank
            return synthetic_close(seed=_seed(symbol)), "synthetic (data unavailable)"

    def state_payload(self, symbol: str) -> dict:
        """Full HUD payload for a symbol, memoised with a short TTL."""
        cached = self._state_cache.get(symbol)
        if cached and (time.monotonic() - cached[0]) < self.ttl:
            return cached[1]
        close, source = self.close_for(symbol)
        payload = market_state(close, symbol, window=self.settings.window,
                               threshold=self.settings.threshold, strategy=self.settings.strategy)
        payload["dataSource"] = source
        self._state_cache[symbol] = (time.monotonic(), payload)
        return payload


_ORDER_FIELDS = {
    "symbol", "side", "order_type", "time_in_force", "order_class", "qty", "notional",
    "limit_price", "stop_price", "trail_price", "trail_percent",
    "take_profit_limit", "stop_loss_stop", "stop_loss_limit", "extended_hours",
}


def create_app(state: AppState):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="Mamba Terminal", docs_url=None, redoc_url=None)

    # ── page ─────────────────────────────────────────────────────────────────
    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/api/config")
    def config():
        return {
            "mode": state.settings.mode.value,
            "canTrade": state.settings.can_trade,
            "demo": state.demo,
            "account": state.settings.account,
            "accountPaper": state.settings.account_paper,
            "connected": state.broker is not None,
            "symbols": DEFAULT_SYMBOLS,
            "defaultSymbol": state.settings.ticker,
        }

    @app.get("/api/symbols")
    def symbols():
        return {"symbols": DEFAULT_SYMBOLS}

    @app.get("/api/search")
    def search(q: str = ""):
        q = (q or "").strip().upper()
        if not q:
            return {"results": DEFAULT_SYMBOLS}
        starts = [s for s in SYMBOL_UNIVERSE if s.startswith(q)]
        contains = [s for s in SYMBOL_UNIVERSE if q in s and s not in starts]
        results = (starts + contains)[:12]
        if q not in results:  # always allow the exact ticker the user typed
            results = [q] + results
        return {"results": results[:12]}

    @app.get("/api/state")
    def state_endpoint(symbol: str = "SPY"):
        symbol = symbol.strip().upper() or "SPY"
        return state.state_payload(symbol)

    @app.get("/api/quote")
    def quote(symbol: str = "SPY"):
        symbol = symbol.strip().upper() or "SPY"
        cached = state._state_cache.get(symbol)
        if cached and (time.monotonic() - cached[0]) < state.ttl:
            p = cached[1]
            return {"ticker": symbol, "lastPrice": p["lastPrice"], "regime": p["regime"]}
        close, _ = state.close_for(symbol)
        return quote_state(close, symbol, window=state.settings.window,
                           threshold=state.settings.threshold)

    @app.get("/api/news")
    def news(symbol: str = "SPY"):
        symbol = symbol.strip().upper() or "SPY"
        return {"symbol": symbol, "items": fetch_news(state.settings, symbol, demo=state.demo)}

    # ── portfolio ────────────────────────────────────────────────────────────
    @app.get("/api/portfolio")
    def portfolio(symbol: str = "SPY"):
        if state.broker is None:
            return {"connected": False, "account": None, "position": None, "openOrders": []}
        try:
            acct = state.broker.get_account()
            pos = state.broker.get_position(symbol.strip().upper())
            orders = state.broker.list_open_orders()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"broker read failed: {exc}")
        return {
            "connected": True,
            "account": {"equity": acct.equity, "cash": acct.cash,
                        "buyingPower": acct.buying_power, "status": acct.status},
            "position": None if pos is None else {
                "symbol": pos.symbol, "qty": pos.qty, "side": pos.side,
                "marketValue": pos.market_value, "unrealizedPl": pos.unrealized_pl},
            "openOrders": [
                {"id": o.id, "symbol": o.symbol, "side": o.side, "type": o.type,
                 "qty": o.qty, "status": o.status}
                for o in orders
            ],
        }

    # ── orders ───────────────────────────────────────────────────────────────
    @app.post("/api/orders")
    async def submit_order(request: Request):
        if state.broker is None:
            raise HTTPException(status_code=403, detail="No account connected. Add one under Accounts.")
        data = await request.json()
        if not isinstance(data, dict) or not data.get("symbol"):
            raise HTTPException(status_code=400, detail="An order needs at least a symbol.")
        ticket = OrderTicket(**{k: v for k, v in data.items() if k in _ORDER_FIELDS})
        try:
            result = state.broker.submit_ticket(ticket)
        except OrderValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except ReadOnlyError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 — Alpaca rejection
            raise HTTPException(status_code=502, detail=f"order rejected: {exc}")
        return {"id": result.id, "status": result.status, "summary": result.summary}

    @app.post("/api/orders/cancel_all")
    def cancel_all():
        if state.broker is None:
            raise HTTPException(status_code=403, detail="No account connected.")
        try:
            n = state.broker.cancel_all_orders()
        except ReadOnlyError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"cancel failed: {exc}")
        return {"cancelled": n}

    # ── accounts ─────────────────────────────────────────────────────────────
    @app.get("/api/accounts")
    def list_accounts():
        try:
            profiles = state.accounts.list()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc))
        return {
            "active": state.accounts.active() if profiles else None,
            "accounts": [{"name": p.name, "paper": p.paper, "active": p.active} for p in profiles],
        }

    def _connect(name: str):
        resolved = state.accounts.resolve(name)
        if resolved is not None:
            state.settings.api_key = resolved.key_id
            state.settings.api_secret = resolved.secret
            state.settings.account = resolved.name
            state.settings.account_paper = resolved.paper
            state.reconnect()

    @app.post("/api/accounts")
    async def add_account(request: Request):
        data = await request.json()
        try:
            state.accounts.add(str(data.get("name", "")), str(data.get("key_id", "")),
                               str(data.get("secret", "")), paper=bool(data.get("paper", True)))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc))
        _connect(str(data.get("name", "")))
        return list_accounts()

    @app.post("/api/accounts/active")
    async def set_active(request: Request):
        data = await request.json()
        name = str(data.get("name", ""))
        try:
            state.accounts.set_active(name)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc))
        _connect(name)
        return list_accounts()

    @app.delete("/api/accounts/{name}")
    def remove_account(name: str):
        try:
            state.accounts.remove(name)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc))
        active = state.accounts.active()
        if active:
            _connect(active)
        else:
            state.settings.account = state.settings.account_paper = None
            state.settings.api_key = state.settings.api_secret = None
            state.reconnect()
        return list_accounts()

    @app.exception_handler(OrderValidationError)
    def _ove(_request, exc):  # pragma: no cover — safety net
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


def main() -> int:
    import argparse
    import threading
    import webbrowser

    import uvicorn

    parser = argparse.ArgumentParser(prog="mamba-web",
                                     description="Mamba Terminal — neon web HUD")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.DASHBOARD.value)
    parser.add_argument("--account", default=None)
    parser.add_argument("--strategy", choices=[s.value for s in Strategy], default=Strategy.FILTER.value)
    parser.add_argument("--demo", action="store_true", help="offline synthetic data, no network")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--selftest", action="store_true",
                        help="Verify the bundle can build the server, then exit (CI smoke test)")
    args = parser.parse_args()

    settings = load_settings(
        account=args.account, ticker=args.symbol,
        mode=Mode(args.mode), strategy=Strategy(args.strategy),
    )
    state = AppState(settings, demo=args.demo)
    app = create_app(state)

    if args.selftest:
        import uvicorn  # noqa: F401 — ensure the server dependency is bundled
        index = os.path.join(STATIC_DIR, "index.html")
        if not os.path.exists(index):
            print(f"selftest FAILED: HUD asset missing at {index}")
            return 1
        print("mamba-web selftest OK — FastAPI app built, HUD asset present.")
        return 0

    url = f"http://{args.host}:{args.port}/"
    print(f"Mamba Terminal web HUD → {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
