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
from .market_data import (
    get_history,
    get_intraday_ohlc,
    get_ohlc,
    synthetic_close,
    synthetic_intraday,
    synthetic_ohlc,
)
from .markov2 import Strategy
from .news import fetch_news
from .orders import OrderTicket, OrderValidationError
from .webstate import _rsi, market_state, quote_state

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


# Full names for the bundled universe (tooltips). When connected to Alpaca the
# real asset names are used for every symbol; this covers the offline case.
ASSET_NAMES = {
    "SPY": "SPDR S&P 500 ETF Trust", "QQQ": "Invesco QQQ Trust (Nasdaq-100)",
    "IWM": "iShares Russell 2000 ETF", "DIA": "SPDR Dow Jones Industrial Average ETF",
    "VOO": "Vanguard S&P 500 ETF", "VTI": "Vanguard Total Stock Market ETF",
    "ARKK": "ARK Innovation ETF", "XLF": "Financial Select Sector SPDR",
    "XLK": "Technology Select Sector SPDR", "XLE": "Energy Select Sector SPDR",
    "GLD": "SPDR Gold Shares", "SLV": "iShares Silver Trust", "USO": "United States Oil Fund",
    "TLT": "iShares 20+ Year Treasury Bond ETF", "HYG": "iShares iBoxx High Yield Corp Bond ETF",
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.", "GOOGL": "Alphabet Inc. (Class A)", "GOOG": "Alphabet Inc. (Class C)",
    "META": "Meta Platforms, Inc.", "TSLA": "Tesla, Inc.", "NFLX": "Netflix, Inc.",
    "AMD": "Advanced Micro Devices, Inc.", "INTC": "Intel Corporation", "MU": "Micron Technology, Inc.",
    "AVGO": "Broadcom Inc.", "QCOM": "QUALCOMM Incorporated", "CRM": "Salesforce, Inc.",
    "ORCL": "Oracle Corporation", "ADBE": "Adobe Inc.", "CSCO": "Cisco Systems, Inc.",
    "IBM": "International Business Machines", "TXN": "Texas Instruments Incorporated",
    "NOW": "ServiceNow, Inc.", "SHOP": "Shopify Inc.", "UBER": "Uber Technologies, Inc.",
    "ABNB": "Airbnb, Inc.", "PLTR": "Palantir Technologies Inc.", "SNOW": "Snowflake Inc.",
    "COIN": "Coinbase Global, Inc.", "SQ": "Block, Inc.", "PYPL": "PayPal Holdings, Inc.",
    "V": "Visa Inc.", "MA": "Mastercard Incorporated", "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corporation", "WFC": "Wells Fargo & Company", "GS": "Goldman Sachs Group",
    "MS": "Morgan Stanley", "C": "Citigroup Inc.", "BRK.B": "Berkshire Hathaway (Class B)",
    "BX": "Blackstone Inc.", "SCHW": "Charles Schwab Corporation", "KO": "The Coca-Cola Company",
    "PEP": "PepsiCo, Inc.", "MCD": "McDonald's Corporation", "SBUX": "Starbucks Corporation",
    "NKE": "NIKE, Inc.", "DIS": "The Walt Disney Company", "WMT": "Walmart Inc.",
    "COST": "Costco Wholesale Corporation", "TGT": "Target Corporation", "HD": "The Home Depot, Inc.",
    "LOW": "Lowe's Companies, Inc.", "PG": "The Procter & Gamble Company", "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer Inc.", "MRK": "Merck & Co., Inc.", "ABBV": "AbbVie Inc.", "LLY": "Eli Lilly and Company",
    "UNH": "UnitedHealth Group Incorporated", "CVS": "CVS Health Corporation",
    "XOM": "Exxon Mobil Corporation", "CVX": "Chevron Corporation", "COP": "ConocoPhillips",
    "OXY": "Occidental Petroleum", "BA": "The Boeing Company", "CAT": "Caterpillar Inc.",
    "GE": "General Electric Company", "F": "Ford Motor Company", "GM": "General Motors Company",
    "T": "AT&T Inc.", "VZ": "Verizon Communications Inc.", "TMUS": "T-Mobile US, Inc.",
    "DAL": "Delta Air Lines, Inc.", "AAL": "American Airlines Group", "UAL": "United Airlines Holdings",
    "RIVN": "Rivian Automotive, Inc.", "LCID": "Lucid Group, Inc.", "NIO": "NIO Inc.",
    "SOFI": "SoFi Technologies, Inc.", "DKNG": "DraftKings Inc.", "ROKU": "Roku, Inc.",
    "ZM": "Zoom Video Communications", "DOCU": "DocuSign, Inc.", "TWLO": "Twilio Inc.",
    "NET": "Cloudflare, Inc.", "DDOG": "Datadog, Inc.", "CRWD": "CrowdStrike Holdings",
    "ZS": "Zscaler, Inc.", "PANW": "Palo Alto Networks, Inc.", "A": "Agilent Technologies, Inc.",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana", "DOGE-USD": "Dogecoin",
}


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
        self._news_cache: dict[str, tuple[float, list]] = {}
        self._alpaca_symbols: set[str] | None = None
        self._alpaca_names: dict[str, str] = {}
        self.ttl = self.CACHE_TTL if demo else 60.0  # live data goes stale sooner

    def reconnect(self) -> None:
        self.broker = None if self.demo else make_broker(self.settings)
        self._state_cache.clear()   # a new account may change the data source
        self._alpaca_symbols = None  # and a different asset universe
        self._alpaca_names = {}

    def _ensure_alpaca_universe(self) -> None:
        if self.broker is None or self._alpaca_symbols is not None:
            return
        try:
            assets = self.broker.list_tradable_assets()  # [{symbol, name}]
            self._alpaca_symbols = {a["symbol"] for a in assets}
            self._alpaca_names = {a["symbol"]: a["name"] for a in assets if a.get("name")}
        except Exception:  # noqa: BLE001 — cache empty to avoid refetch storms
            self._alpaca_symbols = set()
            self._alpaca_names = {}

    def alpaca_symbols(self) -> set[str] | None:
        """Set of tradable Alpaca symbols when connected (cached), else None."""
        if self.broker is None:
            return None
        self._ensure_alpaca_universe()
        return self._alpaca_symbols or None

    def name_for(self, symbol: str) -> str:
        """Full asset name — Alpaca's when connected, else the bundled list."""
        symbol = symbol.upper()
        if self.broker is not None:
            self._ensure_alpaca_universe()
            name = self._alpaca_names.get(symbol)
            if name:
                return name
        return ASSET_NAMES.get(symbol, "")

    def search_universe(self) -> list[str]:
        al = self.alpaca_symbols()
        return sorted(al) if al else SYMBOL_UNIVERSE

    def close_for(self, symbol: str):
        """Close series for a symbol, always returning something renderable."""
        if self.demo:
            return synthetic_close(seed=_seed(symbol)), "synthetic (demo)"
        try:
            return get_history(replace(self.settings, ticker=symbol)), "live"
        except Exception:  # noqa: BLE001 — never leave the HUD blank
            return synthetic_close(seed=_seed(symbol)), "synthetic (data unavailable)"

    def ohlc_for(self, symbol: str):
        """OHLC frame for a symbol (for candlesticks), always renderable."""
        if self.demo:
            return synthetic_ohlc(seed=_seed(symbol)), "synthetic (demo)"
        try:
            return get_ohlc(replace(self.settings, ticker=symbol)), "live"
        except Exception:  # noqa: BLE001 — never leave the HUD blank
            return synthetic_ohlc(seed=_seed(symbol)), "synthetic (data unavailable)"

    def intraday_for(self, symbol: str, tf: str):
        """Intraday OHLC (1D/1W) for a symbol, always renderable."""
        if self.demo:
            return synthetic_intraday(tf, seed=_seed(symbol)), "synthetic (demo)"
        try:
            return get_intraday_ohlc(replace(self.settings, ticker=symbol), tf), "live"
        except Exception:  # noqa: BLE001 — fall back so the chart never blanks
            return synthetic_intraday(tf, seed=_seed(symbol)), "synthetic (data unavailable)"

    def state_payload(self, symbol: str) -> dict:
        """Full HUD payload for a symbol, memoised with a short TTL."""
        cached = self._state_cache.get(symbol)
        if cached and (time.monotonic() - cached[0]) < self.ttl:
            return cached[1]
        df, source = self.ohlc_for(symbol)
        close = df["Close"]
        payload = market_state(close, symbol, window=self.settings.window,
                               threshold=self.settings.threshold, strategy=self.settings.strategy,
                               ohlc=df)
        payload["dataSource"] = source
        payload["name"] = self.name_for(symbol)
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

    @app.post("/api/mode")
    async def set_mode(request: Request):
        data = await request.json()
        m = str(data.get("mode", "")).strip().lower()
        try:
            state.settings.mode = Mode(m)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"invalid mode: {m!r}")
        return config()

    @app.get("/api/search")
    def search(q: str = ""):
        q = (q or "").strip().upper()
        connected = state.alpaca_symbols() is not None
        universe = state.search_universe()
        if not q:
            return {"results": DEFAULT_SYMBOLS, "connected": connected}
        starts = [s for s in universe if s.startswith(q)]
        contains = [s for s in universe if q in s and s not in starts]
        results = (starts + contains)[:30]
        # Offline we can't verify against Alpaca, so allow the exact ticker typed.
        if not connected and q not in results:
            results = [q] + results
        results = results[:30]
        return {"results": [{"symbol": s, "name": state.name_for(s)} for s in results],
                "connected": connected}

    @app.get("/api/validate")
    def validate(symbol: str = ""):
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return {"valid": False, "reason": "empty"}
        al = state.alpaca_symbols()
        if al is not None:
            if symbol in al:
                asset = state.broker.get_asset(symbol)
                return {"valid": True, "source": "alpaca",
                        "tradable": asset["tradable"] if asset else True,
                        "name": asset["name"] if asset else ""}
            return {"valid": False, "source": "alpaca",
                    "reason": f"{symbol} is not an Alpaca-tradable symbol"}
        return {"valid": True, "source": "unverified"}  # not connected — can't check

    @app.get("/api/state")
    def state_endpoint(symbol: str = "SPY"):
        symbol = symbol.strip().upper() or "SPY"
        return state.state_payload(symbol)

    @app.get("/api/candles")
    def candles(symbol: str = "SPY", tf: str = "1D"):
        import pandas as pd
        symbol = symbol.strip().upper() or "SPY"
        tf = (tf or "1D").upper()
        df, source = state.intraday_for(symbol, tf)
        if df is None or df.empty:
            return {"symbol": symbol, "tf": tf, "bars": [], "ma20": [], "ma50": [],
                    "rsi": [], "momentum": [], "source": source}
        df = df.iloc[-400:]
        close = df["Close"]
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        rsi = _rsi(close)
        mom = (close / close.shift(14) - 1.0) * 100.0

        def r4(v):
            return round(float(v), 4)

        def ser(s):
            return [None if pd.isna(v) else round(float(v), 4) for v in s]

        bars = [
            {"t": ts.strftime("%Y-%m-%d %H:%M"), "o": r4(o), "h": r4(h), "l": r4(lo),
             "c": r4(c), "up": bool(c >= o)}
            for ts, o, h, lo, c in zip(df.index, df["Open"], df["High"], df["Low"], df["Close"])
        ]
        return {"symbol": symbol, "tf": tf, "bars": bars, "ma20": ser(ma20), "ma50": ser(ma50),
                "rsi": ser(rsi), "momentum": ser(mom), "source": source}

    @app.get("/api/quote")
    def quote(symbol: str = "SPY"):
        symbol = symbol.strip().upper() or "SPY"
        cached = state._state_cache.get(symbol)
        if cached and (time.monotonic() - cached[0]) < state.ttl:
            p = cached[1]
            return {"ticker": symbol, "lastPrice": p["lastPrice"], "regime": p["regime"],
                    "name": p.get("name", "")}
        close, _ = state.close_for(symbol)
        q = quote_state(close, symbol, window=state.settings.window,
                        threshold=state.settings.threshold)
        q["name"] = state.name_for(symbol)
        return q

    @app.get("/api/news")
    def news(symbol: str = "SPY"):
        symbol = symbol.strip().upper() or "SPY"
        cached = state._news_cache.get(symbol)
        if cached and (time.monotonic() - cached[0]) < 300.0:
            items = cached[1]
        else:
            items = fetch_news(state.settings, symbol, demo=state.demo)
            state._news_cache[symbol] = (time.monotonic(), items)
        return {"symbol": symbol, "items": items}

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
