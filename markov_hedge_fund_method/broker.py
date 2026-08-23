"""Broker abstraction — the execution seam.

The whole point of this module is that the terminal talks to a *Broker*
interface, not to Alpaca directly. In DASHBOARD mode every read works
(account, positions) but `set_target_position` refuses to place orders. Flip
the mode to PAPER and the exact same code path starts submitting orders
against a paper account — that is how "auto-trade later" is made viable
without shipping a live trader today.

Nothing here runs unless credentials are present and a method is called, so
importing this module is always safe (the alpaca SDK is imported lazily).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Mode, Settings
from .orders import OrderTicket, build_order_request, describe


@dataclass
class Account:
    cash: float
    equity: float
    buying_power: float
    status: str
    last_equity: float = 0.0  # yesterday's close — drives day P&L
    # Pattern-day-trader state. FINRA caps a margin account under $25,000 at
    # three day trades per rolling five business days; a fourth flags the
    # account and restricts it for ninety days. Alpaca tracks the count, so the
    # terminal has no excuse for letting a user walk into it blind.
    daytrade_count: int = 0
    pattern_day_trader: bool = False
    daytrading_buying_power: float = 0.0


@dataclass
class Position:
    symbol: str
    qty: float
    market_value: float
    unrealized_pl: float
    side: str  # "long" / "short"
    avg_entry: float = 0.0
    current_price: float = 0.0
    unrealized_plpc: float = 0.0  # fraction, e.g. 0.042 = +4.2%


@dataclass
class OpenOrder:
    id: str
    symbol: str
    side: str
    type: str
    qty: str
    status: str


@dataclass
class OrderResult:
    id: str
    status: str
    summary: str


class ReadOnlyError(RuntimeError):
    """Raised when an order is attempted while the mode forbids trading."""


class AlpacaBroker:
    """Thin wrapper over alpaca-py's TradingClient with a gated order path."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    # ── lazy client ────────────────────────────────────────────────────────
    @property
    def client(self):
        if self._client is None:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(
                self.settings.api_key,
                self.settings.api_secret,
                paper=self.settings.paper,
            )
        return self._client

    # ── reads (allowed in every mode) ──────────────────────────────────────
    def get_account(self) -> Account:
        a = self.client.get_account()
        return Account(
            cash=float(a.cash),
            equity=float(a.equity),
            buying_power=float(a.buying_power),
            status=str(a.status),
            last_equity=float(getattr(a, "last_equity", 0) or 0),
            daytrade_count=int(getattr(a, "daytrade_count", 0) or 0),
            pattern_day_trader=bool(getattr(a, "pattern_day_trader", False)),
            daytrading_buying_power=float(getattr(a, "daytrading_buying_power", 0) or 0),
        )

    def latest_quote(self, symbol: str) -> dict | None:
        """The current bid/ask from Alpaca — the venue the order will meet.

        Analysis runs on whatever feed gives the best picture of the session,
        and the default one is delayed by about fifteen minutes. That is fine
        for deciding *whether* to own something over days, and not fine for
        deciding the limit price you are about to send: fifteen minutes is
        plenty of room for a limit to be posted through the market or left
        stranded behind it.

        So the price that goes on the ticket comes from the broker. Even the
        free IEX quote is the right thing to ask here, because the question is
        not "what is this worth" but "what will my order actually meet".

        None when no quote is available — an order still goes, it just goes
        without the check, which is the behaviour that existed before.
        """
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest

            client = StockHistoricalDataClient(self.settings.api_key,
                                               self.settings.api_secret)
            sym = symbol.strip().upper()
            q = client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=sym))[sym]
            bid, ask = float(q.bid_price or 0), float(q.ask_price or 0)
            if bid <= 0 and ask <= 0:
                return None
            mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else (bid or ask)
            spread = (ask - bid) if (bid > 0 and ask > 0) else 0.0
            return {
                "symbol": sym, "bid": bid, "ask": ask, "mid": round(mid, 4),
                "spread": round(spread, 4),
                "spreadPct": round(spread / mid * 100.0, 4) if mid else None,
                "asOf": str(getattr(q, "timestamp", "") or ""),
            }
        except Exception:  # noqa: BLE001 — a missing quote must not stop an order
            return None

    @staticmethod
    def _position(p) -> Position:
        return Position(
            symbol=str(p.symbol),
            qty=float(p.qty),
            market_value=float(p.market_value),
            unrealized_pl=float(p.unrealized_pl),
            side=str(getattr(getattr(p, "side", "long"), "value", getattr(p, "side", "long"))),
            avg_entry=float(getattr(p, "avg_entry_price", 0) or 0),
            current_price=float(getattr(p, "current_price", 0) or 0),
            unrealized_plpc=float(getattr(p, "unrealized_plpc", 0) or 0),
        )

    def get_position(self, symbol: str) -> Position | None:
        try:
            p = self.client.get_open_position(symbol)
        except Exception:
            return None  # alpaca raises when there is no open position
        return self._position(p)

    def list_positions(self) -> list[Position]:
        """Every open position on the account (for the blotter)."""
        try:
            return [self._position(p) for p in self.client.get_all_positions()]
        except Exception:  # noqa: BLE001
            return []

    # ── the gated write ────────────────────────────────────────────────────
    def set_target_position(self, symbol: str, target: int, notional: float) -> str:
        """Move toward a target exposure of -1 (short) / 0 (flat) / +1 (long).

        THIS IS THE AUTO-TRADE SEAM. It is a hard error unless the mode
        explicitly allows trading, so DASHBOARD mode can never place an order
        even if the executor calls it by mistake.
        """
        if not self.settings.can_trade:
            raise ReadOnlyError(
                f"Mode is {self.settings.mode.value.upper()} — order placement is disabled. "
                "Set mode to PAPER to enable simulated trading."
            )

        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        current = self.get_position(symbol)
        current_signed_qty = 0.0
        if current is not None:
            current_signed_qty = current.qty if current.side == "long" else -current.qty

        # Flat target: close whatever is open.
        if target == 0:
            if current is not None:
                self.client.close_position(symbol)
                return f"close {symbol}"
            return "already flat"

        # Long/short target: for a paper MVP, close any opposite position then
        # open a notional-sized market order in the target direction. A
        # production executor would net the delta instead of round-tripping.
        if current is not None and (current_signed_qty > 0) != (target > 0):
            self.client.close_position(symbol)

        side = OrderSide.BUY if target > 0 else OrderSide.SELL
        order = MarketOrderRequest(
            symbol=symbol,
            notional=round(max(notional, 1.0), 2),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        submitted = self.client.submit_order(order)
        return f"{side.value} {symbol} ~${notional:,.0f} (order {submitted.id})"

    # ── manual execution panel: full order taxonomy ─────────────────────────
    def submit_ticket(self, ticket: OrderTicket) -> OrderResult:
        """Submit any order style (market/limit/stop/stop_limit/trailing_stop,
        simple/bracket/oco/oto). Gated by mode exactly like the auto-trade seam.

        The request is fully validated and constructed before this point, so a
        rejection here comes from Alpaca (buying power, market hours, etc.), not
        from a malformed ticket.
        """
        if not self.settings.can_trade:
            raise ReadOnlyError(
                f"Mode is {self.settings.mode.value.upper()} — order placement is disabled. "
                "Restart with --mode paper (or live) to enable order entry."
            )
        request = build_order_request(ticket)  # raises OrderValidationError on bad input
        submitted = self.client.submit_order(request)
        return OrderResult(
            id=str(submitted.id),
            status=str(getattr(submitted, "status", "accepted")),
            summary=describe(ticket),
        )

    def symbols_opened_today(self) -> set[str]:
        """Symbols with a buy filled in this session — the first leg of a
        potential day trade. Empty on any failure, and the caller must treat
        that as "unknown" rather than "none", or a broker hiccup would quietly
        switch the day-trade guard off.
        """
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        from . import session

        out: set[str] = set()
        try:
            since = session.session_bounds(None)
            after = since[0] if since else None
            orders = self.client.get_orders(filter=GetOrdersRequest(
                status=QueryOrderStatus.CLOSED, after=after, limit=500))
            for o in orders:
                if str(getattr(o, "status", "")).lower() != "filled":
                    continue
                if str(getattr(o, "side", "")).lower().endswith("buy"):
                    out.add(str(o.symbol).upper())
        except Exception:  # noqa: BLE001
            return set()
        return out

    def list_open_orders(self) -> list[OpenOrder]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self.client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
        )
        out: list[OpenOrder] = []
        for o in orders:
            out.append(OpenOrder(
                id=str(o.id),
                symbol=str(o.symbol),
                side=str(getattr(o.side, "value", o.side)),
                type=str(getattr(o.type, "value", o.type)),
                qty=str(o.qty if o.qty is not None else f"${o.notional}"),
                status=str(getattr(o.status, "value", o.status)),
            ))
        return out

    def cancel_all_orders(self) -> int:
        """Cancel every open order. Returns how many cancels were requested."""
        if not self.settings.can_trade:
            raise ReadOnlyError(
                f"Mode is {self.settings.mode.value.upper()} — cancelling is disabled."
            )
        responses = self.client.cancel_orders()
        return len(responses) if responses is not None else 0

    def cancel_order(self, order_id: str) -> None:
        """Cancel a single open order by id."""
        if not self.settings.can_trade:
            raise ReadOnlyError(
                f"Mode is {self.settings.mode.value.upper()} — cancelling is disabled."
            )
        self.client.cancel_order_by_id(order_id)

    def close_position(self, symbol: str, *, cancel_first: bool = True) -> dict:
        """Submit a market order to liquidate the whole position in `symbol`.

        Two things this has to be honest about:

        * Shares reserved by a working order cannot be liquidated — Alpaca
          rejects the close with "insufficient qty available for order". So any
          open order on the symbol is cancelled first, which is the usual reason
          a close appears to fail on the first attempt.
        * `close_position` submits an ORDER; it does not settle instantly. When
          the market is shut the order simply queues. Callers get the order id
          and status back so they can say that plainly instead of claiming the
          position is closed.
        """
        if not self.settings.can_trade:
            raise ReadOnlyError(
                f"Mode is {self.settings.mode.value.upper()} — closing positions is disabled."
            )
        cancelled = 0
        if cancel_first:
            try:
                for o in self.list_open_orders():
                    if str(o.symbol).upper() == symbol.upper():
                        self.cancel_order(o.id)
                        cancelled += 1
            except Exception:  # noqa: BLE001 — a failed cancel must not stop the close
                pass
        order = self.client.close_position(symbol)
        return {
            "symbol": symbol,
            "orderId": str(getattr(order, "id", "")),
            "status": str(getattr(getattr(order, "status", ""), "value",
                                  getattr(order, "status", ""))),
            "qty": str(getattr(order, "qty", "") or ""),
            "cancelledOrders": cancelled,
        }

    def position_qty(self, symbol: str) -> float:
        """Signed size held right now — 0.0 when flat. Used to verify a close."""
        p = self.get_position(symbol)
        if p is None:
            return 0.0
        return float(p.qty) if p.side != "short" else -float(p.qty)

    # ── asset universe (for search + validation) ────────────────────────────
    def get_asset(self, symbol: str) -> dict | None:
        """Look up a single Alpaca asset. None when it doesn't exist."""
        try:
            a = self.client.get_asset(symbol.upper())
        except Exception:  # noqa: BLE001 — unknown symbol / API error
            return None
        return {
            "symbol": a.symbol,
            "name": getattr(a, "name", "") or "",
            "tradable": bool(getattr(a, "tradable", False)),
            "fractionable": bool(getattr(a, "fractionable", False)),
            "status": str(getattr(a.status, "value", getattr(a, "status", ""))),
        }

    def list_tradable_symbols(self) -> list[str]:
        """All active, tradable US-equity symbols (for the search universe)."""
        return sorted({a["symbol"] for a in self.list_tradable_assets()})

    def list_tradable_assets(self) -> list[dict]:
        """All active, tradable US-equity assets as {symbol, name}."""
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest

        assets = self.client.get_all_assets(
            GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        )
        return [{"symbol": a.symbol, "name": getattr(a, "name", "") or "",
                 # Alpaca marks thousands of OTC names tradable but publishes no
                 # bars for them. The exchange is the only dependable way to
                 # tell them apart from a NASDAQ listing.
                 "exchange": str(getattr(getattr(a, "exchange", ""), "value",
                                         getattr(a, "exchange", "")) or "")}
                for a in assets if getattr(a, "tradable", False)]


def make_broker(settings: Settings) -> AlpacaBroker | None:
    """Return a broker when credentials exist, else None (pure-data mode)."""
    if not settings.has_credentials:
        return None
    return AlpacaBroker(settings)
