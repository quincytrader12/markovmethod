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


@dataclass
class Position:
    symbol: str
    qty: float
    market_value: float
    unrealized_pl: float
    side: str  # "long" / "short"


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
        )

    def get_position(self, symbol: str) -> Position | None:
        try:
            p = self.client.get_open_position(symbol)
        except Exception:
            return None  # alpaca raises when there is no open position
        return Position(
            symbol=symbol,
            qty=float(p.qty),
            market_value=float(p.market_value),
            unrealized_pl=float(p.unrealized_pl),
            side=str(getattr(p, "side", "long")),
        )

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
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest

        assets = self.client.get_all_assets(
            GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        )
        return sorted({a.symbol for a in assets if getattr(a, "tradable", False)})


def make_broker(settings: Settings) -> AlpacaBroker | None:
    """Return a broker when credentials exist, else None (pure-data mode)."""
    if not settings.has_credentials:
        return None
    return AlpacaBroker(settings)
