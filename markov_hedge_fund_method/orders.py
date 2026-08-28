"""Order tickets and request building — the full Alpaca order taxonomy.

This module turns a plain :class:`OrderTicket` (what the execution panel
collects from the user) into the exact alpaca-py request object Alpaca expects,
covering **every** order style the broker supports:

    type        : market · limit · stop · stop_limit · trailing_stop
    class       : simple · bracket · oco · oto
    time-in-force: day · gtc · opg · cls · ioc · fok
    side        : buy · sell
    sizing      : qty (shares, fractional ok) or notional ($, market/day only)

The build step is a pure function with no network and no client, so it is
fully unit-testable. Validation raises :class:`OrderValidationError` with a
plain-English message the panel shows verbatim; anything that reaches Alpaca is
already shaped correctly for the API.

alpaca-py is imported lazily so importing this module never requires the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical vocabularies (lower-case, matching alpaca-py enum *values*).
ORDER_TYPES = ("market", "limit", "stop", "stop_limit", "trailing_stop")
ORDER_CLASSES = ("simple", "bracket", "oco", "oto")
TIFS = ("day", "gtc", "opg", "cls", "ioc", "fok")
SIDES = ("buy", "sell")

# A "lot" is a fixed block of shares — 100 for US equities by convention. The
# panel can size in lots instead of shares; the ticket converts lots -> qty so
# everything downstream still speaks Alpaca's share-based API.
DEFAULT_LOT_SIZE = 100


class OrderValidationError(ValueError):
    """A ticket that cannot become a valid Alpaca order (shown to the user)."""


@dataclass
class OrderTicket:
    """Everything the execution panel can specify about a single order."""

    symbol: str
    side: str = "buy"                      # buy | sell
    order_type: str = "market"             # see ORDER_TYPES
    time_in_force: str = "day"             # see TIFS
    order_class: str = "simple"            # see ORDER_CLASSES
    qty: float | None = None               # shares (fractional allowed)
    lots: float | None = None              # size in lots; qty = lots * lot_size
    lot_size: int = DEFAULT_LOT_SIZE       # shares per lot (US equities: 100)
    notional: float | None = None          # dollars (market + day + simple only)
    limit_price: float | None = None       # limit / stop_limit
    stop_price: float | None = None        # stop / stop_limit
    trail_price: float | None = None       # trailing_stop ($ offset)
    trail_percent: float | None = None     # trailing_stop (% offset)
    take_profit_limit: float | None = None  # bracket / oco / oto leg
    stop_loss_stop: float | None = None     # bracket / oco / oto leg
    stop_loss_limit: float | None = None    # optional: make the SL leg a stop-limit
    extended_hours: bool = False           # limit/day only
    client_order_id: str | None = None

    def normalized(self) -> "OrderTicket":
        # Sizing in lots is just a share multiple — resolve it to qty here so
        # validation and request building only ever deal with shares.
        lot_size = int(DEFAULT_LOT_SIZE if self.lot_size is None else self.lot_size)
        if lot_size <= 0:
            raise OrderValidationError("Lot size must be a positive number of shares.")
        qty = self.qty
        if self.lots is not None:
            if float(self.lots) <= 0:
                raise OrderValidationError("Lots must be greater than zero.")
            qty = float(self.lots) * lot_size
        return OrderTicket(
            symbol=(self.symbol or "").strip().upper(),
            side=(self.side or "").strip().lower(),
            order_type=(self.order_type or "").strip().lower(),
            time_in_force=(self.time_in_force or "").strip().lower(),
            order_class=(self.order_class or "").strip().lower(),
            qty=qty,
            lots=self.lots,
            lot_size=lot_size,
            notional=self.notional,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            trail_price=self.trail_price,
            trail_percent=self.trail_percent,
            take_profit_limit=self.take_profit_limit,
            stop_loss_stop=self.stop_loss_stop,
            stop_loss_limit=self.stop_loss_limit,
            extended_hours=bool(self.extended_hours),
            client_order_id=self.client_order_id or None,
        )


def _require_positive(value: float | None, label: str) -> float:
    if value is None:
        raise OrderValidationError(f"{label} is required for this order.")
    if value <= 0:
        raise OrderValidationError(f"{label} must be greater than 0.")
    return value


def validate(ticket: OrderTicket) -> OrderTicket:
    """Validate a ticket, returning the normalized copy. Raises on any problem."""
    t = ticket.normalized()

    if not t.symbol:
        raise OrderValidationError("Symbol is required.")
    if t.side not in SIDES:
        raise OrderValidationError(f"Side must be one of {', '.join(SIDES)}.")
    if t.order_type not in ORDER_TYPES:
        raise OrderValidationError(f"Order type must be one of {', '.join(ORDER_TYPES)}.")
    if t.time_in_force not in TIFS:
        raise OrderValidationError(f"Time-in-force must be one of {', '.join(TIFS)}.")
    if t.order_class not in ORDER_CLASSES:
        raise OrderValidationError(
            f"Order class must be one of {', '.join(ORDER_CLASSES)} "
            "(multi-leg options are not supported here)."
        )

    # ── sizing: exactly one of qty / notional ────────────────────────────────
    if (t.qty is None) == (t.notional is None):
        raise OrderValidationError("Provide exactly one of quantity (shares) or notional ($).")
    if t.qty is not None:
        _require_positive(t.qty, "Quantity")
    if t.notional is not None:
        _require_positive(t.notional, "Notional")
        if t.order_type != "market" or t.order_class != "simple" or t.time_in_force != "day":
            raise OrderValidationError(
                "Notional (dollar) orders are only allowed for a simple MARKET order "
                "with time-in-force DAY. Use quantity for anything else."
            )
        if t.extended_hours:
            raise OrderValidationError("Notional orders cannot use extended hours.")

    # ── per-type price requirements ───────────────────────────────────────────
    if t.order_type == "limit":
        _require_positive(t.limit_price, "Limit price")
    elif t.order_type == "stop":
        _require_positive(t.stop_price, "Stop price")
    elif t.order_type == "stop_limit":
        _require_positive(t.stop_price, "Stop price")
        _require_positive(t.limit_price, "Limit price")
    elif t.order_type == "trailing_stop":
        if (t.trail_price is None) == (t.trail_percent is None):
            raise OrderValidationError(
                "Trailing stop needs exactly one of trail price ($) or trail percent (%)."
            )
        _require_positive(t.trail_price if t.trail_price is not None else t.trail_percent,
                          "Trail amount")
        if t.order_class != "simple":
            raise OrderValidationError("Trailing-stop orders must use the SIMPLE order class.")

    # extended hours is a limit-order-only feature on Alpaca.
    if t.extended_hours and t.order_type != "limit":
        raise OrderValidationError("Extended hours is only valid for LIMIT orders.")

    # ── advanced order classes (attached exit legs) ──────────────────────────
    if t.order_class in ("bracket", "oco", "oto"):
        if t.notional is not None:
            raise OrderValidationError(
                f"{t.order_class.upper()} orders require a share quantity, not notional."
            )
        if t.order_type not in ("market", "limit"):
            if t.order_class == "oco":
                pass  # handled below (OCO is forced to limit)
            else:
                raise OrderValidationError(
                    f"{t.order_class.upper()} entry must be a MARKET or LIMIT order."
                )

    if t.order_class == "bracket":
        _require_positive(t.take_profit_limit, "Take-profit limit price")
        _require_positive(t.stop_loss_stop, "Stop-loss stop price")
    elif t.order_class == "oto":
        if (t.take_profit_limit is None) and (t.stop_loss_stop is None):
            raise OrderValidationError(
                "OTO needs at least one attached leg: a take-profit or a stop-loss price."
            )
        if t.take_profit_limit is not None:
            _require_positive(t.take_profit_limit, "Take-profit limit price")
        if t.stop_loss_stop is not None:
            _require_positive(t.stop_loss_stop, "Stop-loss stop price")
    elif t.order_class == "oco":
        # OCO is an exit pair on an existing position: one limit + one stop,
        # submitted as a LIMIT base with both legs. Force the base type.
        t.order_type = "limit"
        _require_positive(t.take_profit_limit, "Take-profit limit price")
        _require_positive(t.stop_loss_stop, "Stop-loss stop price")

    if t.stop_loss_limit is not None:
        _require_positive(t.stop_loss_limit, "Stop-loss limit price")
        if t.stop_loss_stop is None:
            raise OrderValidationError("A stop-loss limit price also needs a stop-loss stop price.")

    return t


def build_order_request(ticket: OrderTicket):
    """Validate a ticket and return the matching alpaca-py order request object.

    Pure aside from the lazy alpaca-py import — no network, no client.
    """
    t = validate(ticket)

    from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
    from alpaca.trading.requests import (
        LimitOrderRequest,
        MarketOrderRequest,
        StopLimitOrderRequest,
        StopLossRequest,
        StopOrderRequest,
        TakeProfitRequest,
        TrailingStopOrderRequest,
    )

    side = OrderSide.BUY if t.side == "buy" else OrderSide.SELL
    tif = TimeInForce(t.time_in_force)
    order_class = OrderClass(t.order_class)

    common: dict = dict(
        symbol=t.symbol,
        side=side,
        time_in_force=tif,
        order_class=order_class,
        extended_hours=t.extended_hours,
    )
    if t.client_order_id:
        common["client_order_id"] = t.client_order_id
    if t.qty is not None:
        common["qty"] = t.qty
    if t.notional is not None:
        common["notional"] = t.notional

    # Attached exit legs for bracket / oco / oto.
    if t.take_profit_limit is not None:
        common["take_profit"] = TakeProfitRequest(limit_price=t.take_profit_limit)
    if t.stop_loss_stop is not None:
        sl: dict = {"stop_price": t.stop_loss_stop}
        if t.stop_loss_limit is not None:
            sl["limit_price"] = t.stop_loss_limit
        common["stop_loss"] = StopLossRequest(**sl)

    if t.order_type == "market":
        return MarketOrderRequest(**common)
    if t.order_type == "limit":
        return LimitOrderRequest(limit_price=t.limit_price, **common)
    if t.order_type == "stop":
        return StopOrderRequest(stop_price=t.stop_price, **common)
    if t.order_type == "stop_limit":
        return StopLimitOrderRequest(stop_price=t.stop_price, limit_price=t.limit_price, **common)
    if t.order_type == "trailing_stop":
        trail = {}
        if t.trail_price is not None:
            trail["trail_price"] = t.trail_price
        if t.trail_percent is not None:
            trail["trail_percent"] = t.trail_percent
        return TrailingStopOrderRequest(**trail, **common)

    raise OrderValidationError(f"Unsupported order type: {t.order_type}")  # unreachable


def describe(ticket: OrderTicket) -> str:
    """Short, human-readable one-liner for the confirmation log."""
    t = ticket.normalized()
    if t.lots is not None:
        size = f"{t.lots:g} lot{'s' if t.lots != 1 else ''} ({t.qty:g} sh)"
    elif t.qty is not None:
        size = f"{t.qty:g} sh"
    else:
        size = f"${t.notional:,.2f}"
    parts = [t.side.upper(), size, t.symbol, t.order_type.replace("_", "-")]
    if t.order_type in ("limit", "stop_limit") and t.limit_price:
        parts.append(f"lmt {t.limit_price:g}")
    if t.order_type in ("stop", "stop_limit") and t.stop_price:
        parts.append(f"stp {t.stop_price:g}")
    if t.order_type == "trailing_stop":
        parts.append(f"trail {t.trail_price:g}$" if t.trail_price is not None
                     else f"trail {t.trail_percent:g}%")
    parts.append(t.time_in_force.upper())
    if t.order_class != "simple":
        legs = []
        if t.take_profit_limit is not None:
            legs.append(f"TP {t.take_profit_limit:g}")
        if t.stop_loss_stop is not None:
            legs.append(f"SL {t.stop_loss_stop:g}")
        parts.append(f"{t.order_class.upper()}[{' '.join(legs)}]")
    if t.extended_hours:
        parts.append("ext")
    return " ".join(parts)
