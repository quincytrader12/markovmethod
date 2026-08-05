"""Verifies OrderTicket -> alpaca-py request covers the full order taxonomy.

Requires alpaca-py (the `terminal` extra). Run: pytest -q
"""

from __future__ import annotations

import pytest

from markov_hedge_fund_method.orders import (
    ORDER_TYPES,
    TIFS,
    OrderTicket,
    OrderValidationError,
    build_order_request,
    describe,
    validate,
)


def _v(t) -> str:
    """enum-or-str value."""
    return getattr(t, "value", t)


# ── every order TYPE builds and round-trips its type ─────────────────────────
@pytest.mark.parametrize("otype,extra", [
    ("market", {}),
    ("limit", {"limit_price": 100.0}),
    ("stop", {"stop_price": 90.0}),
    ("stop_limit", {"stop_price": 90.0, "limit_price": 89.5}),
    ("trailing_stop", {"trail_percent": 2.5}),
])
def test_every_order_type_builds(otype, extra):
    req = build_order_request(OrderTicket(symbol="spy", side="buy", order_type=otype,
                                          qty=10, **extra))
    assert _v(req.type) == otype
    assert req.symbol == "SPY"
    assert _v(req.side) == "buy"
    assert float(req.qty) == 10


def test_all_order_types_covered():
    assert set(ORDER_TYPES) == {"market", "limit", "stop", "stop_limit", "trailing_stop"}


# ── every TIME-IN-FORCE round-trips ──────────────────────────────────────────
@pytest.mark.parametrize("tif", TIFS)
def test_every_tif(tif):
    req = build_order_request(OrderTicket(symbol="AAPL", side="sell", order_type="market",
                                          qty=1, time_in_force=tif))
    assert _v(req.time_in_force) == tif


# ── sizing rules ─────────────────────────────────────────────────────────────
def test_notional_market_day_ok():
    req = build_order_request(OrderTicket(symbol="SPY", side="buy", qty=None, notional=500.0))
    assert float(req.notional) == 500.0


def test_notional_requires_market_day():
    with pytest.raises(OrderValidationError):
        build_order_request(OrderTicket(symbol="SPY", side="buy", order_type="limit",
                                        notional=500.0, limit_price=100.0))
    with pytest.raises(OrderValidationError):
        build_order_request(OrderTicket(symbol="SPY", side="buy", order_type="market",
                                        notional=500.0, time_in_force="gtc"))


def test_exactly_one_of_qty_or_notional():
    with pytest.raises(OrderValidationError):
        build_order_request(OrderTicket(symbol="SPY", side="buy"))  # neither
    with pytest.raises(OrderValidationError):
        build_order_request(OrderTicket(symbol="SPY", side="buy", qty=1, notional=1))  # both


def test_negative_size_rejected():
    with pytest.raises(OrderValidationError):
        build_order_request(OrderTicket(symbol="SPY", side="buy", qty=-5))


# ── per-type price requirements ──────────────────────────────────────────────
def test_limit_needs_price():
    with pytest.raises(OrderValidationError):
        build_order_request(OrderTicket(symbol="SPY", side="buy", order_type="limit", qty=1))


def test_stop_needs_price():
    with pytest.raises(OrderValidationError):
        build_order_request(OrderTicket(symbol="SPY", side="buy", order_type="stop", qty=1))


def test_stop_limit_needs_both():
    with pytest.raises(OrderValidationError):
        build_order_request(OrderTicket(symbol="SPY", side="buy", order_type="stop_limit",
                                        qty=1, stop_price=90))


def test_trailing_stop_exactly_one_offset():
    with pytest.raises(OrderValidationError):  # none
        build_order_request(OrderTicket(symbol="SPY", side="sell", order_type="trailing_stop",
                                        qty=1))
    with pytest.raises(OrderValidationError):  # both
        build_order_request(OrderTicket(symbol="SPY", side="sell", order_type="trailing_stop",
                                        qty=1, trail_price=2, trail_percent=2))
    req = build_order_request(OrderTicket(symbol="SPY", side="sell", order_type="trailing_stop",
                                          qty=1, trail_price=2.5))
    assert float(req.trail_price) == 2.5


def test_extended_hours_limit_only():
    with pytest.raises(OrderValidationError):
        build_order_request(OrderTicket(symbol="SPY", side="buy", order_type="market",
                                        qty=1, extended_hours=True))
    req = build_order_request(OrderTicket(symbol="SPY", side="buy", order_type="limit",
                                          qty=1, limit_price=100, extended_hours=True))
    assert req.extended_hours is True


# ── advanced order CLASSES ───────────────────────────────────────────────────
def test_bracket_builds_both_legs():
    req = build_order_request(OrderTicket(symbol="SPY", side="buy", order_type="market", qty=10,
                                          order_class="bracket", time_in_force="gtc",
                                          take_profit_limit=110, stop_loss_stop=95))
    assert _v(req.order_class) == "bracket"
    assert float(req.take_profit.limit_price) == 110
    assert float(req.stop_loss.stop_price) == 95


def test_bracket_requires_both_legs():
    with pytest.raises(OrderValidationError):
        build_order_request(OrderTicket(symbol="SPY", side="buy", order_type="market", qty=10,
                                        order_class="bracket", take_profit_limit=110))


def test_oco_forces_limit_base_with_both_legs():
    req = build_order_request(OrderTicket(symbol="SPY", side="sell", order_type="market", qty=10,
                                          order_class="oco", time_in_force="gtc",
                                          take_profit_limit=120, stop_loss_stop=95))
    assert _v(req.order_class) == "oco"
    assert _v(req.type) == "limit"  # OCO is coerced to a limit base
    assert float(req.take_profit.limit_price) == 120
    assert float(req.stop_loss.stop_price) == 95


def test_oto_one_leg_ok_none_fails():
    req = build_order_request(OrderTicket(symbol="SPY", side="buy", order_type="limit", qty=10,
                                          limit_price=100, order_class="oto", time_in_force="gtc",
                                          stop_loss_stop=90))
    assert _v(req.order_class) == "oto"
    assert float(req.stop_loss.stop_price) == 90
    with pytest.raises(OrderValidationError):
        build_order_request(OrderTicket(symbol="SPY", side="buy", order_type="limit", qty=10,
                                        limit_price=100, order_class="oto"))


def test_advanced_class_rejects_notional():
    with pytest.raises(OrderValidationError):
        build_order_request(OrderTicket(symbol="SPY", side="buy", order_type="market",
                                        notional=500, order_class="bracket",
                                        take_profit_limit=110, stop_loss_stop=95))


def test_stop_loss_limit_needs_stop():
    with pytest.raises(OrderValidationError):
        build_order_request(OrderTicket(symbol="SPY", side="buy", order_type="market", qty=1,
                                        order_class="oto", stop_loss_limit=95))


def test_bad_enums_rejected():
    for bad in (
        OrderTicket(symbol="SPY", side="hold", qty=1),
        OrderTicket(symbol="SPY", side="buy", order_type="iceberg", qty=1),
        OrderTicket(symbol="SPY", side="buy", qty=1, time_in_force="week"),
        OrderTicket(symbol="SPY", side="buy", qty=1, order_class="mleg"),
        OrderTicket(symbol="", side="buy", qty=1),
    ):
        with pytest.raises(OrderValidationError):
            validate(bad)


def test_describe_is_readable():
    s = describe(OrderTicket(symbol="spy", side="buy", order_type="stop_limit", qty=3,
                             stop_price=90, limit_price=89, time_in_force="gtc"))
    assert "BUY" in s and "SPY" in s and "stop-limit" in s and "GTC" in s
