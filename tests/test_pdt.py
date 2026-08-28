"""Pattern-day-trader guard and the last-look quote.

The guard has two ways to fail and they pull in opposite directions. Too loud
and it nags a swing trader who will never day trade, which trains them to click
past it — so the case that matters most is the one where it says nothing. Too
quiet and it lets through the single order that costs ninety days of account
access.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from markov_hedge_fund_method import pdt
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.pdt import (
    EQUITY_FLOOR,
    FREE_DAY_TRADES,
    evaluate,
    exempt,
    remaining_trades,
    summary,
    would_be_day_trade,
)
from markov_hedge_fund_method.web import AppState, create_app

POOR = 12_000.0
RICH = 40_000.0


# ── who the rule binds ──────────────────────────────────────────────────────
def test_above_the_floor_the_rule_does_not_apply():
    assert exempt(RICH) and not exempt(POOR)
    assert exempt(EQUITY_FLOOR), "the floor itself is enough"


def test_remaining_is_none_rather_than_a_number_when_exempt():
    """None is not 'many left' — rendering it as a count would be a lie."""
    assert remaining_trades(0, RICH) is None
    assert remaining_trades(1, POOR) == FREE_DAY_TRADES - 1


def test_the_count_never_goes_negative():
    assert remaining_trades(9, POOR) == 0


# ── what counts as a day trade ──────────────────────────────────────────────
def test_selling_something_bought_today_is_a_day_trade():
    assert would_be_day_trade("AAPL", "sell", {"AAPL"})


def test_selling_something_held_from_before_is_not():
    """The whole reason this guard is quiet for a swing trader."""
    assert not would_be_day_trade("AAPL", "sell", {"MSFT"})
    assert not would_be_day_trade("AAPL", "sell", set())


def test_buying_more_of_it_is_not_a_day_trade():
    """Adding to a position opened today closes nothing."""
    assert not would_be_day_trade("AAPL", "buy", {"AAPL"})


def test_the_symbol_match_ignores_case_and_padding():
    assert would_be_day_trade(" aapl ", "SELL", {"AAPL"})


# ── the verdict ─────────────────────────────────────────────────────────────
def _v(**kw):
    base = dict(symbol="AAPL", side="sell", equity=POOR, used=0,
                opened_today={"AAPL"})
    return evaluate(**{**base, **kw})


def test_a_multi_day_hold_is_never_mentioned():
    """The common case for this user. Silence is the correct output."""
    v = _v(opened_today=set())
    assert v.allowed and not v.is_day_trade and v.severity == "ok"
    assert v.headline == ""


def test_a_rich_account_is_never_mentioned_either():
    v = _v(equity=RICH, used=3)
    assert v.allowed and v.severity == "ok" and v.headline == ""


def test_the_fourth_day_trade_is_blocked():
    v = _v(used=3)
    assert v.blocking and v.severity == "block"
    assert "4th day trade" in v.headline
    assert "90 days" in v.detail


def test_the_block_names_the_cheaper_alternative():
    """Refusing without saying what to do instead is just an obstacle."""
    assert "overnight" in _v(used=3).detail


def test_the_third_warns_but_allows():
    v = _v(used=2)
    assert v.allowed and v.severity == "warn"
    assert "Last day trade" in v.headline


def test_the_earlier_ones_only_inform():
    v = _v(used=0)
    assert v.allowed and v.severity == "info" and v.is_day_trade


def test_an_explicit_closing_flag_overrides_a_missing_history():
    """A broker hiccup returning no fills must not switch the guard off."""
    v = _v(opened_today=set(), closing=True, used=3)
    assert v.blocking, "an empty history silently disabled the guard"


def test_closing_false_is_respected():
    assert _v(closing=False, used=3).allowed


# ── the standing counter ────────────────────────────────────────────────────
def test_the_summary_counts_for_a_small_account():
    s = summary(POOR, 2)
    assert s["applies"] and s["remaining"] == 1
    assert "2 of 3" in s["text"]


def test_the_summary_says_the_rule_is_moot_for_a_big_one():
    s = summary(RICH, 2)
    assert s["applies"] is False and s["remaining"] is None
    assert "does not apply" in s["text"]


# ── the API ─────────────────────────────────────────────────────────────────
def _client():
    return TestClient(create_app(AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD),
                                          demo=True)))


def test_the_counter_endpoint_survives_no_broker():
    d = _client().get("/api/daytrades").json()
    assert d["connected"] is False and d["applies"] is False


def test_the_guard_is_skipped_when_no_broker_is_connected():
    state = AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)
    assert state.pdt_check(type("T", (), {"symbol": "AAPL", "side": "sell"})()) is None


# ── the last-look price check ───────────────────────────────────────────────
def test_a_marketable_limit_is_called_out():
    """A buy limit at or through the ask fills at the touch, which is not what
    someone typing a limit price usually meant."""
    from markov_hedge_fund_method.web import create_app as _c  # noqa: F401

    app = create_app(AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True))
    drift = _drift(app)
    t = type("T", (), {"limit_price": 101.0, "side": "buy"})()
    q = {"bid": 99.0, "ask": 100.0, "mid": 99.5}
    out = drift(t, q)
    assert out["throughTheBook"] is True and "marketable" in out["note"]


def test_a_resting_limit_reports_its_distance():
    app = create_app(AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True))
    drift = _drift(app)
    t = type("T", (), {"limit_price": 95.0, "side": "buy"})()
    out = drift(t, {"bid": 99.0, "ask": 100.0, "mid": 99.5})
    assert out["throughTheBook"] is False
    assert out["driftPct"] == pytest.approx(-4.523, abs=0.01)
    assert "below the mid" in out["note"]


def test_a_market_order_has_no_drift_to_report():
    app = create_app(AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True))
    drift = _drift(app)
    t = type("T", (), {"limit_price": None, "side": "buy"})()
    assert drift(t, {"bid": 99.0, "ask": 100.0, "mid": 99.5}) is None


def _drift(app):
    """Reach the closure the order endpoint uses, so the test exercises the
    real function rather than a copy of its arithmetic."""
    for route in app.routes:
        fn = getattr(route, "endpoint", None)
        if fn is not None and getattr(fn, "__name__", "") == "submit_order":
            for cell in fn.__closure__ or ():
                v = cell.cell_contents
                if callable(v) and getattr(v, "__name__", "") == "_limit_drift":
                    return v
    raise AssertionError("_limit_drift not reachable from the order endpoint")


# ── end to end through the order endpoint ───────────────────────────────────
class _Broker:
    """Enough of a broker to exercise the guard, and nothing more."""

    def __init__(self, equity=POOR, used=0, opened=(), quote=None):
        self.account = type("A", (), {
            "equity": equity, "cash": equity, "buying_power": equity,
            "status": "ACTIVE", "last_equity": equity,
            "daytrade_count": used, "pattern_day_trader": False,
            "daytrading_buying_power": 0.0})()
        self._opened = set(opened)
        self._quote = quote
        self.submitted = []

    def get_account(self):
        return self.account

    def symbols_opened_today(self):
        return set(self._opened)

    def latest_quote(self, symbol):
        return self._quote

    def list_positions(self):
        return []

    def submit_ticket(self, ticket):
        from markov_hedge_fund_method.broker import OrderResult
        self.submitted.append(ticket)
        return OrderResult(id="x1", status="accepted", summary="ok")


def _app(broker):
    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER), demo=True)
    state.broker = broker
    return TestClient(create_app(state)), broker


def _order(client, **kw):
    body = {"symbol": "AAPL", "side": "sell", "qty": 10, "type": "market",
            "order_class": "simple", "time_in_force": "day"}
    return client.post("/api/orders", json={**body, **kw})


def test_the_fourth_day_trade_is_refused_by_the_endpoint():
    c, b = _app(_Broker(used=3, opened={"AAPL"}))
    r = _order(c)
    assert r.status_code == 409
    assert "4th day trade" in r.json()["detail"]
    assert b.submitted == [], "the order went out anyway"


def test_the_refusal_can_be_overridden_deliberately():
    """A guard with no way past it becomes a reason to stop using the terminal."""
    c, b = _app(_Broker(used=3, opened={"AAPL"}))
    assert _order(c, skipPdtGuard=True).status_code == 200
    assert len(b.submitted) == 1


def test_a_swing_exit_is_not_touched():
    """Nothing bought today, so no day trade and nothing to say."""
    c, b = _app(_Broker(used=3, opened=set()))
    r = _order(c)
    assert r.status_code == 200
    assert "dayTrade" not in r.json()
    assert len(b.submitted) == 1


def test_an_earlier_day_trade_is_reported_but_allowed():
    c, b = _app(_Broker(used=1, opened={"AAPL"}))
    r = _order(c)
    assert r.status_code == 200
    assert r.json()["dayTrade"]["severity"] in ("info", "warn")
    assert len(b.submitted) == 1


def test_a_rich_account_is_never_stopped():
    c, b = _app(_Broker(equity=RICH, used=9, opened={"AAPL"}))
    r = _order(c)
    assert r.status_code == 200 and "dayTrade" not in r.json()


def test_the_quote_rides_back_with_the_fill():
    q = {"symbol": "AAPL", "bid": 99.0, "ask": 100.0, "mid": 99.5,
         "spread": 1.0, "spreadPct": 1.005, "asOf": ""}
    c, _ = _app(_Broker(quote=q))
    body = c.post("/api/orders", json={
        "symbol": "AAPL", "side": "buy", "qty": 10, "type": "limit",
        "limit_price": 101.0, "order_class": "simple",
        "time_in_force": "day"}).json()
    assert body["quote"]["mid"] == 99.5
    assert body["priceCheck"]["throughTheBook"] is True


def test_a_missing_quote_does_not_stop_the_order():
    """The check is advisory. Losing it must cost nothing but the check."""
    c, b = _app(_Broker(quote=None))
    r = c.post("/api/orders", json={"symbol": "AAPL", "side": "buy", "qty": 1,
                                    "type": "market", "order_class": "simple",
                                    "time_in_force": "day"})
    assert r.status_code == 200 and "quote" not in r.json()
    assert len(b.submitted) == 1


def test_a_broken_guard_does_not_take_the_order_down():
    class Broken(_Broker):
        def get_account(self):
            raise RuntimeError("alpaca down")

    c, b = _app(Broken(used=3, opened={"AAPL"}))
    assert _order(c).status_code == 200
    assert len(b.submitted) == 1


# ── the UI ──────────────────────────────────────────────────────────────────
def _html():
    return _client().get("/").text


def test_the_counter_chip_exists_and_starts_hidden():
    """A counter that is always on screen is a counter nobody reads. It appears
    only for an account the rule actually binds."""
    html = _html()
    assert 'id="chip-pdt"' in html and 'style="display:none"' in html
    assert "function loadDayTrades" in html


def test_the_chip_hides_itself_when_the_rule_does_not_apply():
    html = _html()
    body = html.split("async function loadDayTrades")[1].split("/* ---------- entry timing")[0]
    assert "!d.applies" in body and "display='none'" in body


def test_a_refusal_arms_the_override_rather_than_losing_the_ticket():
    """Printing the refusal and dropping the order means retyping it, which is
    how a safety net becomes a reason to stop using the terminal."""
    html = _html()
    assert "PDT_OVERRIDE=true;" in html
    assert "press Submit again" in html


def test_the_override_is_not_sticky():
    """It must arm for one order only, or the guard is off from then on."""
    html = _html()
    assert "META_OVERRIDE=false; PDT_OVERRIDE=false;" in html
    body = html.split("function pickSymbol")[1][:600] if "function pickSymbol" in html else html
    assert "PDT_OVERRIDE=false;" in html


def test_a_marketable_limit_is_reported_after_the_fill():
    html = _html()
    assert "through the book" in html and "priceCheck" in html
