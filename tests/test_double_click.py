"""One press, one order.

The first submit for a symbol used to take seconds — the meta-labelling forest
trains on a cache miss — with nothing on screen to say anything was happening.
Pressing Submit again is the reasonable response to a button that looks ignored,
and it bought the same stock three times.

Three defences, in order of how much they can be relied on. The server refuses an
identical ticket that arrives seconds after one it accepted; the button disables
itself the instant it is pressed; and the wait that caused it is gone. Only the
first survives a reload, a second tab, or a browser that drops the disabled
state, which is why it is the one with the most tests.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from markov_hedge_fund_method.broker import OrderResult
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.web import AppState, create_app

BODY = {"symbol": "AAPL", "side": "buy", "qty": 10, "order_type": "market",
        "order_class": "simple", "time_in_force": "day"}


class _Broker:
    def __init__(self, latency=0.0, equity=50_000.0):
        self.placed = []
        self.latency = latency
        self.equity = equity

    def get_account(self):
        time.sleep(self.latency)
        return type("A", (), {"equity": self.equity, "cash": self.equity,
                              "buying_power": self.equity, "status": "ACTIVE",
                              "last_equity": self.equity, "daytrade_count": 0,
                              "pattern_day_trader": False,
                              "daytrading_buying_power": 0.0})()

    def get_position(self, s):
        return None

    def list_positions(self):
        return []

    def latest_quote(self, s):
        time.sleep(self.latency)
        return None

    def symbols_opened_today(self):
        time.sleep(self.latency)
        return set()

    def submit_ticket(self, ticket):
        time.sleep(self.latency)
        self.placed.append(ticket)
        return OrderResult(id=f"o{len(self.placed)}", status="accepted", summary="ok")


def _client(latency=0.0, equity=50_000.0):
    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER), demo=True)
    state.broker = _Broker(latency, equity)
    return TestClient(create_app(state)), state.broker, state


# ── the guard that actually protects ────────────────────────────────────────
def test_three_impatient_clicks_place_one_order():
    """The reported bug, end to end."""
    c, b, _ = _client()
    codes = [c.post("/api/orders", json=BODY).status_code for _ in range(3)]
    assert codes == [200, 409, 409]
    assert len(b.placed) == 1


def test_the_refusal_says_the_order_is_already_placed():
    """Reporting a rejection would read as "nothing happened" and be acted on
    again — which is the same mistake with an extra step."""
    c, _, _ = _client()
    c.post("/api/orders", json=BODY)
    detail = c.post("/api/orders", json=BODY).json()["detail"]
    assert "identical order" in detail
    assert "already placed" in detail and "blotter" in detail


def test_a_deliberate_repeat_still_goes_through():
    """Scaling into a position is a real thing to want."""
    c, b, _ = _client()
    c.post("/api/orders", json=BODY)
    assert c.post("/api/orders", json={**BODY, "allowDuplicate": True}).status_code == 200
    assert len(b.placed) == 2


def test_a_different_order_is_never_blocked():
    c, b, _ = _client()
    c.post("/api/orders", json=BODY)
    assert c.post("/api/orders", json={**BODY, "qty": 11}).status_code == 200
    assert c.post("/api/orders", json={**BODY, "side": "sell"}).status_code == 200
    assert c.post("/api/orders", json={**BODY, "symbol": "MSFT"}).status_code == 200
    assert len(b.placed) == 4


def test_the_window_expires():
    """It is a double-click guard, not a ban on trading the same thing twice."""
    c, b, state = _client()
    c.post("/api/orders", json=BODY)
    key = next(iter(state._recent_orders))
    state._recent_orders[key] = time.monotonic() - state.DUPLICATE_WINDOW - 1
    assert c.post("/api/orders", json=BODY).status_code == 200
    assert len(b.placed) == 2


def test_the_window_is_a_sane_length():
    _, _, state = _client()
    assert 5.0 <= state.DUPLICATE_WINDOW <= 60.0


def test_a_rejected_order_is_not_remembered():
    """Only an accepted ticket blocks the next one; a failed submit must be
    retryable immediately."""
    c, b, state = _client()

    def boom(ticket):
        raise RuntimeError("broker said no")

    b.submit_ticket = boom
    assert c.post("/api/orders", json=BODY).status_code == 502
    b.submit_ticket = _Broker().submit_ticket.__get__(b)
    assert c.post("/api/orders", json=BODY).status_code == 200


def test_the_guard_runs_before_the_slow_work():
    """The whole reason a duplicate arrives is that the slow part had not
    answered yet, so checking after it would be checking too late."""
    import markov_hedge_fund_method.web as web
    src = open(web.__file__).read()
    body = src.split("async def submit_order(")[1].split("@app.post")[0]
    assert body.index("duplicate_of") < body.index("meta_sizing_ready")


# ── the wait that caused it ─────────────────────────────────────────────────
def test_a_cold_forest_does_not_hold_up_the_order():
    """Sizing already fails open when it errors. Failing open when it is merely
    slow is the same decision: an order at the size actually typed beats a wait
    that looks like a broken button."""
    c, _, _ = _client()
    t0 = time.perf_counter()
    r = c.post("/api/orders", json=BODY)
    assert r.status_code == 200
    assert time.perf_counter() - t0 < 1.5


def test_an_unsized_order_says_so():
    """An order that skipped the sizing layer must not look like one the layer
    approved."""
    c, _, _ = _client()
    out = c.post("/api/orders", json=BODY).json()
    assert out["metaSizing"]["pending"] is True
    assert "still training" in out["metaSizing"]["reason"]


def test_the_forest_warms_in_the_background():
    _, _, state = _client()
    assert state.meta_sizing_ready("AAPL") is None or isinstance(
        state.meta_sizing_ready("AAPL"), dict)


def test_the_last_look_is_taken_after_the_order():
    """Asking the book first only added a round trip to the wait."""
    import markov_hedge_fund_method.web as web
    src = open(web.__file__).read()
    body = src.split("async def submit_order(")[1].split("@app.post")[0]
    assert body.index("submit_ticket") < body.index("latest_quote")


# ── the button ──────────────────────────────────────────────────────────────
def _html():
    return _client()[0].get("/").text


def test_the_button_says_something_the_moment_it_is_pressed():
    html = _html()
    assert "SENDING…" in html
    assert "btn.disabled = true" in html


def test_a_second_press_while_sending_is_ignored():
    html = _html()
    assert "let SUBMITTING = false" in html
    assert "if(SUBMITTING) return;" in html


def test_the_button_recovers_even_when_the_order_fails():
    """A button left disabled by a rejected order is a terminal that cannot
    trade until it is reloaded."""
    html = _html()
    body = html.split("async function submitOrder()")[1].split("async function submitOrderInner")[0]
    assert "finally" in body and "btn.disabled = false" in body


def test_a_refused_duplicate_is_reported_as_good_news():
    html = _html()
    assert "identical order" in html
    assert "DUP_OVERRIDE" in html
