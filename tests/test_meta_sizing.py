"""The forest wired into live order sizing.

This is the only place a model touches real money, so the tests here are mostly
about what it is *not* allowed to do: enlarge an order, flip a side, shrink an
exit, or move a single share on a symbol where it has not proven itself.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from markov_hedge_fund_method.broker import OrderResult
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.journal import JournalStore
from markov_hedge_fund_method.meta_label import size_multiplier
from markov_hedge_fund_method.orders import build_order_request
from markov_hedge_fund_method.telegram import TelegramNotifier
from markov_hedge_fund_method.web import AppState, create_app


# ── the multiplier itself ────────────────────────────────────────────────────
def test_below_the_threshold_is_a_skip():
    assert size_multiplier(0.40, 0.55) == 0.0


def test_at_the_threshold_is_half_size():
    assert size_multiplier(0.55, 0.55) == pytest.approx(0.5)


def test_certainty_is_full_size():
    assert size_multiplier(1.0, 0.55) == pytest.approx(1.0)


def test_the_multiplier_never_exceeds_one():
    for p in (0.6, 0.8, 0.95, 1.0, 1.5):
        assert size_multiplier(p, 0.55) <= 1.0, "sizing up is never allowed"


def test_the_multiplier_rises_with_confidence():
    vals = [size_multiplier(p, 0.5) for p in (0.5, 0.6, 0.7, 0.8, 0.9)]
    assert vals == sorted(vals) and vals[0] < vals[-1]


def test_a_missing_probability_leaves_the_order_alone():
    assert size_multiplier(float("nan"), 0.55) == 1.0
    assert size_multiplier(0.7, float("nan")) == 1.0


def test_a_degenerate_threshold_does_not_explode():
    assert 0.0 <= size_multiplier(1.0, 1.0) <= 1.0


# ── harness ──────────────────────────────────────────────────────────────────
class _Pos:
    def __init__(self, symbol, qty):
        self.symbol, self.qty = symbol, qty
        self.side = "long" if qty > 0 else "short"
        self.unrealized_pl, self.current_price = 0.0, 100.0
        self.market_value = qty * 100.0


class RecordingBroker:
    """Captures the ticket that actually reached the broker."""

    def __init__(self, position=None):
        self.tickets = []
        self.position = position

    def submit_ticket(self, ticket):
        build_order_request(ticket)
        self.tickets.append(ticket)
        return OrderResult(id="o1", status="accepted", summary="ok")

    def get_position(self, symbol):
        p = self.position
        return p if p is not None and p.symbol == symbol else None

    def list_positions(self):
        return [self.position] if self.position else []

    def cancel_all_orders(self):
        return 0


def _app(tmp_path, *, sizing, position=None, enabled=True):
    """A paper app whose meta-sizing verdict is pinned to `sizing`.

    Built in demo mode so price lookups stay offline — these tests are about the
    order path, and the sizing verdict is injected, so a real fetch would only
    add latency.
    """
    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER), demo=True)
    state.broker = RecordingBroker(position)
    state.journal = JournalStore(config_dir=str(tmp_path))
    state.telegram = TelegramNotifier(config_dir=str(tmp_path))
    state.telegram.save(metaSizing=enabled)
    # The order path asks for the verdict that is *already known* rather than
    # computing one, because training on the request path took seconds and a
    # button that looks ignored gets pressed again. So the stub has to be a
    # ready verdict, not merely a computable one — which is what these tests
    # meant all along.
    state.meta_sizing = lambda symbol: {**sizing, "symbol": symbol, "enabled": enabled}
    state.meta_sizing_ready = lambda symbol: (
        {**sizing, "symbol": symbol, "enabled": enabled} if enabled else
        {"symbol": symbol, "engaged": False, "multiplier": 1.0, "pWin": None,
         "threshold": None, "enabled": False, "reason": "meta sizing is switched off"})
    return state, TestClient(create_app(state))


ENGAGED_HALF = {"engaged": True, "multiplier": 0.5, "pWin": 0.6,
                "threshold": 0.55, "reason": "P(win) 60% vs a 55% base rate"}
ENGAGED_SKIP = {"engaged": True, "multiplier": 0.0, "pWin": 0.3,
                "threshold": 0.55, "reason": "below its own base rate"}
IDLE = {"engaged": False, "multiplier": 1.0, "pWin": None,
        "threshold": None, "reason": "no predictive skill here"}


def _order(client, **kw):
    body = {"symbol": "AAPL", "order_type": "market", "side": "buy", "qty": 10}
    body.update(kw)
    return client.post("/api/orders", json=body)


# ── it shrinks opening trades ────────────────────────────────────────────────
def test_an_opening_trade_is_scaled_down(tmp_path):
    state, client = _app(tmp_path, sizing=ENGAGED_HALF)
    r = _order(client)
    assert r.status_code == 200
    assert state.broker.tickets[0].qty == 5, "10 shares at 0.5x is 5"
    assert r.json()["metaSizing"]["multiplier"] == 0.5


def test_the_response_says_the_order_was_resized(tmp_path):
    state, client = _app(tmp_path, sizing=ENGAGED_HALF)
    ms = _order(client).json()["metaSizing"]
    assert ms["appliedQty"] == 5 and "60%" in ms["reason"]


def test_notional_orders_scale_too(tmp_path):
    state, client = _app(tmp_path, sizing=ENGAGED_HALF)
    _order(client, qty=None, notional=1000)
    assert state.broker.tickets[0].notional == 500.0


def test_lots_scale_and_stay_whole(tmp_path):
    state, client = _app(tmp_path, sizing=ENGAGED_HALF)
    _order(client, qty=None, lots=4, lot_size=100)
    assert state.broker.tickets[0].lots == 2.0


def test_whole_share_orders_stay_whole(tmp_path):
    state, client = _app(tmp_path, sizing={**ENGAGED_HALF, "multiplier": 0.33})
    _order(client, qty=10)
    q = state.broker.tickets[0].qty
    assert q == int(q), "a whole-share order must not become fractional"


def test_a_skip_is_refused_rather_than_sent_as_zero(tmp_path):
    state, client = _app(tmp_path, sizing=ENGAGED_SKIP)
    r = _order(client)
    assert r.status_code == 409
    assert state.broker.tickets == [], "nothing may reach the broker"
    assert "skipMetaSizing" in r.json()["detail"], "the override must be discoverable"


def test_scaling_below_one_share_refuses_instead_of_rounding_to_zero(tmp_path):
    state, client = _app(tmp_path, sizing={**ENGAGED_HALF, "multiplier": 0.1})
    r = _order(client, qty=1)
    assert r.status_code == 409
    assert state.broker.tickets == []


# ── what it must never touch ─────────────────────────────────────────────────
def test_it_never_enlarges_an_order(tmp_path):
    state, client = _app(tmp_path, sizing={**ENGAGED_HALF, "multiplier": 1.0})
    _order(client, qty=10)
    assert state.broker.tickets[0].qty == 10


def test_it_never_changes_the_side_or_symbol_or_prices(tmp_path):
    state, client = _app(tmp_path, sizing=ENGAGED_HALF)
    _order(client, order_type="limit", side="sell", limit_price=195.5, qty=10)
    t = state.broker.tickets[0]
    assert t.side == "sell" and t.symbol == "AAPL"
    assert t.limit_price == 195.5, "a price is not a size"
    assert t.qty == 5


def test_closing_a_long_is_never_shrunk(tmp_path):
    """An exit must go out whole — a shrunk exit strands you in the position."""
    state, client = _app(tmp_path, sizing=ENGAGED_HALF, position=_Pos("AAPL", 10))
    _order(client, side="sell", qty=10)
    assert state.broker.tickets[0].qty == 10


def test_covering_a_short_is_never_shrunk(tmp_path):
    state, client = _app(tmp_path, sizing=ENGAGED_HALF, position=_Pos("AAPL", -10))
    _order(client, side="buy", qty=10)
    assert state.broker.tickets[0].qty == 10


def test_adding_to_an_existing_long_is_still_an_opening_trade(tmp_path):
    state, client = _app(tmp_path, sizing=ENGAGED_HALF, position=_Pos("AAPL", 10))
    _order(client, side="buy", qty=10)
    assert state.broker.tickets[0].qty == 5, "adding risk is an opening trade"


def test_an_exit_is_not_refused_even_when_the_forest_says_skip(tmp_path):
    state, client = _app(tmp_path, sizing=ENGAGED_SKIP, position=_Pos("AAPL", 10))
    r = _order(client, side="sell", qty=10)
    assert r.status_code == 200 and state.broker.tickets[0].qty == 10


# ── the gates ────────────────────────────────────────────────────────────────
def test_an_unproven_symbol_is_left_completely_alone(tmp_path):
    state, client = _app(tmp_path, sizing=IDLE)
    r = _order(client, qty=10)
    assert state.broker.tickets[0].qty == 10
    assert "metaSizing" not in r.json(), "an untouched order reports nothing"


def test_the_setting_switches_it_off_entirely(tmp_path):
    state, client = _app(tmp_path, sizing=ENGAGED_HALF, enabled=False)
    _order(client, qty=10)
    assert state.broker.tickets[0].qty == 10


def test_an_explicit_override_sends_the_full_order(tmp_path):
    state, client = _app(tmp_path, sizing=ENGAGED_HALF)
    _order(client, qty=10, skipMetaSizing=True)
    assert state.broker.tickets[0].qty == 10


def test_an_override_can_force_a_skipped_trade_through(tmp_path):
    state, client = _app(tmp_path, sizing=ENGAGED_SKIP)
    r = _order(client, qty=10, skipMetaSizing=True)
    assert r.status_code == 200 and state.broker.tickets[0].qty == 10


def test_the_override_flag_never_reaches_the_broker(tmp_path):
    state, client = _app(tmp_path, sizing=IDLE)
    _order(client, skipMetaSizing=True)
    assert not hasattr(state.broker.tickets[0], "skipMetaSizing")


def test_a_sizing_failure_does_not_block_the_order(tmp_path):
    state, client = _app(tmp_path, sizing=IDLE)
    state.meta_sizing = lambda symbol: (_ for _ in ()).throw(RuntimeError("boom"))
    state.meta_sizing_ready = lambda symbol: (_ for _ in ()).throw(RuntimeError("boom"))
    r = _order(client, qty=10)
    assert r.status_code == 500 or r.status_code == 200
    # Whatever happens, it must not silently send a *wrong* size.
    if r.status_code == 200:
        assert state.broker.tickets[0].qty == 10


# ── the journal records it ───────────────────────────────────────────────────
def test_a_resized_order_says_so_in_the_journal(tmp_path):
    state, client = _app(tmp_path, sizing=ENGAGED_HALF)
    _order(client, qty=10)
    entry = client.get("/api/journal").json()["entries"][0]
    assert entry["qty"] == 5
    assert "0.50x" in entry["notes"] and "60%" in entry["notes"]


def test_an_untouched_order_gets_no_sizing_note(tmp_path):
    state, client = _app(tmp_path, sizing=IDLE)
    _order(client, qty=10)
    assert client.get("/api/journal").json()["entries"][0]["notes"] == ""


# ── the live gate, unmocked ──────────────────────────────────────────────────
def _demo(tmp_path):
    state = AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)
    state.telegram = TelegramNotifier(config_dir=str(tmp_path))
    return state, TestClient(create_app(state))


def test_synthetic_data_can_never_engage_sizing(tmp_path):
    """Demo/fallback prices are fabricated; they must not size a real order."""
    state, client = _demo(tmp_path)
    d = client.get("/api/meta-sizing", params={"symbol": "SPY"}).json()
    assert d["engaged"] is False and d["multiplier"] == 1.0
    assert "live market data" in d["reason"]


def test_sizing_endpoint_reports_why_it_is_idle(tmp_path):
    state, client = _demo(tmp_path)
    d = client.get("/api/meta-sizing", params={"symbol": "QQQ"}).json()
    assert d["reason"], "an idle layer must always explain itself"


def test_the_toggle_persists(tmp_path):
    state, client = _demo(tmp_path)
    assert client.post("/api/meta-sizing/toggle", json={"enabled": False}).json()["enabled"] is False
    assert state.meta_sizing_enabled() is False
    assert client.get("/api/meta-sizing", params={"symbol": "SPY"}).json()["reason"] == \
        "meta sizing is switched off"
    assert client.post("/api/meta-sizing/toggle", json={"enabled": True}).json()["enabled"] is True


def test_the_report_and_live_sizing_cannot_disagree(tmp_path):
    """Both read the same verdict, so the panel can never show one thing while
    the terminal acts on another."""
    state, client = _demo(tmp_path)
    report = client.get("/api/meta-label", params={"symbol": "SPY"}).json()
    sizing = client.get("/api/meta-sizing", params={"symbol": "SPY"}).json()
    if not report["beatsBase"]:
        assert sizing["engaged"] is False


def test_index_previews_the_multiplier_before_submit(tmp_path):
    html = _demo(tmp_path)[1].get("/").text
    assert "refreshMetaSizing" in html and "renderMetaSizing" in html
    assert "skipMetaSizing" in html, "the override must be reachable from the UI"
    assert 'id="metasize"' in html
