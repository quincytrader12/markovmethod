"""Keeping the terminal responsive while the sweep works.

The sweep is the lowest-priority thing in the process: it exists so the research
is already done when the user arrives, and it is worth nothing if it makes the
terminal stutter while they are there. Scoring is CPU-bound pandas, so every
sweep thread contends with request handling for the interpreter — measured at
four times the page's response time before this.

The fix rests on knowing when someone is actually present, which the server
cannot infer from traffic: the page's own heartbeats made it look busy in bursts
and a person reading without clicking looked idle. Real input events are the
signal.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from markov_hedge_fund_method import sweep as sw
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.sweep import MarketSweep
from markov_hedge_fund_method.web import AppState, create_app


def _state():
    return AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)


HTML = TestClient(create_app(_state())).get("/").text


# ── the beacon ──────────────────────────────────────────────────────────────
def test_the_beacon_marks_the_user_present():
    state = _state()
    c = TestClient(create_app(state))
    time.sleep(0.05)
    assert c.post("/api/active").status_code == 204
    assert state.idle_for() < 0.5


def test_a_heartbeat_does_not_pretend_to_be_a_person():
    """Counting timer polls as interaction is what made the signal useless: the
    terminal believed the user was busy in bursts and absent the rest of the
    time, which is neither of the two things the sweep needed to know."""
    state = _state()
    c = TestClient(create_app(state))
    c.post("/api/active")
    time.sleep(0.4)
    before = state.idle_for()
    for path in ("/api/state?symbol=SPY", "/api/portfolio?symbol=SPY",
                 "/api/alerts", "/api/health", "/api/news?symbol=SPY",
                 "/api/daytrades", "/api/timing?symbol=SPY"):
        c.get(path)
    assert state.idle_for() >= before, "a heartbeat reset the idle clock"


def test_the_things_only_a_person_can_do_still_count():
    state = _state()
    c = TestClient(create_app(state))
    time.sleep(0.3)
    c.get("/api/tpo?symbol=SPY")
    assert state.idle_for() < 0.3, "opening the profile is someone acting"


# ── the sweep stands aside ──────────────────────────────────────────────────
def test_the_sweep_waits_after_a_click():
    state = _state()
    sweep = MarketSweep(state)
    state.note_activity()
    t0 = time.perf_counter()
    sweep.yield_to_user(timeout=10.0)
    assert time.perf_counter() - t0 >= sw.IDLE_BEFORE_WORK - 0.5


def test_the_sweep_works_freely_when_nobody_is_there():
    state = _state()
    sweep = MarketSweep(state)
    state._last_activity = time.monotonic() - 3600
    t0 = time.perf_counter()
    sweep.yield_to_user(timeout=10.0)
    assert time.perf_counter() - t0 < 0.5


def test_a_busy_session_cannot_starve_the_sweep_forever():
    """Without the timeout, someone leaving the terminal open and active all day
    would mean the research never gets done."""
    state = _state()
    sweep = MarketSweep(state)
    state.note_activity()
    t0 = time.perf_counter()
    sweep.yield_to_user(timeout=0.6)
    assert time.perf_counter() - t0 < 2.0


def test_the_back_off_is_long_enough_to_be_felt():
    """A second and a half gave a clicking user that much quiet and then took
    the machine back mid-thought."""
    assert sw.IDLE_BEFORE_WORK >= 4.0


def test_the_slice_and_worker_count_favour_the_user():
    """Both were tuned for sweep throughput and cost about four times the page's
    response time. The sweep has all day; the person using it does not."""
    assert sw.SLICE <= 4
    assert sw.WORKERS == 1


# ── the page side ───────────────────────────────────────────────────────────
def test_the_page_reports_real_input_not_traffic():
    assert "function pingActive" in HTML
    for ev in ("pointerdown", "keydown", "wheel"):
        assert f"'{ev}'" in HTML


def test_the_beacon_is_throttled():
    """One ping per couple of seconds; a drag would otherwise be a flood."""
    body = HTML.split("function pingActive")[1][:400]
    assert "LAST_ACTIVE_PING < 2000" in body


def test_the_beacon_does_not_collide_with_the_watchlist_highlighter():
    """There was already a markActive(sym) that highlights the selected row.
    Declaring a second one hoisted over the first, so the listener called the
    highlighter with no argument — no beacon was ever sent, and every click
    quietly cleared the watchlist highlight instead."""
    assert HTML.count("function markActive") == 1
    assert "window.addEventListener(ev, pingActive" in HTML


def test_a_hidden_tab_is_nobody():
    assert "visibilitychange" in HTML


def test_the_beacon_survives_a_failed_send():
    body = HTML.split("function pingActive")[1][:400]
    assert ".catch(()=>{})" in body
