"""Switching views.

Flipping timeframe, or into the profile and back, asks the same question of the
same bars. Doing real work each time is what made those transitions feel slow,
and blanking the old view while the new one loads is what made them feel broken
even when they were fast. Both are pinned here.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.web import AppState, create_app


def _state():
    return AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)


def _client(state=None):
    return TestClient(create_app(state or _state()))


HTML = _client().get("/").text


# ── the server shares one set of bars ───────────────────────────────────────
def test_intraday_bars_are_fetched_once_for_all_three_views():
    """The chart, the profile and the timing read are three questions about one
    set of bars. Fetching them three times is three round trips for data the
    process is already holding."""
    state = _state()
    calls = {"n": 0}
    inner = state.intraday_for.__func__

    def counted(self, symbol, tf):
        key = (symbol.upper(), (tf or "1D").upper())
        if key not in self._intraday_cache:
            calls["n"] += 1
        return inner(self, symbol, tf)

    type(state).intraday_for = counted
    try:
        c = TestClient(create_app(state))
        c.get("/api/candles", params={"symbol": "SPY", "tf": "1D"})
        c.get("/api/tpo", params={"symbol": "SPY", "tf": "1D"})
        c.get("/api/timing", params={"symbol": "SPY", "tf": "1D"})
        assert calls["n"] == 1
    finally:
        type(state).intraday_for = inner


def test_the_cache_is_keyed_by_timeframe():
    """One key for all timeframes would serve 4-hour bars under a 1-day chart."""
    state = _state()
    state.intraday_for("SPY", "1D")
    state.intraday_for("SPY", "4H")
    assert ("SPY", "1D") in state._intraday_cache
    assert ("SPY", "4H") in state._intraday_cache


def test_the_cache_expires():
    state = _state()
    state.intraday_for("SPY", "1D")
    stamp, df, src = state._intraday_cache[("SPY", "1D")]
    state._intraday_cache[("SPY", "1D")] = (stamp - state.INTRADAY_TTL - 1, df, src)
    state.intraday_for("SPY", "1D")
    assert state._intraday_cache[("SPY", "1D")][0] > stamp - state.INTRADAY_TTL


def test_the_ttl_is_short_enough_to_stay_live():
    """Intraday bars go stale in minutes. This is a switching cache, not a
    substitute for fetching."""
    assert 10.0 <= _state().INTRADAY_TTL <= 120.0


def test_a_warm_view_does_no_work_at_all():
    """Originally a wall-clock comparison, which raced: an 80ms cold path and a
    15ms warm one invert whenever the process is scheduled out mid-measurement,
    so it failed under load and passed alone. Counting the fetches asks the same
    question and cannot be lost to a timing slip."""
    state = _state()
    calls = {"n": 0}
    inner = state.intraday_for.__func__

    def counted(self, symbol, tf):
        if (symbol.upper(), (tf or "1D").upper()) not in self._intraday_cache:
            calls["n"] += 1
        return inner(self, symbol, tf)

    type(state).intraday_for = counted
    try:
        c = TestClient(create_app(state))
        c.get("/api/candles?symbol=SPY&tf=1D")
        assert calls["n"] == 1
        c.get("/api/candles?symbol=SPY&tf=1D")
        assert calls["n"] == 1, "the warm view fetched again"
    finally:
        type(state).intraday_for = inner


# ── the client does not refetch what it has ─────────────────────────────────
def test_the_page_caches_intraday_payloads():
    assert "INTRADAY_CACHE" in HTML and "function tfKey" in HTML


def test_the_page_caches_profiles_by_every_setting():
    """Keyed by value area and period too, or flipping a setting back would
    refetch a profile the browser already has."""
    assert "TPO_CACHE" in HTML
    assert "[SYMBOL, tf, TPO_VA, TPO_PERIOD].join('|')" in HTML


def test_a_cached_view_paints_before_any_fetch():
    body = HTML.split("async function loadIntraday")[1].split("function setTF")[0]
    assert body.index("INTRADAY = hit.data; renderChart();") < body.index("await api(")


def test_a_fresh_cached_view_skips_the_fetch_entirely():
    body = HTML.split("async function loadIntraday")[1].split("function setTF")[0]
    assert "if(Date.now() - hit.t < FRESH_MS) return;" in body


def test_changing_symbol_drops_the_caches():
    """Otherwise the previous name's bars are painted under the new name."""
    assert "INTRADAY_CACHE.clear(); TPO_CACHE.clear();" in HTML


# ── nothing blanks mid-switch ───────────────────────────────────────────────
def test_a_drawn_profile_is_not_replaced_by_a_loading_line():
    """Swapping a rendered profile for a line of text is a flash of blank, not
    feedback — and it reads as breakage however fast the reload is."""
    body = HTML.split("async function loadTPO")[1].split("function renderTPO")[0]
    assert "else if(!TPO_DATA)" in body
    assert "Building the profile…" in body


def test_the_chart_dims_rather_than_clearing_while_loading():
    assert ".loading canvas.px{opacity:.35}" in HTML


def test_a_cached_timeframe_does_not_flash_the_loading_state():
    body = HTML.split("async function loadIntraday")[1].split("function setTF")[0]
    assert "if(card && !hit) card.classList.add('loading');" in body
