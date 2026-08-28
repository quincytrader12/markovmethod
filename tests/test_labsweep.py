"""Sweeping the lab across many symbols.

Searching one symbol is a search of a hundred and fifty-four. Searching fifty is
a search of seven and a half thousand, and every test here is about that
sentence: the pooled correction, the two ways breadth flatters a leaderboard,
and the survivorship a watchlist carries by construction.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from markov_hedge_fund_method import lab
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.labsweep import LabSweep
from markov_hedge_fund_method.market_data import synthetic_close, synthetic_ohlc
from markov_hedge_fund_method.web import AppState, create_app

SEEDS = {"AAA": 1, "BBB": 2, "CCC": 3}


def _state(live=True):
    st = AppState(Settings(ticker="AAA", mode=Mode.BACKTEST), demo=True)
    if live:
        st.ohlc_for = lambda sym: (synthetic_ohlc(1600, seed=SEEDS.get(sym, 0)), "live")
    return st


def _run(state, symbols=tuple(SEEDS), timeout=90.0):
    sweep = LabSweep(state, config_dir=None)
    import tempfile
    sweep._dir = tempfile.mkdtemp()
    sweep.start(list(symbols))
    deadline = time.time() + timeout
    while sweep.running and time.time() < deadline:
        time.sleep(0.2)
    return sweep


RESULTS = [lab.search(synthetic_close(1600, seed=i), symbol=f"S{i}") for i in (1, 2, 3)]
POOLED = lab.pool(RESULTS)


# ── the pooled correction ───────────────────────────────────────────────────
def test_the_trial_count_is_the_whole_sweep():
    """Three symbols is one search of three times the size, not three searches."""
    assert POOLED["nTrials"] == sum(r.n_trials for r in RESULTS)


def test_pooling_is_harsher_than_judging_a_symbol_alone():
    """The correction this module exists for. A winner chosen across every
    symbol must clear the bar set by every attempt, not by its own symbol's."""
    harsher = [r for r in POOLED["ranked"] if r["dsr"] < r["symbolDsr"]]
    assert harsher, "pooling let every winner keep its per-symbol confidence"


def test_the_luck_bar_rises_with_the_sweep():
    """More symbols, more attempts, a higher score reachable by luck alone."""
    small = lab.pool(RESULTS[:1])
    assert POOLED["luckBar"] >= small["luckBar"]


def test_both_deflated_sharpes_are_reported():
    """The gap between them is how much of the confidence came from having
    looked at only one symbol, so hiding either hides the point."""
    row = POOLED["ranked"][0]
    assert "dsr" in row and "symbolDsr" in row


def test_pooling_a_single_result_is_not_a_crash():
    assert lab.pool([])["nTrials"] == 0


# ── breadth ─────────────────────────────────────────────────────────────────
def test_breadth_counts_bets_not_rows():
    b = lab.breadth(RESULTS)
    assert b["effectiveBets"] <= b["n"]


def test_identical_strategies_collapse_to_one_bet():
    """Momentum on five correlated names is one bet held five times, and it
    presents itself as five independent confirmations."""
    idx = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(0)
    shared = pd.Series(rng.normal(0, 0.01, 400), index=idx)

    class _T:
        def __init__(self, r):
            self.returns = r

    class _R:
        def __init__(self, sym, r):
            self.symbol = sym
            self.trials = [type("X", (), {"name": "same", "track": _T(r)})()]

    same = [_R(f"S{i}", shared.copy()) for i in range(6)]
    b = lab.breadth(same)
    assert b["avgCorrelation"] > 0.95
    assert b["effectiveBets"] < 1.5
    assert "one idea repeated" in b["note"]


def test_uncorrelated_strategies_count_separately():
    idx = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(1)

    class _T:
        def __init__(self, r):
            self.returns = r

    class _R:
        def __init__(self, sym, r):
            self.symbol = sym
            self.trials = [type("X", (), {"name": "diff", "track": _T(r)})()]

    indep = [_R(f"S{i}", pd.Series(rng.normal(0, 0.01, 400), index=idx))
             for i in range(6)]
    b = lab.breadth(indep)
    assert abs(b["avgCorrelation"]) < 0.2
    assert b["effectiveBets"] > 3


# ── survivorship ────────────────────────────────────────────────────────────
def test_every_trial_carries_its_excess_over_buy_and_hold():
    """A watchlist is a list of names someone already liked. A strategy that
    made money on one of them may have done nothing but be long something that
    went up."""
    for row in POOLED["ranked"][:5]:
        assert "excess" in row


def test_the_benchmark_is_the_same_period_as_the_search():
    r = RESULTS[0]
    assert isinstance(r.benchmark_sharpe, float)
    top = r.trials[0]
    assert top.excess == pytest.approx(top.track.sharpe - r.benchmark_sharpe, abs=1e-6)


def test_holdout_rows_say_whether_they_beat_owning_the_symbol():
    for h in RESULTS[0].holdout:
        assert "beatBuyAndHold" in h


# ── the sweep runner ────────────────────────────────────────────────────────
def test_a_sweep_searches_every_symbol():
    sweep = _run(_state())
    assert sorted(sweep.done) == sorted(SEEDS)
    assert sweep.pooled["nTrials"] > 0


def test_synthetic_data_is_refused_rather_than_pooled():
    """A search over invented prices describes the generator, not a market, and
    pooling it would corrupt the bar every other symbol is judged against."""
    sweep = _run(_state(live=False))
    assert sweep.done == []
    assert all("no real price data" in s["why"] for s in sweep.skipped)


def test_a_symbol_without_history_is_skipped_not_fatal():
    state = _state()
    state.ohlc_for = lambda sym: ((synthetic_ohlc(1600, seed=1), "live") if sym != "BBB"
                                  else (pd.DataFrame(), "live"))
    sweep = _run(state)
    assert "BBB" in [s["symbol"] for s in sweep.skipped]
    assert "AAA" in sweep.done


def test_a_broken_symbol_does_not_stop_the_sweep():
    state = _state()
    real = state.ohlc_for

    def flaky(sym):
        if sym == "BBB":
            raise RuntimeError("feed down")
        return real(sym)

    state.ohlc_for = flaky
    sweep = _run(state)
    assert "AAA" in sweep.done and "CCC" in sweep.done


def test_a_sweep_can_be_stopped():
    state = _state()
    sweep = LabSweep(state)
    import tempfile
    sweep._dir = tempfile.mkdtemp()
    sweep.start(list(SEEDS))
    sweep.stop()
    deadline = time.time() + 30
    while sweep.running and time.time() < deadline:
        time.sleep(0.1)
    assert not sweep.running


def test_each_symbol_is_saved_as_it_finishes():
    """Stopping half way must still leave everything the sweep learned."""
    sweep = _run(_state())
    stored = lab.load_all(sweep.config_dir())
    assert len(stored) >= len(sweep.done)


def test_the_sweep_stands_aside_for_the_user():
    """Scoring is CPU-bound work under one interpreter lock — the same lesson
    the market sweep was measured for."""
    from markov_hedge_fund_method import labsweep
    assert labsweep.WORKERS == 1
    state = _state()
    state.note_activity()
    sweep = LabSweep(state)
    t0 = time.perf_counter()
    sweep.yield_to_user(timeout=5.0)
    assert time.perf_counter() - t0 >= labsweep.IDLE_BEFORE_WORK - 0.4


# ── the API ─────────────────────────────────────────────────────────────────
def _client():
    return TestClient(create_app(AppState(Settings(ticker="SPY", mode=Mode.BACKTEST),
                                          demo=True)))


def test_the_sweep_endpoints_answer():
    c = _client()
    assert c.get("/api/lab/sweep").json()["running"] is False
    started = c.post("/api/lab/sweep/start", json={"symbols": ["SPY"]}).json()
    assert started["symbols"] == ["SPY"]
    c.post("/api/lab/sweep/stop")


def test_the_symbol_list_is_capped():
    """Past a point the trial count deflates every winner into insignificance
    anyway, so the extra hours buy nothing."""
    c = _client()
    many = [f"S{i}" for i in range(200)]
    started = c.post("/api/lab/sweep/start", json={"symbols": many}).json()
    assert len(started["symbols"]) <= 60
    c.post("/api/lab/sweep/stop")


def test_the_page_offers_the_sweep():
    html = _client().get("/").text
    assert "function toggleLabSweep" in html and "SWEEP WATCHLIST" in html


def test_the_ui_shows_both_deflated_sharpes():
    html = _client().get("/").text
    assert "Pooled DSR" in html and "Own DSR" in html


def test_the_ui_refuses_a_survivor_that_trailed_buy_and_hold():
    html = _client().get("/").text
    assert "trailed simply owning the symbol" in html


def test_the_gate_reads_the_pooled_deflated_sharpe():
    """A candidate found in a sweep of eleven hundred trials must answer to
    eleven hundred, not to the hundred and fifty made on its own symbol —
    otherwise the pooling is a display that nothing acts on."""
    sweep = _run(_state())
    for row in sweep.survivors():
        if row.get("pooledDsr") is not None:
            assert row["dsr"] == row["pooledDsr"]


def test_the_button_promises_only_what_the_server_accepts():
    """A survivor can beat buy-and-hold and still be refused on the pooled DSR.
    A button offering to integrate something the server then rejects is worse
    than one that explains why."""
    html = _client().get("/").text
    body = html.split("function renderLabSweep")[1].split("function ")[0]
    assert "okDsr" in body and "okSharpe" in body
    assert "beat && okDsr && okSharpe" in body
