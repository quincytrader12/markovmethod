"""The backtest engine and the strategy library.

Two properties matter more than any metric here, because breaking either one
produces a *better looking* result rather than an error: positions must be lagged
before they meet a return, and costs must be charged on turnover. Both have a
test that fails loudly if they are ever quietly removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markov_hedge_fund_method import backtest as bt
from markov_hedge_fund_method import strategies as st
from markov_hedge_fund_method.market_data import synthetic_close

CLOSE = synthetic_close(2520, seed=7)


def rising(n=300, step=0.01):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * (1 + step) ** np.arange(n), index=idx)


# ── the lag ─────────────────────────────────────────────────────────────────
def test_a_position_cannot_trade_the_bar_that_produced_it():
    """The most common way a backtest invents money, and it is invisible: the
    equity curve simply looks good."""
    c = rising(50)
    # A position that is only long on the single bar that rises most. Without
    # the lag it captures that bar; with it, it captures the one after.
    pos = pd.Series(0.0, index=c.index)
    pos.iloc[10] = 1.0
    t = bt.run(pos, c, cost_bps=0.0)
    assert t.returns.iloc[10] == pytest.approx(0.0), "it traded on its own signal bar"
    assert t.returns.iloc[11] > 0


def test_perfect_foresight_is_worth_more_than_the_real_thing():
    """A sanity check on the lag: a strategy that knows tomorrow beats one that
    only knows today, and by a lot. If they score the same, nothing is lagged."""
    c = CLOSE
    tomorrow = np.sign(c.pct_change().shift(-1)).fillna(0.0)
    cheat = bt.run(tomorrow, c, cost_bps=0.0)
    honest = bt.run(np.sign(c.pct_change()).fillna(0.0), c, cost_bps=0.0)
    assert cheat.sharpe > honest.sharpe + 3.0


# ── costs ───────────────────────────────────────────────────────────────────
def test_costs_destroy_a_daily_coin_flip():
    """A search with no cost model reliably crowns the highest-turnover noise in
    the library, because flipping every day looks like edge until it pays a
    spread."""
    rng = np.random.default_rng(0)
    flip = pd.Series(rng.choice([-1.0, 1.0], len(CLOSE)), index=CLOSE.index)
    t = bt.run(flip, CLOSE)
    assert t.turnover > 200
    assert t.sharpe < t.gross_sharpe - 0.5
    assert t.cost_drag > 0.05


def test_a_patient_strategy_barely_pays_anything():
    t = bt.run(pd.Series(1.0, index=CLOSE.index), CLOSE)
    assert t.turnover < 1.0
    assert t.cost_drag < 0.001


def test_costs_scale_with_the_rate_charged():
    rng = np.random.default_rng(1)
    flip = pd.Series(rng.choice([-1.0, 1.0], len(CLOSE)), index=CLOSE.index)
    cheap = bt.run(flip, CLOSE, cost_bps=1.0)
    dear = bt.run(flip, CLOSE, cost_bps=10.0)
    assert dear.cost_drag > cheap.cost_drag * 5


def test_there_is_no_cost_free_default():
    assert bt.DEFAULT_COST_BPS > 0


# ── metrics ─────────────────────────────────────────────────────────────────
def test_a_rising_market_held_long_makes_money():
    t = bt.run(pd.Series(1.0, index=rising().index), rising(), cost_bps=0.0)
    assert t.cagr > 0 and t.max_drawdown == pytest.approx(0.0, abs=1e-9)


def test_drawdown_is_negative_or_zero():
    t = bt.run(pd.Series(1.0, index=CLOSE.index), CLOSE)
    assert t.max_drawdown <= 0


def test_an_empty_position_is_flat_not_a_crash():
    t = bt.run(pd.Series(0.0, index=CLOSE.index), CLOSE)
    assert t.sharpe == 0.0 and t.exposure == 0.0


def test_the_summary_is_json_safe():
    import json
    json.dumps(bt.run(pd.Series(1.0, index=CLOSE.index), CLOSE).summary())


# ── the holdout split ───────────────────────────────────────────────────────
def test_the_split_does_not_overlap():
    a, b = bt.split(CLOSE, 0.25)
    assert len(a) + len(b) == len(CLOSE)
    assert a.index[-1] < b.index[0]


def test_the_holdout_is_the_end_of_history_not_the_middle():
    """Holding out an interior slice leaves the strategy fitted on both sides of
    it, which is not the question anyone is asking."""
    a, b = bt.split(CLOSE, 0.25)
    assert b.index[-1] == CLOSE.index[-1]


def test_the_holdout_is_bounded():
    for frac in (-1.0, 0.0, 0.9, 5.0):
        a, b = bt.split(CLOSE, frac)
        assert len(a) > 0 and len(b) > 0


# ── causality ───────────────────────────────────────────────────────────────
def test_every_strategy_in_the_library_is_causal():
    """A centred window, a full-series normalisation or a stray backfill breaks
    this silently, and the only symptom is a better equity curve."""
    for name, fn in st.library().items():
        assert bt.is_causal(fn, CLOSE), f"{name} sees the future"


def test_the_causality_check_catches_a_leak():
    """The check has to be able to fail, or it is decoration."""
    def leaky(close):
        c = pd.Series(close)
        return pd.Series(np.sign(c.pct_change().shift(-1)), index=c.index).fillna(0.0)
    assert not bt.is_causal(leaky, CLOSE)


def test_a_full_sample_normalisation_is_caught():
    def leaky(close):
        c = pd.Series(close).astype(float)
        return ((c - c.mean()) / c.std()).clip(-1, 1)     # mean over all history
    assert not bt.is_causal(leaky, CLOSE)


# ── the library ─────────────────────────────────────────────────────────────
def test_the_library_has_the_standard_families():
    lib = st.library()
    joined = " ".join(lib)
    for family in ("momentum", "ma_cross", "breakout", "reversion", "vol"):
        assert family in joined


def test_buy_and_hold_is_in_the_library():
    """The bar every strategy has to beat. A search that cannot see it will
    happily recommend something worse."""
    assert "buy_and_hold" in st.library()


def test_every_position_stays_within_one_unit():
    for name, fn in st.library().items():
        pos = pd.Series(fn(CLOSE)).dropna()
        assert pos.abs().max() <= 1.0 + 1e-9, f"{name} leveraged itself"


def test_no_strategy_returns_nothing():
    for name, fn in st.library().items():
        assert len(pd.Series(fn(CLOSE))) == len(CLOSE), name


# ── pairing ─────────────────────────────────────────────────────────────────
def test_a_gate_only_trades_when_both_agree():
    a = lambda c: pd.Series(1.0, index=pd.Series(c).index)      # noqa: E731
    b = lambda c: pd.Series(-1.0, index=pd.Series(c).index)     # noqa: E731
    assert pd.Series(st.gate(a, b)(CLOSE)).abs().sum() == 0
    assert pd.Series(st.gate(a, a)(CLOSE)).abs().sum() > 0


def test_a_blend_sits_between_its_parts():
    a = lambda c: pd.Series(1.0, index=pd.Series(c).index)      # noqa: E731
    b = lambda c: pd.Series(0.0, index=pd.Series(c).index)      # noqa: E731
    assert pd.Series(st.blend(a, b, 0.5)(CLOSE)).iloc[-1] == pytest.approx(0.5)


def test_every_pairing_of_causal_parts_is_causal():
    lib = st.library()
    a, b = lib["momentum_126"], lib["ma_cross_50_200"]
    for kind, combiner in st.PAIRINGS.items():
        assert bt.is_causal(combiner(a, b), CLOSE), kind


# ── walk-forward selection ──────────────────────────────────────────────────
def test_walk_forward_chooses_from_the_past_only():
    lib = {k: st.library()[k] for k in ("momentum_252", "buy_and_hold")}
    track, picks = bt.walk_forward_choice(lib, CLOSE, train=504, step=252)
    assert picks and all("chose" in p for p in picks)
    assert track.n_obs == len(CLOSE)


def test_walk_forward_survives_a_broken_candidate():
    def broken(close):
        raise RuntimeError("bad strategy")
    lib = {"ok": st.library()["momentum_252"], "broken": broken}
    track, picks = bt.walk_forward_choice(lib, CLOSE, train=504, step=252)
    assert all(p["chose"] == "ok" for p in picks)


def test_walk_forward_with_no_candidates_is_flat():
    track, picks = bt.walk_forward_choice({}, CLOSE)
    assert track.sharpe == 0.0 and picks == []
