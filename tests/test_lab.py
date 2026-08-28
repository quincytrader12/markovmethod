"""The strategy lab.

Most of these test that the lab is hard to fool rather than that it searches
well, because searching well is easy and being fooled is the default. The test
that matters most is the last one in the first section: on a series with no edge
in it, the lab's own winner must fail its own holdout — and the lab must say so
rather than recommending it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from markov_hedge_fund_method import lab
from markov_hedge_fund_method import sharpe_stats as ss
from markov_hedge_fund_method.market_data import synthetic_close

CLOSE = synthetic_close(2520, seed=7)
RESULT = lab.search(CLOSE, symbol="TEST")


# ── the search ──────────────────────────────────────────────────────────────
def test_it_tries_singles_and_pairings():
    kinds = {t.kind for t in RESULT.trials}
    assert "single" in kinds
    assert kinds & set(("blend", "gate", "switch", "scale"))


def test_it_tries_enough_to_be_worth_deflating():
    assert RESULT.n_trials > 50


def test_every_trial_is_kept_including_the_losers():
    """The trial count is what the correction needs. Dropping the failures
    understates it, which is the same as overstating the winner."""
    sharpes = [t.track.sharpe for t in RESULT.trials]
    assert min(sharpes) < 0, "only the winners were kept"
    assert len(RESULT.trials) == RESULT.n_trials


# ── the correction ──────────────────────────────────────────────────────────
def test_ranking_is_by_deflated_sharpe_not_by_yield():
    dsrs = [t.dsr for t in RESULT.trials]
    assert dsrs == sorted(dsrs, reverse=True)


def test_the_bar_rises_with_the_number_of_attempts():
    """Ten tries and a thousand tries do not deserve the same benefit of the
    doubt."""
    v = RESULT.sharpe_variance
    assert ss.expected_max_sharpe(1000, v) > ss.expected_max_sharpe(10, v)


def test_deflation_is_harsher_than_the_undeflated_number():
    for t in RESULT.trials[:10]:
        assert t.dsr <= t.psr + 1e-9


def test_the_summary_states_the_luck_bar_in_plain_terms():
    text = lab.summarise(RESULT)
    assert "luck alone" in text and "That is the bar, not zero" in text


# ── the holdout, and the point of the whole thing ───────────────────────────
def test_the_holdout_is_never_searched():
    assert RESULT.holdout_bars > 0
    assert RESULT.searched_bars + RESULT.holdout_bars == len(CLOSE)


def test_only_a_few_candidates_are_shown_the_holdout():
    """Every look spends a little of it. Checking all 154 would turn the holdout
    into just another thing that was searched."""
    assert 0 < len(RESULT.holdout) <= lab.HOLDOUT_LOOKS


def test_the_holdout_number_is_reported_beside_the_searched_one():
    for h in RESULT.holdout:
        assert "searched" in h and "holdout" in h
        assert "decay" in h


def test_a_winner_that_fails_out_of_sample_is_marked_as_failing():
    """The finding this whole module exists for.

    On a random walk there is no edge to find, yet the search still produces a
    winner with a Sharpe above two and a deflated Sharpe calling it strong
    evidence — because a hundred and fifty variations of trend-following, fitted
    to one trending sample, will always agree with each other. The untouched
    quarter is what disagrees. If this test ever starts failing because the
    winner held up on synthetic data, something is leaking.
    """
    best = RESULT.holdout[0]
    assert best["searched"]["sharpe"] > 1.0, "the search did not find a 'winner'"
    assert best["holdout"]["sharpe"] < best["searched"]["sharpe"]
    assert best["heldUp"] is False
    assert "did not hold up" in lab.summarise(RESULT)


def test_deflation_alone_would_have_believed_it():
    """Stated as a test because it is the reason the holdout is not optional:
    the deflated Sharpe called this strong evidence and it was still wrong."""
    assert RESULT.trials[0].dsr > 0.9
    assert RESULT.holdout[0]["heldUp"] is False


# ── recommendations ─────────────────────────────────────────────────────────
def test_nothing_is_recommended_that_failed_its_holdout(tmp_path):
    lab.save(RESULT, str(tmp_path))
    assert lab.recommendations(str(tmp_path)) == []


def test_a_strategy_that_held_up_is_recommended(tmp_path):
    blob = RESULT.to_dict()
    blob["holdout"][0].update({"heldUp": True, "dsr": 0.97})
    with open(tmp_path / "lab_results.json", "w", encoding="utf-8") as fh:
        json.dump([blob], fh)
    recs = lab.recommendations(str(tmp_path))
    assert len(recs) == 1
    assert recs[0]["symbol"] == "TEST"
    assert recs[0]["holdoutSharpe"] is not None


def test_recommendations_carry_the_trial_count(tmp_path):
    """A recommendation without the number of attempts behind it is not a
    recommendation, it is a claim."""
    blob = RESULT.to_dict()
    blob["holdout"][0].update({"heldUp": True, "dsr": 0.97})
    with open(tmp_path / "lab_results.json", "w", encoding="utf-8") as fh:
        json.dump([blob], fh)
    assert lab.recommendations(str(tmp_path))[0]["nTrials"] == RESULT.n_trials


# ── persistence ─────────────────────────────────────────────────────────────
def test_a_run_is_saved_and_read_back(tmp_path):
    lab.save(RESULT, str(tmp_path))
    blob = lab.load(str(tmp_path), "TEST")
    assert blob and blob["nTrials"] == RESULT.n_trials
    assert len(blob["ranked"]) == RESULT.n_trials


def test_rerunning_a_symbol_replaces_its_record(tmp_path):
    lab.save(RESULT, str(tmp_path))
    lab.save(RESULT, str(tmp_path))
    assert len(lab.load_all(str(tmp_path))) == 1


def test_the_log_survives_a_corrupt_file(tmp_path):
    (tmp_path / "lab_results.json").write_text("{ not json")
    assert lab.load_all(str(tmp_path)) == []
    lab.save(RESULT, str(tmp_path))
    assert len(lab.load_all(str(tmp_path))) == 1


def test_saving_is_json_serialisable(tmp_path):
    lab.save(RESULT, str(tmp_path))
    json.loads((tmp_path / "lab_results.json").read_text())


# ── degenerate input ────────────────────────────────────────────────────────
def test_too_little_history_is_not_a_crash():
    r = lab.search(CLOSE.iloc[:40], symbol="SHORT")
    assert r.n_trials >= 0
    assert isinstance(lab.summarise(r), str)


def test_a_flat_series_produces_no_false_winner():
    """A market that never moves has nothing to find, so nothing may show a
    profit. The best available outcome is slightly negative rather than zero,
    because entering a position costs the spread even when the price then does
    nothing — which is the cost model working, not a bug."""
    flat = pd.Series(100.0, index=pd.bdate_range("2015-01-01", periods=1200))
    r = lab.search(flat, symbol="FLAT")
    assert all(t.track.sharpe <= 0.0 for t in r.trials)
    assert all(t.track.cagr <= 0.0 for t in r.trials)
