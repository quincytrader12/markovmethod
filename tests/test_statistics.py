"""Phase 1 statistics: matrix uncertainty, signal gating, PSR/DSR.

These are strictly additive — the existing matrix, signal and Sharpe values
must come out byte-identical, only accompanied by their uncertainty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markov_hedge_fund_method.market_data import synthetic_close
from markov_hedge_fund_method.regime import (build_transition_matrix, label_regimes,
                                             matrix_uncertainty, signal_confidence,
                                             transition_counts, wilson_interval,
                                             walk_forward_backtest)
from markov_hedge_fund_method.sharpe_stats import (deannualize, deflated_sharpe,
                                                   expected_max_sharpe,
                                                   probabilistic_sharpe, verdict)
from markov_hedge_fund_method.webstate import market_state


# ── the existing engine must be unchanged ────────────────────────────────────
def test_matrix_math_is_unchanged():
    close = synthetic_close(seed=7)
    labels = label_regimes(close)
    P = build_transition_matrix(labels, stride=20)
    counts = transition_counts(labels, stride=20)
    rows = counts.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1.0
    assert np.allclose(P, counts / rows)          # counts are the same evidence
    assert np.allclose(P.sum(axis=1), 1.0)        # still a valid stochastic matrix


# ── Wilson intervals ─────────────────────────────────────────────────────────
def test_wilson_interval_bounds_and_width():
    lo, hi = wilson_interval(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    # more evidence must mean a tighter interval around the same rate
    lo_small, hi_small = wilson_interval(9, 18)
    lo_big, hi_big = wilson_interval(100, 200)
    assert (hi_big - lo_big) < (hi_small - lo_small)
    assert wilson_interval(0, 0) == (0.0, 1.0)    # no data -> no knowledge


def test_matrix_uncertainty_brackets_the_estimate():
    close = synthetic_close(seed=3)
    labels = label_regimes(close)
    counts = transition_counts(labels, stride=20)
    P = build_transition_matrix(labels, stride=20)
    u = matrix_uncertainty(counts)
    for i in range(3):
        assert u["n"][i] == counts[i].sum()
        for j in range(3):
            assert u["lo"][i][j] <= P[i][j] <= u["hi"][i][j]


# ── signal gating ────────────────────────────────────────────────────────────
def test_signal_confidence_scales_with_evidence():
    """Same probabilities, more observations -> more confidence."""
    small = np.array([[0, 0, 0], [0, 0, 0], [2, 3, 10]], dtype=float)
    large = small * 20
    lo = signal_confidence(small, 2)
    hi = signal_confidence(large, 2)
    assert abs(lo["signal"] - hi["signal"]) < 1e-9      # identical point estimate
    assert hi["confidence"] > lo["confidence"]          # but far more certain
    assert hi["reliable"] and not lo["reliable"]


def test_signal_confidence_handles_empty_row():
    s = signal_confidence(np.zeros((3, 3)), 1)
    assert s["n"] == 0 and s["reliable"] is False


# ── PSR / DSR ────────────────────────────────────────────────────────────────
def test_psr_rises_with_sample_length():
    sr = deannualize(1.5)
    short = probabilistic_sharpe(sr, 60)
    long = probabilistic_sharpe(sr, 2000)
    assert 0.0 <= short < long <= 1.0


def test_psr_punished_by_fat_tails_and_negative_skew():
    sr = deannualize(1.5)
    clean = probabilistic_sharpe(sr, 1000, skew=0.0, kurtosis=3.0)
    nasty = probabilistic_sharpe(sr, 1000, skew=-1.5, kurtosis=9.0)
    assert nasty < clean


def test_expected_max_sharpe_grows_with_trials():
    """Trying more strategies raises the bar luck alone can clear."""
    assert expected_max_sharpe(2, 0.01) < expected_max_sharpe(100, 0.01)
    assert expected_max_sharpe(100, 0.0) == 0.0


def test_deflated_is_never_kinder_than_psr():
    sr, n = deannualize(2.0), 1000
    psr = probabilistic_sharpe(sr, n)
    dsr = deflated_sharpe(sr, n, n_trials=100, sharpe_variance=0.01)
    assert dsr <= psr


def test_verdict_wording():
    assert verdict(0.99) == "strong evidence"
    assert verdict(0.85) == "some evidence"
    assert verdict(0.10) == "likely luck"


# ── wiring into the payload ──────────────────────────────────────────────────
def test_payload_carries_uncertainty():
    st = market_state(synthetic_close(seed=5), "SPY")
    ci = st["matrixCI"]
    assert len(ci["n"]) == 3 and len(ci["lo"]) == 3
    for i in range(3):
        for j in range(3):
            assert ci["lo"][i][j] <= st["matrix"][i][j] <= ci["hi"][i][j]
    ss = st["signalStats"]
    assert 0.0 <= ss["confidence"] <= 1.0 and isinstance(ss["reliable"], bool)
    m = st["metrics"]
    assert 0.0 <= m["psr"] <= 1.0 and m["psrVerdict"]
    assert m["nObs"] > 0


def test_backtest_reports_distribution_shape():
    close = synthetic_close(seed=11)
    r = walk_forward_backtest(close, label_regimes(close))
    assert r["n_obs"] > 0 and r["kurtosis"] > 0
    assert isinstance(r["skew"], float)


# ── scanner deflation ────────────────────────────────────────────────────────
def _row(sym, sharpe, n_obs=1000):
    return {"symbol": sym, "sharpe": sharpe, "nObs": n_obs, "skew": 0.0,
            "kurtosis": 3.0, "score": 70, "daysInRegime": 3, "winRate": 0.55}


def test_deflate_penalises_wide_scans():
    """The same Sharpe is less impressive when it was the best of many."""
    from markov_hedge_fund_method.scanner import deflate
    few = deflate([_row("A", 1.8), _row("B", 1.2)])
    many = deflate([_row("A", 1.8)] + [_row(f"S{i}", 1.0 + i * 0.02) for i in range(60)])
    a_few = next(r for r in few if r["symbol"] == "A")["dsr"]
    a_many = next(r for r in many if r["symbol"] == "A")["dsr"]
    assert a_many < a_few, "scanning more names must lower confidence in the winner"


def test_deflate_marks_missing_backtests():
    from markov_hedge_fund_method.scanner import deflate
    out = deflate([{"symbol": "X", "sharpe": None, "nObs": 0}])
    assert out[0]["dsr"] is None and out[0]["edgeVerdict"] is None


def test_proven_filter_uses_deflated_probability():
    from markov_hedge_fund_method.scanner import rank
    rows = [dict(_row("GOOD", 2.5), dsr=0.99), dict(_row("MEH", 1.1), dsr=0.60)]
    out = rank(rows, top=10, proven_only=True)
    assert [r["symbol"] for r in out["results"]] == ["GOOD"]


# ── plain-English readout ────────────────────────────────────────────────────
def test_plain_summary_is_readable_and_honest():
    from markov_hedge_fund_method.webstate import plain_summary
    strong = plain_summary("SPY", "bull", 4, 58.0, 0.99, True, 26)
    assert "SPY Bull" in strong and "day 4" in strong
    assert "58%" in strong and "26 comparable" in strong
    assert "supports the read" in strong

    weak = plain_summary("XYZ", "bear", 2, 44.0, 0.60, False, 12)
    assert "XYZ Bear" in weak and "too thin to trade on" in weak

    mid = plain_summary("A", "bull", 3, 55.0, 0.85, False, 25)
    assert "short of significance" in mid

    empty = plain_summary("ABC", "sideways", 0, 50.0, 0.0, False, 0)
    assert "Insufficient history" in empty


def test_payload_includes_plain_summary():
    st = market_state(synthetic_close(seed=3), "SPY")
    assert st["plainSummary"].startswith("SPY ")
    assert "comparable setups" in st["plainSummary"]
