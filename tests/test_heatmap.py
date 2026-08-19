"""Heatmaps: the regime grid, the signal map and the correlation matrix."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.heatmap import correlation_matrix, regime_grid, signal_map
from markov_hedge_fund_method.web import AppState, create_app

IDX = pd.bdate_range("2020-01-01", periods=400)


def _series(vals):
    return pd.Series(vals, index=IDX[: len(vals)])


# ── regime grid ──────────────────────────────────────────────────────────────
def test_regime_grid_buckets_days_into_weeks():
    labels = {"AAA": _series([2] * 130)}
    closes = {"AAA": _series(np.linspace(100, 130, 130))}
    g = regime_grid(closes, labels, buckets=26, bucket_days=5)
    assert len(g["rows"]) == 1
    row = g["rows"][0]
    assert row["symbol"] == "AAA" and row["current"] == "bull"
    assert len(row["cells"]) == 26, "130 days at 5 a bucket is 26 columns"
    assert all(c["regime"] == "bull" and c["strength"] == 1.0 for c in row["cells"])
    assert len(g["dates"]) == 26


def test_a_contested_week_reports_lower_strength():
    labels = {"AAA": _series([2, 2, 2, 0, 0])}
    closes = {"AAA": _series(np.linspace(100, 105, 5))}
    cell = regime_grid(closes, labels, buckets=1, bucket_days=5)["rows"][0]["cells"][0]
    assert cell["regime"] == "bull" and cell["strength"] == 0.6


def test_regime_grid_picks_the_dominant_regime_per_bucket():
    labels = {"AAA": _series([0, 0, 0, 0, 2])}
    closes = {"AAA": _series(np.linspace(100, 105, 5))}
    cell = regime_grid(closes, labels, buckets=1, bucket_days=5)["rows"][0]["cells"][0]
    assert cell["regime"] == "bear"


def test_regime_grid_sorts_symbols_and_skips_empty_ones():
    labels = {"ZZZ": _series([1] * 20), "AAA": _series([2] * 20), "EMPTY": pd.Series(dtype=int)}
    closes = {k: _series(np.linspace(100, 110, 20)) for k in ("ZZZ", "AAA", "EMPTY")}
    g = regime_grid(closes, labels, buckets=4, bucket_days=5)
    assert [r["symbol"] for r in g["rows"]] == ["AAA", "ZZZ"]


def test_regime_grid_handles_nothing_at_all():
    assert regime_grid({}, {}) == {"dates": [], "rows": []}


# ── signal map ───────────────────────────────────────────────────────────────
def _st(sym, sig, regime="bull", conf=0.9, n=30, source="live"):
    return {"ticker": sym, "signal": sig, "regime": regime, "dataSource": source,
            "name": sym + " Inc", "signalStats": {"confidence": conf, "n": n}}


def test_signal_map_ranks_and_scores_percentiles():
    d = signal_map([_st("A", 0.1), _st("B", 0.5), _st("C", -0.3)])
    assert [c["symbol"] for c in d["cells"]] == ["B", "A", "C"]
    assert d["cells"][0]["percentile"] == 1.0
    assert d["cells"][-1]["percentile"] == 0.0
    assert d["median"] == 0.1


def test_signal_map_counts_direction():
    d = signal_map([_st("A", 0.4), _st("B", 0.3), _st("C", -0.4), _st("D", 0.0)])
    assert d["bullish"] == 2 and d["bearish"] == 1


def test_signal_map_flags_symbols_without_real_data():
    d = signal_map([_st("A", 0.2), _st("B", 0.1, source="synthetic (data unavailable)")])
    flags = {c["symbol"]: c["real"] for c in d["cells"]}
    assert flags["A"] is True and flags["B"] is False


def test_signal_map_skips_entries_with_no_signal():
    d = signal_map([{"ticker": "X"}, _st("A", 0.2)])
    assert [c["symbol"] for c in d["cells"]] == ["A"]


def test_signal_map_handles_a_single_symbol():
    d = signal_map([_st("A", 0.2)])
    assert d["cells"][0]["percentile"] == 1.0 and d["spread"] == 0.0


def test_signal_map_handles_nothing():
    assert signal_map([])["cells"] == []


# ── correlation ──────────────────────────────────────────────────────────────
def test_perfectly_correlated_symbols_report_one():
    base = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 200)) * 100
    closes = {"A": _series(base), "B": _series(base * 3.0)}
    d = correlation_matrix(closes)
    assert d["avgCorrelation"] == 1.0
    assert d["matrix"][0][1] == 1.0


def test_an_inverse_pair_reports_a_hedge():
    r = np.random.default_rng(1).normal(0, 0.01, 200)
    closes = {"A": _series(np.cumprod(1 + r) * 100),
              "B": _series(np.cumprod(1 - r) * 100)}
    d = correlation_matrix(closes)
    assert d["avgCorrelation"] < -0.9


def test_correlation_diagonal_is_one_and_matrix_is_square():
    rng = np.random.default_rng(2)
    closes = {c: _series(np.cumprod(1 + rng.normal(0, 0.01, 200)) * 100) for c in "ABCD"}
    d = correlation_matrix(closes)
    assert len(d["matrix"]) == len(d["symbols"]) == 4
    for i in range(4):
        assert d["matrix"][i][i] == 1.0
        assert len(d["matrix"][i]) == 4


def test_correlation_lists_the_tightest_pairs_first():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 200)
    closes = {"A": _series(np.cumprod(1 + r) * 100),
              "B": _series(np.cumprod(1 + r) * 100),          # identical to A
              "C": _series(np.cumprod(1 + rng.normal(0, 0.01, 200)) * 100)}
    d = correlation_matrix(closes)
    top = d["clusters"][0]
    assert {top["a"], top["b"]} == {"A", "B"}


def test_correlation_needs_two_symbols():
    d = correlation_matrix({"A": _series(np.linspace(100, 120, 200))})
    assert d["matrix"] == [] and d["avgCorrelation"] is None


def test_correlation_ignores_symbols_with_almost_no_history():
    rng = np.random.default_rng(4)
    closes = {"A": _series(np.cumprod(1 + rng.normal(0, 0.01, 200)) * 100),
              "B": _series(np.cumprod(1 + rng.normal(0, 0.01, 200)) * 100),
              "TINY": _series(np.linspace(100, 101, 5))}
    d = correlation_matrix(closes)
    assert "TINY" not in d["symbols"]


def test_correlation_caps_the_symbol_count():
    rng = np.random.default_rng(5)
    closes = {f"S{i}": _series(np.cumprod(1 + rng.normal(0, 0.01, 200)) * 100)
              for i in range(60)}
    d = correlation_matrix(closes, max_symbols=10)
    assert len(d["symbols"]) <= 10


# ── API ──────────────────────────────────────────────────────────────────────
def _client():
    return TestClient(create_app(AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)))


def test_heatmap_regime_endpoint():
    d = _client().get("/api/heatmap", params={"view": "regime", "watchlist": "SPY,QQQ,AAPL"}).json()
    assert d["view"] == "regime" and sorted(d["symbols"]) == ["AAPL", "QQQ", "SPY"]
    assert len(d["regime"]["rows"]) == 3
    assert d["regime"]["rows"][0]["cells"]


def test_heatmap_signal_endpoint():
    d = _client().get("/api/heatmap", params={"view": "signal", "watchlist": "SPY,QQQ"}).json()
    cells = d["signal"]["cells"]
    assert len(cells) == 2
    assert cells[0]["signal"] >= cells[1]["signal"], "ranked strongest first"
    assert "percentile" in cells[0]


def test_heatmap_correlation_endpoint():
    d = _client().get("/api/heatmap",
                      params={"view": "correlation", "watchlist": "SPY,QQQ,AAPL"}).json()
    assert len(d["correlation"]["symbols"]) == 3
    assert d["correlation"]["avgCorrelation"] is not None


def test_heatmap_defaults_to_the_regime_view():
    assert _client().get("/api/heatmap", params={"watchlist": "SPY,QQQ"}).json()["view"] == "regime"


def test_heatmap_accepts_a_scan_group():
    d = _client().get("/api/heatmap", params={"view": "regime", "scope": "crypto"}).json()
    assert "BTC-USD" in d["symbols"]


def test_heatmap_accepts_explicit_symbols():
    d = _client().get("/api/heatmap", params={"symbols": "TSLA,NVDA"}).json()
    assert sorted(d["symbols"]) == ["NVDA", "TSLA"]


def test_heatmap_caps_the_board_size():
    many = ",".join(f"S{i}" for i in range(200))
    d = _client().get("/api/heatmap", params={"symbols": many}).json()
    assert len(d["symbols"]) <= 60, "an unbounded board would be an unbounded request"


def test_heatmap_is_cached():
    c = _client()
    p = {"view": "regime", "watchlist": "SPY,QQQ,AAPL,MSFT"}
    t0 = time.monotonic(); c.get("/api/heatmap", params=p); first = time.monotonic() - t0
    t0 = time.monotonic(); c.get("/api/heatmap", params=p); second = time.monotonic() - t0
    assert second < max(first * 0.4, 0.05)


def test_heatmap_never_makes_the_dashboard_slower():
    """The regime and correlation views must not run the walk-forward engine."""
    c = _client()
    c.get("/api/heatmap", params={"view": "regime", "watchlist": "SPY,QQQ,AAPL"})
    t0 = time.monotonic()
    c.get("/api/state", params={"symbol": "SPY"})
    assert (time.monotonic() - t0) < 3.0


def test_index_has_the_heatmap_tab():
    html = _client().get("/").text
    assert "openHeat" in html and "🔥 HEATMAP" in html
    for view in ("regime", "signal", "correlation"):
        assert f'data-view="{view}"' in html
    assert "renderRegimeMap" in html and "renderCorrMap" in html
