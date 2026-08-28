"""Meta-labelling forest: labels, purging, the skill gate, and the promise
that none of it changes a single existing number."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from markov_hedge_fund_method import meta_label as M
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.har import vol_forecast_for
from markov_hedge_fund_method.market_data import synthetic_ohlc
from markov_hedge_fund_method.regime import label_regimes, walk_forward_backtest
from markov_hedge_fund_method.web import AppState, create_app


@pytest.fixture(scope="module")
def series():
    ohlc = synthetic_ohlc(2520, seed=3)
    close = ohlc["Close"]
    return close, ohlc, label_regimes(close), vol_forecast_for(close, ohlc)


# ── triple-barrier labelling ─────────────────────────────────────────────────
def test_triple_barrier_labels_are_binary_and_resolve_forward():
    px = pd.Series(np.linspace(100, 130, 200), index=pd.bdate_range("2020-01-01", periods=200))
    side = pd.Series(1.0, index=px.index)
    bars = M.triple_barrier(px, side, horizon=5)
    assert set(bars["label"].unique()) <= {0, 1}
    assert (bars["t1"] > np.arange(len(bars))).all(), "a trade must resolve after it opens"
    # A monotonically rising series with long signals should win almost always.
    assert bars["label"].mean() > 0.9


def test_a_falling_market_loses_long_signals():
    px = pd.Series(np.linspace(130, 100, 200), index=pd.bdate_range("2020-01-01", periods=200))
    bars = M.triple_barrier(px, pd.Series(1.0, index=px.index), horizon=5)
    assert bars["label"].mean() < 0.1


def test_short_signals_win_when_price_falls():
    px = pd.Series(np.linspace(130, 100, 200), index=pd.bdate_range("2020-01-01", periods=200))
    bars = M.triple_barrier(px, pd.Series(-1.0, index=px.index), horizon=5)
    assert bars["label"].mean() > 0.9, "a short into a downtrend is a win"


def test_flat_signal_days_are_dropped():
    px = pd.Series(np.linspace(100, 120, 100), index=pd.bdate_range("2020-01-01", periods=100))
    side = pd.Series([0.0] * 50 + [1.0] * 50, index=px.index)
    bars = M.triple_barrier(px, side, horizon=5)
    assert len(bars) <= 50 and (bars["side"] != 0).all()


def test_barriers_widen_with_forecast_volatility():
    px = pd.Series(100 * np.exp(np.cumsum(np.full(300, 0.001))),
                   index=pd.bdate_range("2020-01-01", periods=300))
    side = pd.Series(1.0, index=px.index)
    calm = M.triple_barrier(px, side, sigma=pd.Series(0.01, index=px.index), horizon=5)
    wild = M.triple_barrier(px, side, sigma=pd.Series(0.80, index=px.index), horizon=5)
    # Same drift, same horizon: a calm forecast sets a near barrier that gets
    # touched quickly, a violent one sets a barrier the drift cannot reach, so
    # the trade runs to the time limit instead.
    held_calm = (calm["t1"].to_numpy() - np.arange(len(calm))).mean()
    held_wild = (wild["t1"].to_numpy() - np.arange(len(wild))).mean()
    assert held_calm < held_wild
    assert held_wild == pytest.approx(5.0, abs=0.1), "wide barriers time out"


# ── overlap accounting ───────────────────────────────────────────────────────
def test_uniqueness_weights_penalise_overlap():
    solo = M.uniqueness_weights(np.array([0, 10]), np.array([4, 14]), 20)
    assert np.allclose(solo, 1.0), "non-overlapping labels are fully unique"

    stacked = M.uniqueness_weights(np.arange(5), np.arange(5) + 4, 20)
    assert stacked.max() < 1.0
    assert stacked.mean() < solo.mean()


def test_purged_kfold_removes_overlapping_training_samples():
    n = 100
    t0 = np.arange(n)
    t1 = t0 + 5
    splits = M.purged_kfold_splits(t0, t1, n_splits=5, embargo_pct=0.02)
    assert len(splits) == 5
    for train, test in splits:
        assert not set(train) & set(test)
        lo, hi = int(t0[test[0]]), int(t1[test[-1]])
        for i in train:
            assert t1[i] < lo or t0[i] > hi, "a training label overlaps the test fold"


def test_purged_kfold_is_smaller_than_plain_kfold():
    n = 200
    t0, t1 = np.arange(n), np.arange(n) + 10
    purged = M.purged_kfold_splits(t0, t1, n_splits=5)
    for train, test in purged:
        assert len(train) < n - len(test), "purging must actually drop samples"


def test_auc_ranks_correctly():
    y = np.array([0, 0, 1, 1])
    assert M.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert M.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert M.auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert M.auc(np.array([1, 1]), np.array([0.5, 0.6])) == 0.5  # one class only


# ── the forest itself ────────────────────────────────────────────────────────
def test_tree_learns_a_clean_split():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    y = (X[:, 1] > 0).astype(float)
    tree = M.DecisionTree(max_depth=2, min_samples_leaf=20).fit(X, y)
    p = tree.predict_proba(X)
    assert ((p > 0.5) == (y == 1)).mean() > 0.95
    assert int(np.argmax(tree.importances_)) == 1, "it should key on the real feature"


def test_forest_beats_chance_on_a_learnable_problem():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 5))
    y = (X[:, 0] + 0.5 * X[:, 2] + rng.normal(0, 0.5, 800) > 0).astype(float)
    f = M.RandomForest(n_estimators=20, max_depth=3, min_samples_leaf=30).fit(X[:600], y[:600])
    assert M.auc(y[600:], f.predict_proba(X[600:])) > 0.75


def test_forest_finds_no_skill_in_pure_noise():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(800, 5))
    y = rng.integers(0, 2, 800).astype(float)
    f = M.RandomForest(n_estimators=20, max_depth=3, min_samples_leaf=50).fit(X[:600], y[:600])
    assert abs(M.auc(y[600:], f.predict_proba(X[600:])) - 0.5) < 0.12


def test_forest_respects_the_leaf_minimum():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] > 0).astype(float)
    tree = M.DecisionTree(max_depth=4, min_samples_leaf=50).fit(X, y)

    def depth(node):
        if node is None or node.feature < 0:
            return 0
        return 1 + max(depth(node.left), depth(node.right))

    assert depth(tree.root) <= 1, "120 rows cannot support two levels at leaf>=50"


def test_forest_is_deterministic():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 4))
    y = (X[:, 2] > 0).astype(float)
    a = M.RandomForest(n_estimators=10, min_samples_leaf=30).fit(X, y).predict_proba(X)
    b = M.RandomForest(n_estimators=10, min_samples_leaf=30).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b), "the same data must give the same model"


def test_forest_declines_to_fit_a_tiny_sample():
    f = M.RandomForest(min_samples_leaf=50).fit(np.zeros((10, 3)), np.zeros(10))
    assert f.trees == []
    assert np.allclose(f.predict_proba(np.zeros((2, 3))), 0.5)


# ── features ─────────────────────────────────────────────────────────────────
def test_feature_frame_shape_and_causality(series):
    close, ohlc, labels, vf = series
    feats = M.feature_frame(close, labels, vol_forecast=vf)
    assert list(feats.columns) == M.FEATURE_NAMES
    assert len(feats) > 1500
    assert np.isfinite(feats.to_numpy()).all(), "no NaN may reach the forest"
    assert feats["state"].isin([0, 1, 2]).all()
    assert (feats["n"] >= 0).all()
    assert feats["pBull5"].between(0, 1).all()

    # A shorter history must produce a prefix of the same rows: nothing in the
    # features may depend on data that arrives later.
    cut = 400
    shorter = M.feature_frame(close.iloc[:-cut], labels.iloc[:-cut],
                              vol_forecast=vf.iloc[:-cut] if len(vf) else None)
    shared = feats.index.intersection(shorter.index)
    assert len(shared) > 800
    cols = ["signal", "z", "n", "state", "daysInRegime", "rsi14", "mom20"]
    assert np.allclose(feats.loc[shared, cols].to_numpy(),
                       shorter.loc[shared, cols].to_numpy(), atol=1e-9)


def test_feature_frame_is_empty_without_enough_history():
    close = pd.Series(np.linspace(100, 110, 100),
                      index=pd.bdate_range("2020-01-01", periods=100))
    assert M.feature_frame(close, label_regimes(close)).empty


# ── the walk-forward layer, and its promise not to break anything ────────────
def test_meta_probabilities_are_probabilities(series):
    close, ohlc, labels, vf = series
    res = M.meta_probabilities(close, labels, vol_forecast=vf, refit_every=126)
    assert len(res["prob"]) > 500
    assert res["prob"].between(0.0, 1.0).all()
    assert 0.0 <= res["threshold"] <= 1.0
    assert res["prob"].index.is_monotonic_increasing


def test_meta_layer_leaves_every_existing_number_untouched(series):
    close, ohlc, labels, vf = series
    res = M.meta_probabilities(close, labels, vol_forecast=vf, refit_every=126)

    before = walk_forward_backtest(close, labels, vol_forecast=vf)
    after = walk_forward_backtest(close, labels, vol_forecast=vf,
                                  meta_prob=res["prob"], meta_threshold=res["threshold"])
    for key in ("sharpe", "max_drawdown", "n_trades", "win_rate", "equity",
                "equity_index", "skew", "kurtosis", "n_obs"):
        assert after[key] == before[key], f"{key} changed — the layer is not additive"
    assert after["vt"] == before["vt"]
    assert "meta" not in before and "meta" in after


def test_meta_block_reports_its_own_trade_count(series):
    close, ohlc, labels, vf = series
    res = M.meta_probabilities(close, labels, vol_forecast=vf, refit_every=126)
    out = walk_forward_backtest(close, labels, vol_forecast=vf,
                                meta_prob=res["prob"], meta_threshold=res["threshold"])
    m = out["meta"]
    assert 0.0 <= m["take_rate"] <= 1.0
    assert m["n_trades"] <= out["n_trades"], "a filter can only remove trades"
    assert len(m["equity"]) == len(out["equity"])


def test_days_without_a_probability_are_still_traded(series):
    """The forest's warm-up must not silently delete the first year of trades."""
    close, ohlc, labels, vf = series
    empty = pd.Series(dtype=float)
    out = walk_forward_backtest(close, labels, meta_prob=pd.Series(
        [0.99], index=[labels.index[-1]]), meta_threshold=0.5)
    assert out["meta"]["take_rate"] > 0.99, "no opinion must mean 'take it', not 'skip it'"
    assert walk_forward_backtest(close, labels, meta_prob=empty).get("meta") is None


def test_a_gate_that_never_opens_is_a_perfect_no_op(series):
    """The safety property: no demonstrated skill, no change to the curve."""
    close, ohlc, labels, vf = series
    res = M.meta_probabilities(close, labels, vol_forecast=vf, refit_every=126,
                               min_auc=0.99, min_t=99.0)     # unreachable bar
    assert res["active"] is False
    assert (res["prob"] == 1.0).all(), "standing down means passing everything through"

    base = walk_forward_backtest(close, labels, vol_forecast=vf)
    gated = walk_forward_backtest(close, labels, vol_forecast=vf,
                                  meta_prob=res["prob"], meta_threshold=res["threshold"])
    assert gated["meta"]["take_rate"] == 1.0
    assert gated["meta"]["sharpe"] == pytest.approx(base["sharpe"], rel=1e-9)
    assert gated["meta"]["equity"] == pytest.approx(base["equity"], rel=1e-9)


def test_a_wide_open_gate_does_filter(series):
    close, ohlc, labels, vf = series
    res = M.meta_probabilities(close, labels, vol_forecast=vf, refit_every=126,
                               min_auc=0.0, min_t=-99.0)
    assert res["active"] is True and res["nTrain"] > 100
    assert sum(res["importances"].values()) == pytest.approx(1.0, abs=1e-6)
    out = walk_forward_backtest(close, labels, vol_forecast=vf,
                                meta_prob=res["prob"], meta_threshold=res["threshold"])
    assert out["meta"]["take_rate"] < 1.0


def test_meta_probabilities_abstain_on_a_short_history():
    close = pd.Series(np.linspace(100, 120, 200),
                      index=pd.bdate_range("2020-01-01", periods=200))
    res = M.meta_probabilities(close, label_regimes(close))
    assert res["prob"].empty and res["active"] is False


# ── API surface ──────────────────────────────────────────────────────────────
def _demo():
    return TestClient(create_app(AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)))


def test_state_endpoint_does_not_pay_for_the_forest():
    d = _demo().get("/api/state", params={"symbol": "SPY"}).json()
    assert d["metrics"]["meta"] is None, "the default payload must stay cheap"
    assert d["metrics"]["vt"] is not None


def test_meta_label_endpoint_returns_all_three_curves():
    d = _demo().get("/api/meta-label", params={"symbol": "SPY"}).json()
    assert d["symbol"] == "SPY"
    assert d["base"]["equity"] and d["base"]["sharpe"] is not None
    assert d["vt"] is not None and d["meta"] is not None
    m = d["meta"]
    assert 0.0 <= m["takeRate"] <= 1.0
    assert 0.0 <= m["cvAuc"] <= 1.0
    assert isinstance(m["active"], bool)
    assert isinstance(m["importances"], list)
    assert isinstance(d["beatsBase"], bool)


def test_meta_label_endpoint_deflates_both_sharpes():
    d = _demo().get("/api/meta-label", params={"symbol": "SPY"}).json()
    assert 0.0 <= d["base"]["dsr"] <= 1.0
    assert 0.0 <= d["meta"]["dsr"] <= 1.0
    assert d["base"]["verdict"] and d["meta"]["verdict"]


def test_meta_label_endpoint_is_cached():
    import time as _t
    c = _demo()
    t0 = _t.monotonic(); c.get("/api/meta-label", params={"symbol": "SPY"}); first = _t.monotonic() - t0
    t0 = _t.monotonic(); c.get("/api/meta-label", params={"symbol": "SPY"}); second = _t.monotonic() - t0
    assert second < max(first * 0.2, 0.1), "the second call must come from cache"
