"""HAR-RV volatility forecasting and vol-targeted sizing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markov_hedge_fund_method.har import (close_to_close_vol, fit_har, forecast_series,
                                          garman_klass_vol, har_design, vol_forecast_for)
from markov_hedge_fund_method.market_data import synthetic_ohlc
from markov_hedge_fund_method.regime import label_regimes, walk_forward_backtest


def _garch_ohlc(seed=1, a=0.06, b=0.92, n=1800):
    """Bars whose volatility genuinely clusters, like real markets."""
    rng = np.random.default_rng(seed)
    steps, w = 78, 3e-6
    var = np.empty(n); var[0] = w / (1 - a - b)
    O = np.empty(n); H = np.empty(n); L = np.empty(n); C = np.empty(n)
    px, dr = 100.0, np.zeros(n)
    for t in range(n):
        sig = np.sqrt(max(var[t], 1e-12) / steps)
        path = px * np.exp(np.cumsum(sig * rng.standard_normal(steps)))
        O[t], H[t], L[t], C[t] = px, path.max(), path.min(), path[-1]
        dr[t] = np.log(C[t] / O[t]); px = C[t]
        if t + 1 < n:
            var[t + 1] = max(w + a * dr[t] ** 2 + b * var[t], 1e-12)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    ohlc = pd.DataFrame({"Open": O, "High": H, "Low": L, "Close": C}, index=idx)
    return ohlc, pd.Series(np.sqrt(var), index=idx) * np.sqrt(252)


# ── estimator ────────────────────────────────────────────────────────────────
def test_garman_klass_is_positive_and_tracks_range():
    ohlc = synthetic_ohlc(seed=2)
    rv = garman_klass_vol(ohlc)
    assert (rv > 0).all() and rv.notna().all()
    # a wider bar must imply higher estimated volatility
    wide = ohlc.copy()
    wide["High"] = wide["High"] * 1.05
    wide["Low"] = wide["Low"] * 0.95
    assert garman_klass_vol(wide).mean() > rv.mean()


def test_design_matrix_has_the_three_horizons():
    rv = garman_klass_vol(synthetic_ohlc(seed=2))
    X, y, idx = har_design(rv)
    assert X.shape[1] == 4 and len(y) == len(idx) == X.shape[0]
    assert np.allclose(X[:, 0], 1.0)          # intercept


def test_fit_is_stable_on_short_series():
    coefs, var = fit_har(pd.Series([0.01] * 10))
    assert len(coefs) == 4 and var >= 0.0     # degrades gracefully, no crash


# ── the forecast ─────────────────────────────────────────────────────────────
def test_forecast_tracks_true_volatility():
    ohlc, true_vol = _garch_ohlc()
    f = forecast_series(garman_klass_vol(ohlc))
    assert len(f) > 100
    assert f.corr(true_vol.reindex(f.index)) > 0.6, "should track latent vol"
    assert (f > 0).all()


def test_forecast_has_no_lookahead():
    """Truncating the future must not change past forecasts."""
    ohlc, _ = _garch_ohlc(seed=4)
    full = forecast_series(garman_klass_vol(ohlc))
    cut = forecast_series(garman_klass_vol(ohlc.iloc[:-200]))
    shared = full.index.intersection(cut.index)
    assert len(shared) > 50
    assert np.allclose(full.reindex(shared), cut.reindex(shared))


def test_har_adapts_to_memory_length():
    """Long-memory vol should load the monthly term more than short-memory."""
    short, _ = _garch_ohlc(seed=3, a=0.09, b=0.70)
    long, _ = _garch_ohlc(seed=3, a=0.05, b=0.94)
    c_short, _ = fit_har(garman_klass_vol(short))
    c_long, _ = fit_har(garman_klass_vol(long))
    assert c_long[3] > c_short[3]


def test_falls_back_to_close_only_data():
    close = synthetic_ohlc(seed=6)["Close"]
    assert (close_to_close_vol(close).dropna() > 0).all()
    f = vol_forecast_for(close, None)
    assert len(f) > 0 and (f > 0).all()


# ── vol targeting ────────────────────────────────────────────────────────────
def test_backtest_unchanged_without_vol_forecast():
    ohlc = synthetic_ohlc(seed=8)
    close = ohlc["Close"]
    labels = label_regimes(close)
    base = walk_forward_backtest(close, labels)
    assert "vt" not in base                       # strictly opt-in
    with_vf = walk_forward_backtest(close, labels,
                                    vol_forecast=vol_forecast_for(close, ohlc))
    # the unscaled numbers must be identical — vol targeting is additive
    assert with_vf["sharpe"] == base["sharpe"]
    assert with_vf["max_drawdown"] == base["max_drawdown"]
    assert with_vf["equity"] == base["equity"]
    assert "vt" in with_vf


def test_vol_targeting_reduces_drawdown():
    ohlc = synthetic_ohlc(seed=3)
    close = ohlc["Close"]
    r = walk_forward_backtest(close, label_regimes(close),
                              vol_forecast=vol_forecast_for(close, ohlc))
    vt = r["vt"]
    assert abs(vt["max_drawdown"]) < abs(r["max_drawdown"])
    assert 0 < vt["avg_leverage"] <= vt["leverage_cap"]
    assert len(vt["equity"]) == len(r["equity"])


def test_leverage_is_capped():
    ohlc = synthetic_ohlc(seed=5)
    close = ohlc["Close"]
    vf = vol_forecast_for(close, ohlc) * 0.001     # absurdly calm -> wants huge size
    r = walk_forward_backtest(close, label_regimes(close), vol_forecast=vf,
                              leverage_cap=2.0)
    assert r["vt"]["avg_leverage"] <= 2.0 + 1e-9


def test_payload_exposes_forecast_and_ab_curve():
    from markov_hedge_fund_method.webstate import market_state
    ohlc = synthetic_ohlc(seed=3)
    st = market_state(ohlc["Close"], "SPY", ohlc=ohlc)
    assert st["volForecast"] > 0
    vt = st["metrics"]["vt"]
    assert vt["sharpe"] is not None and vt["avgLeverage"] > 0
    assert len(vt["equity"]) == len(st["metrics"]["equity"])
