"""A library of quant strategies, and the ways two of them can be paired.

Every strategy here is a function from a close series to a position series in
[-1, +1], and every one is *causal*: the position at a bar depends only on bars
at or before it. That is not a style preference — it is what lets the whole
history be evaluated in one pass without leaking the future, and
`backtest.is_causal` checks each one rather than trusting the claim. A centred
window, a full-series normalisation or a stray `bfill` breaks it silently, and
the only symptom is a better equity curve.

The families are the standard ones, deliberately: trend, mean reversion,
volatility, and the terminal's own regime chain. The point of the lab is not to
invent an exotic signal, it is to find out which of these ordinary ones survive
costs and which pairing of two survives being chosen.

Buy-and-hold is in the library on purpose. It is the thing every strategy has to
beat to be worth running, and a search that cannot see it will happily recommend
something worse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── helpers ─────────────────────────────────────────────────────────────────
def _clip(pos: pd.Series) -> pd.Series:
    return pos.clip(-1.0, 1.0).fillna(0.0)


def _z(x: pd.Series, window: int) -> pd.Series:
    """Rolling z-score. Rolling, not full-sample: the mean and deviation a
    strategy standardises against must be ones it could have known."""
    mu = x.rolling(window, min_periods=window).mean()
    sd = x.rolling(window, min_periods=window).std(ddof=0)
    return (x - mu) / sd.replace(0.0, np.nan)


def realised_vol(close: pd.Series, window: int = 20) -> pd.Series:
    return close.pct_change().rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252)


# ── trend ───────────────────────────────────────────────────────────────────
def ma_cross(fast: int = 20, slow: int = 100):
    """Long above the slow average, short below. The oldest trend rule there is."""
    def f(close: pd.Series) -> pd.Series:
        c = pd.Series(close).astype(float)
        a = c.rolling(fast, min_periods=fast).mean()
        b = c.rolling(slow, min_periods=slow).mean()
        return _clip(np.sign(a - b))
    return f


def momentum(lookback: int = 126):
    """Time-series momentum: hold what has gone up over the lookback.

    The most replicated anomaly in the literature and still the one most likely
    to survive out of sample, which makes it the honest benchmark for anything
    cleverer.
    """
    def f(close: pd.Series) -> pd.Series:
        c = pd.Series(close).astype(float)
        return _clip(np.sign(c.pct_change(lookback)))
    return f


def breakout(window: int = 55):
    """Donchian: long on a new high of the window, short on a new low, and hold
    the last decision in between."""
    def f(close: pd.Series) -> pd.Series:
        c = pd.Series(close).astype(float)
        hi = c.rolling(window, min_periods=window).max()
        lo = c.rolling(window, min_periods=window).min()
        pos = pd.Series(np.nan, index=c.index)
        pos[c >= hi] = 1.0
        pos[c <= lo] = -1.0
        return _clip(pos.ffill())
    return f


# ── mean reversion ──────────────────────────────────────────────────────────
def zscore_reversion(window: int = 20, entry: float = 1.5):
    """Fade a stretched move. Sized by how stretched, capped at one."""
    def f(close: pd.Series) -> pd.Series:
        c = pd.Series(close).astype(float)
        z = _z(c, window)
        pos = (-z / entry).where(z.abs() >= entry, 0.0)
        return _clip(pos)
    return f


def rsi_reversion(window: int = 14, low: float = 30.0, high: float = 70.0):
    def f(close: pd.Series) -> pd.Series:
        c = pd.Series(close).astype(float)
        d = c.diff()
        up = d.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
        rsi = 100 - 100 / (1 + up / dn.replace(0.0, np.nan))
        pos = pd.Series(0.0, index=c.index)
        pos[rsi <= low] = 1.0
        pos[rsi >= high] = -1.0
        return _clip(pos)
    return f


# ── volatility ──────────────────────────────────────────────────────────────
def vol_target(target: float = 0.15, window: int = 20, cap: float = 1.0):
    """Long, sized so the position's volatility is roughly constant.

    Not a signal at all — a sizing rule. It earns its place because pairing it
    with a directional strategy is often worth more than any second signal.
    """
    def f(close: pd.Series) -> pd.Series:
        v = realised_vol(pd.Series(close).astype(float), window)
        return _clip((target / v.replace(0.0, np.nan)).clip(upper=cap))
    return f


def low_vol_filter(window: int = 20, pct: float = 0.7):
    """Hold only while volatility sits below its own rolling percentile."""
    def f(close: pd.Series) -> pd.Series:
        v = realised_vol(pd.Series(close).astype(float), window)
        thresh = v.rolling(252, min_periods=60).quantile(pct)
        return _clip((v <= thresh).astype(float))
    return f


def buy_and_hold():
    """The bar. Anything that cannot beat this is not worth the turnover."""
    def f(close: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=pd.Series(close).index)
    return f


# ── the terminal's own chain ────────────────────────────────────────────────
def markov_regime(window: int = 20, threshold: float = 0.02):
    """The regime signal the rest of the terminal runs on, as a lab candidate.

    Included so the search can answer the question that matters most to this
    terminal: does the chain add anything over the ordinary alternatives, and
    does pairing it with one of them beat either alone.
    """
    def f(close: pd.Series) -> pd.Series:
        c = pd.Series(close).astype(float)
        roll = c.pct_change(window)
        pos = pd.Series(0.0, index=c.index)
        pos[roll > threshold] = 1.0
        pos[roll < -threshold] = -1.0
        return _clip(pos)
    return f


# ── pairing ─────────────────────────────────────────────────────────────────
def blend(a, b, weight: float = 0.5):
    """Hold both, weighted. Diversification between signals rather than a veto."""
    def f(close: pd.Series) -> pd.Series:
        return _clip(weight * pd.Series(a(close)) + (1.0 - weight) * pd.Series(b(close)))
    return f


def gate(a, b):
    """Trade `a`'s position, but only while `b` agrees on direction.

    The most useful pairing in practice and the one most likely to be confused
    with an edge: filtering out disagreements always improves a backtest's
    Sharpe by removing trades, whether or not the filter knows anything. Costs
    and the trial count are what stop that being mistaken for skill.
    """
    def f(close: pd.Series) -> pd.Series:
        pa, pb = pd.Series(a(close)), pd.Series(b(close))
        agree = np.sign(pa) == np.sign(pb)
        return _clip(pa.where(agree & (np.sign(pa) != 0), 0.0))
    return f


def switch(a, b, window: int = 20, pct: float = 0.6):
    """`a` in calm markets, `b` in volatile ones."""
    def f(close: pd.Series) -> pd.Series:
        c = pd.Series(close).astype(float)
        v = realised_vol(c, window)
        thresh = v.rolling(252, min_periods=60).quantile(pct)
        calm = (v <= thresh)
        return _clip(pd.Series(a(close)).where(calm, pd.Series(b(close))))
    return f


def long_only(a):
    """Clip a strategy to the long side.

    Not merely a preference. Shorting a single name costs borrow this engine
    does not model, can be recalled at the worst moment, and turns a swing into
    something with a day-trade profile — so a short-side edge found in a
    backtest is less tradeable than the number suggests.
    """
    def f(close: pd.Series) -> pd.Series:
        return _clip(pd.Series(a(close)).clip(lower=0.0))
    return f


def scale(a, factor):
    """Resize one strategy by another used purely as a sizing rule."""
    def f(close: pd.Series) -> pd.Series:
        return _clip(pd.Series(a(close)) * pd.Series(factor(close)).abs())
    return f


# ── the catalogue ───────────────────────────────────────────────────────────
def library() -> dict:
    """Every single strategy, named. Parameters are a small, fixed spread —
    deliberately small, because every extra variant is another trial the winner
    has to be deflated by, and a hundred lookbacks of the same idea buys noise
    rather than coverage.
    """
    out = {
        "buy_and_hold": buy_and_hold(),
        "vol_target_15": vol_target(0.15),
        "low_vol_filter": low_vol_filter(),
        "markov_regime": markov_regime(),
    }
    for f, s in ((20, 100), (50, 200)):
        out[f"ma_cross_{f}_{s}"] = ma_cross(f, s)
    for lb in (63, 126, 252):
        out[f"momentum_{lb}"] = momentum(lb)
    for w in (20, 55):
        out[f"breakout_{w}"] = breakout(w)
    for w in (10, 20):
        out[f"reversion_z{w}"] = zscore_reversion(w)
    out["rsi_reversion"] = rsi_reversion()
    return out


PAIRINGS = {"blend": blend, "gate": gate, "switch": switch, "scale": scale}
