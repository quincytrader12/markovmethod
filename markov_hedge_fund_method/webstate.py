"""Build the JSON payload the web HUD renders from a close-price series.

Everything the neon dashboard needs — price + moving averages, a per-bar
Bull/Bear/Sideways regime ribbon, momentum/RSI oscillators, a composite
Greed/Fear index, the honest transition matrix, the signal, the stationary
mix and the walk-forward metrics — computed here so the API stays thin and the
math stays reusable/testable. Pure: a pandas Series in, plain dict out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import analyze2
from .markov2 import Strategy
from .regime import STATES, label_regimes, walk_forward_backtest

_REGIME_KEY = {0: "bear", 1: "sideways", 2: "bull"}


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def greed_fear(close: pd.Series) -> dict:
    """0–100 composite: momentum, strength (RSI), 52-week range position,
    trend, and low-volatility. Mirrors the components shown in the HUD."""
    c = close.astype(float)
    last = float(c.iloc[-1])

    # Momentum: 20-day return mapped so +/-10% spans the scale.
    mom_ret = last / float(c.iloc[-21]) - 1.0 if len(c) > 21 else 0.0
    momentum = _clip01(0.5 + mom_ret / 0.20)

    # Strength: RSI(14) normalised to 0..1.
    rsi_last = float(_rsi(c).iloc[-1])
    strength = _clip01(rsi_last / 100.0)

    # 52-week range position.
    window = c.iloc[-252:] if len(c) >= 252 else c
    lo, hi = float(window.min()), float(window.max())
    rng_pos = _clip01((last - lo) / (hi - lo)) if hi > lo else 0.5

    # Trend: price vs its 50-day MA, +/-10% spans the scale.
    ma50 = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else last
    trend = _clip01(0.5 + (last / ma50 - 1.0) / 0.20) if ma50 else 0.5

    # Low volatility = greedy; high vol = fearful. 20-day annualised vol.
    daily = c.pct_change().dropna()
    vol = float(daily.iloc[-20:].std()) * np.sqrt(252) if len(daily) >= 20 else 0.2
    low_vol = _clip01(1.0 - vol / 0.60)

    components = {
        "momentum": momentum,
        "strength": strength,
        "range": rng_pos,
        "trend": trend,
        "lowVol": low_vol,
    }
    score = int(round(100 * sum(components.values()) / len(components)))
    if score >= 75:
        label = "Extreme Greed"
    elif score >= 55:
        label = "Greed"
    elif score > 45:
        label = "Neutral"
    elif score > 25:
        label = "Fear"
    else:
        label = "Extreme Fear"
    return {
        "score": score,
        "label": label,
        "components": {k: round(v * 100, 1) for k, v in components.items()},
    }


def quote_state(close: pd.Series, ticker: str, *, window: int = 20, threshold: float = 0.02) -> dict:
    """Cheap watchlist quote: last price + current regime, no matrix/backtest."""
    labels = label_regimes(close, window=window, threshold=threshold)
    current = int(labels.iloc[-1]) if len(labels) else 1
    return {
        "ticker": ticker,
        "lastPrice": round(float(close.iloc[-1]), 4),
        "regime": _REGIME_KEY[current],
    }


def market_state(close: pd.Series, ticker: str, *, window: int = 20, threshold: float = 0.02,
                 strategy: Strategy = Strategy.FILTER, tail: int = 520,
                 include_metrics: bool = True) -> dict:
    """Full HUD payload for one symbol from its close series.

    `tail` is how many recent bars of chart series to send — large enough that
    the client can slice its own timeframe (1M/3M/6M/1Y/2Y) with no round-trip.
    """
    snap = analyze2(close, ticker, window=window, threshold=threshold, strategy=strategy)
    labels = label_regimes(close, window=window, threshold=threshold)

    tail_close = close.iloc[-tail:]
    ma20 = close.rolling(20).mean().iloc[-tail:]
    ma50 = close.rolling(50).mean().iloc[-tail:]
    rsi = _rsi(close).iloc[-tail:]
    mom = (close / close.shift(20) - 1.0).iloc[-tail:] * 100.0
    lab_tail = labels.reindex(tail_close.index).ffill().fillna(1).astype(int)

    def series(s):
        return [None if pd.isna(v) else round(float(v), 4) for v in s]

    bars = [
        {
            "t": ts.strftime("%Y-%m-%d"),
            "c": round(float(px), 4),
            "regime": _REGIME_KEY[int(rg)],
        }
        for ts, px, rg in zip(tail_close.index, tail_close.to_numpy(), lab_tail.to_numpy())
    ]

    metrics = walk_forward_backtest(close, labels) if include_metrics else {
        "sharpe": float("nan"), "max_drawdown": float("nan"), "n_trades": 0}

    return {
        "ticker": ticker,
        "asOf": snap.end,
        "start": snap.start,
        "nRows": snap.n_rows,
        "lastPrice": round(snap.last_price, 4),
        "currentState": snap.current_state,
        "currentStateName": snap.current_state_name,
        "regime": _REGIME_KEY[snap.current_state],
        "signal": round(snap.signal, 4),
        "targetLabel": snap.target_label,
        "matrix": [[round(float(v), 4) for v in row] for row in snap.honest_matrix],
        "stationary": [round(float(v), 4) for v in snap.stationary],
        "states": STATES,
        "diagonalInflation": [round(float(v), 2) for v in snap.comparison.inflation],
        "verified": bool(snap.verification.passed),
        "greedFear": greed_fear(close),
        "chart": {
            "bars": bars,
            "ma20": series(ma20),
            "ma50": series(ma50),
            "rsi": series(rsi),
            "momentum": series(mom),
        },
        "metrics": {
            "sharpe": None if pd.isna(metrics["sharpe"]) else round(metrics["sharpe"], 3),
            "maxDrawdown": None if pd.isna(metrics["max_drawdown"]) else round(metrics["max_drawdown"], 4),
            "nTrades": metrics["n_trades"],
        },
    }
