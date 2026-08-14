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
from .regime import (
    STATES,
    label_regimes,
    matrix_uncertainty,
    n_step_forecast,
    signal_confidence,
    transition_counts,
    walk_forward_backtest,
)
from .har import vol_forecast_for
from .sharpe_stats import deannualize, probabilistic_sharpe, verdict

_REGIME_KEY = {0: "bear", 1: "sideways", 2: "bull"}


def _forecast(P, current: int, horizons=(1, 5, 20)) -> list[dict]:
    """Chapman-Kolmogorov: regime probabilities n steps ahead from `current`."""
    out = []
    for h in horizons:
        row = n_step_forecast(P, h)[current]
        out.append({
            "h": h,
            "bear": round(float(row[0]), 4),
            "sideways": round(float(row[1]), 4),
            "bull": round(float(row[2]), 4),
        })
    return out


def _timeline(labels: pd.Series) -> dict:
    """Days in the current regime + the most recent regime flips."""
    arr = labels.to_numpy()
    idx = labels.index
    if not len(arr):
        return {"daysInRegime": 0, "recentFlips": []}
    cur = int(arr[-1])
    days = 0
    for v in arr[::-1]:
        if int(v) == cur:
            days += 1
        else:
            break
    flips, prev = [], int(arr[0])
    for i in range(1, len(arr)):
        v = int(arr[i])
        if v != prev:
            flips.append({"date": idx[i].strftime("%Y-%m-%d"),
                          "from": _REGIME_KEY[prev], "to": _REGIME_KEY[v]})
            prev = v
    return {"daysInRegime": days, "recentFlips": flips[-6:][::-1]}


def _downsample(values: list, index: list, target: int = 140):
    if not values:
        return [], []
    step = max(1, len(values) // target)
    v = values[::step]
    ix = index[::step] if index else []
    if v[-1] != values[-1]:
        v.append(values[-1])
        if ix:
            ix.append(index[-1])
    return [round(float(x), 5) for x in v], ix


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


def plain_summary(ticker: str, regime: str, days: int, stay_pct: float,
                  confidence: float, reliable: bool, n: int) -> str:
    """One terse read-out of what the numbers on this screen amount to.

    Written in desk shorthand, not explainer prose: regime, how long it has
    run, how often it persisted historically, and — the part the rest of the
    panel cannot show — whether the sample is big enough to act on.
    """
    label = {"bull": "Bull", "bear": "Bear", "sideways": "Sideways"}.get(regime, "Sideways")
    head = f"{ticker} {label}" + (f", day {days}." if days else ".")
    if not n:
        return f"{head} Insufficient history to estimate persistence."
    body = f" Persisted in {round(stay_pct)}% of {n} comparable setups"
    if reliable:
        tail = " — sample supports the read."
    elif confidence >= 0.8:
        tail = " — suggestive, but short of significance."
    else:
        tail = " — sample too thin to trade on."
    return head + body + tail


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
                 include_metrics: bool = True, ohlc: pd.DataFrame | None = None,
                 with_vol: bool = True) -> dict:
    """Full HUD payload for one symbol from its close series.

    `tail` is how many recent bars of chart series to send — large enough that
    the client can slice its own timeframe (1M/3M/6M/1Y/2Y) with no round-trip.
    `ohlc` (Open/High/Low/Close) drives the candlesticks; when absent, candles
    are derived from close (open = previous close) so the chart still renders.
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

    r4 = lambda v: round(float(v), 4)
    if ohlc is not None and not ohlc.empty:
        od = ohlc.reindex(tail_close.index).ffill()
        bars = [
            {"t": ts.strftime("%Y-%m-%d"), "o": r4(o), "h": r4(h), "l": r4(lo), "c": r4(c),
             "up": bool(c >= o), "regime": _REGIME_KEY[int(rg)]}
            for ts, o, h, lo, c, rg in zip(
                tail_close.index, od["Open"], od["High"], od["Low"], od["Close"], lab_tail.to_numpy())
        ]
    else:
        cvals = tail_close.to_numpy()
        bars = []
        for i, (ts, c, rg) in enumerate(zip(tail_close.index, cvals, lab_tail.to_numpy())):
            o = float(cvals[i - 1]) if i > 0 else float(c)
            bars.append({"t": ts.strftime("%Y-%m-%d"), "o": r4(o), "h": r4(max(o, c) * 1.002),
                         "l": r4(min(o, c) * 0.998), "c": r4(c), "up": bool(c >= o),
                         "regime": _REGIME_KEY[int(rg)]})

    # How much evidence sits behind the honest matrix. Stride sampling makes the
    # estimates truthful but sparse, so the counts and intervals are the only
    # way to tell a well-supported probability from a near-guess.
    counts = transition_counts(labels, stride=window)
    unc = matrix_uncertainty(counts)
    ci = {
        "n": [int(v) for v in unc["n"]],
        "lo": [[round(float(v), 4) for v in row] for row in unc["lo"]],
        "hi": [[round(float(v), 4) for v in row] for row in unc["hi"]],
        "thin": [bool(v < 20) for v in unc["n"]],
    }
    sc = signal_confidence(counts, snap.current_state)
    sig_stats = {
        "n": sc["n"],
        "stderr": None if sc["stderr"] is None else round(sc["stderr"], 4),
        "z": round(sc["z"], 2),
        "confidence": round(sc["confidence"], 4),
        "reliable": sc["reliable"],
    }

    # Walk-forward volatility forecast — used only to size positions, never to
    # change the signal. Falls back to close-to-close when no OHLC is present.
    vol_fc = None
    if include_metrics and with_vol:
        try:
            vol_fc = vol_forecast_for(close, ohlc)
        except Exception:  # noqa: BLE001 — sizing is optional, never fatal
            vol_fc = None

    metrics = walk_forward_backtest(close, labels, vol_forecast=vol_fc) if include_metrics else {
        "sharpe": float("nan"), "max_drawdown": float("nan"), "n_trades": 0,
        "win_rate": float("nan"), "equity": [], "equity_index": []}
    eq, eq_idx = _downsample(metrics.get("equity", []), metrics.get("equity_index", []))

    vt_raw = metrics.get("vt")
    vt_block = None
    if vt_raw:
        vt_eq, _ = _downsample(vt_raw.get("equity", []), vt_raw.get("equity_index", []))
        vt_block = {
            "sharpe": None if pd.isna(vt_raw["sharpe"]) else round(vt_raw["sharpe"], 3),
            "maxDrawdown": None if pd.isna(vt_raw["max_drawdown"]) else round(vt_raw["max_drawdown"], 4),
            "winRate": None if pd.isna(vt_raw["win_rate"]) else round(vt_raw["win_rate"], 4),
            "equity": vt_eq,
            "avgLeverage": None if pd.isna(vt_raw["avg_leverage"]) else round(vt_raw["avg_leverage"], 2),
            "targetVol": vt_raw["target_vol"],
            "leverageCap": vt_raw["leverage_cap"],
        }

    # Today's forecast for tomorrow's volatility, annualised.
    vol_now = None
    if vol_fc is not None and len(vol_fc):
        vol_now = round(float(vol_fc.iloc[-1]), 4)

    psr = None
    if not pd.isna(metrics.get("sharpe", float("nan"))) and metrics.get("n_obs", 0) > 2:
        psr = round(probabilistic_sharpe(
            deannualize(metrics["sharpe"]), metrics["n_obs"],
            skew=metrics.get("skew", 0.0), kurtosis=metrics.get("kurtosis", 3.0)), 4)

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
        "matrixCI": ci,
        "signalStats": sig_stats,
        "plainSummary": plain_summary(
            ticker, _REGIME_KEY[snap.current_state],
            _timeline(labels)["daysInRegime"],
            float(snap.honest_matrix[snap.current_state][snap.current_state]) * 100.0,
            sc["confidence"], sc["reliable"], sc["n"]),
        "stationary": [round(float(v), 4) for v in snap.stationary],
        "states": STATES,
        "diagonalInflation": [round(float(v), 2) for v in snap.comparison.inflation],
        "verified": bool(snap.verification.passed),
        "greedFear": greed_fear(close),
        "forecast": _forecast(snap.honest_matrix, snap.current_state),
        "regimeTimeline": _timeline(labels),
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
            "winRate": None if pd.isna(metrics.get("win_rate", float("nan"))) else round(metrics["win_rate"], 4),
            "equity": eq,
            "equityIndex": eq_idx,
            # Is that Sharpe believable given the sample length and fat tails?
            "psr": psr,
            "psrVerdict": None if psr is None else verdict(psr),
            "nObs": metrics.get("n_obs", 0),
            "skew": round(float(metrics.get("skew", 0.0)), 3),
            "kurtosis": round(float(metrics.get("kurtosis", 3.0)), 3),
            # Vol-targeted variant, shown next to the plain curve for comparison
            "vt": vt_block,
        },
        "volForecast": vol_now,
    }
