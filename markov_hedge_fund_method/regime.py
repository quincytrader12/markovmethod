"""Observable Markov regime model.

Labels each day Bull (1), Bear (-1), or Sideways (0) using a rolling
return threshold, then builds a 3x3 transition matrix via MLE counting,
solves for the stationary distribution, and runs a walk-forward backtest.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

STATES = ["Bear", "Sideways", "Bull"]  # index 0, 1, 2


def label_regimes(close: pd.Series, window: int = 20, threshold: float = 0.02) -> pd.Series:
    """Label each day as Bull / Bear / Sideways from rolling return.

    Bull   : rolling return > +threshold
    Bear   : rolling return < -threshold
    Sideways: otherwise
    """
    rolling_return = close.pct_change(window)
    labels = pd.Series(1, index=close.index, dtype=int)  # default Sideways
    labels[rolling_return > threshold] = 2  # Bull
    labels[rolling_return < -threshold] = 0  # Bear
    return labels.dropna()


def build_transition_matrix(labels: pd.Series, stride: int = 1) -> np.ndarray:
    """MLE estimate of the 3x3 transition matrix from a sequence of labels.

    `stride` controls the sampling of the label sequence before counting:

      stride == 1        legacy / overlapping. Counts day-to-day transitions.
                         When labels come from an N-day *rolling* return,
                         consecutive labels share N-1 days of data, which
                         inflates the diagonal (fake persistence). See
                         markov2.py — this is the flaw Fix 1 corrects.

      stride == window   non-overlapping. Samples labels window-days apart so
                         adjacent observations share no data. Statistically
                         honest, at the cost of ~window× fewer transitions.
    """
    counts = transition_counts(labels, stride)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid divide-by-zero on empty rows
    return counts / row_sums


def transition_counts(labels: pd.Series, stride: int = 1) -> np.ndarray:
    """Raw 3x3 transition counts — the evidence behind the probabilities.

    Kept separate from the matrix itself because the counts are what tell you
    whether a probability is trustworthy: honest (stride-sampled) estimates can
    rest on very few observations, and a rate has no meaning without its n.
    """
    counts = np.zeros((3, 3), dtype=float)
    arr = labels.to_numpy()[:: max(1, stride)]
    for i in range(len(arr) - 1):
        counts[arr[i], arr[i + 1]] += 1
    return counts


def wilson_interval(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n.

    Preferred over the textbook normal interval because it stays inside [0, 1]
    and behaves sensibly for small n — which is exactly the regime we are in
    once the labels are stride-sampled.
    """
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def matrix_uncertainty(counts: np.ndarray, z: float = 1.96) -> dict:
    """Confidence intervals for every cell of a transition matrix.

    Returns per-row sample sizes and a 3x3 grid of (lo, hi) bounds, so the UI
    can show '57.7% [41-73%], n=18' instead of a bare number that looks just as
    solid as one backed by 200 observations.
    """
    counts = np.asarray(counts, dtype=float)
    row_n = counts.sum(axis=1)
    lo = np.zeros((3, 3))
    hi = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            lo[i, j], hi[i, j] = wilson_interval(counts[i, j], row_n[i], z)
    return {"n": row_n, "lo": lo, "hi": hi}


def signal_confidence(counts: np.ndarray, state: int) -> dict:
    """How much evidence stands behind the Bull-minus-Bear signal.

    The signal is a difference of two proportions estimated from the *same*
    multinomial row, so its variance is (p_bull + p_bear - (p_bull-p_bear)^2)/n.
    Comparing the signal to that spread answers the only question that matters
    before trading it: could this just be sampling noise?
    """
    counts = np.asarray(counts, dtype=float)
    n = float(counts[state].sum())
    if n <= 0:
        return {"n": 0, "signal": 0.0, "stderr": None, "z": 0.0,
                "confidence": 0.0, "reliable": False}
    p_bear = counts[state, 0] / n
    p_bull = counts[state, 2] / n
    diff = p_bull - p_bear
    var = (p_bull + p_bear - diff * diff) / n
    stderr = float(np.sqrt(var)) if var > 0 else 0.0
    if stderr <= 0:
        z = 0.0
    else:
        z = diff / stderr
    # Two-sided: how confident are we the sign is not an artefact of small n?
    confidence = float(math.erf(abs(z) / math.sqrt(2.0)))
    return {
        "n": int(n),
        "signal": float(diff),
        "stderr": stderr,
        "z": float(z),
        "confidence": confidence,
        "reliable": bool(confidence >= 0.95 and n >= 20),
    }



def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of P with eigenvalue 1, normalised to sum to 1."""
    eigvals, eigvecs = np.linalg.eig(P.T)
    # Find the eigenvector closest to eigenvalue 1
    idx = np.argmin(np.abs(eigvals - 1.0))
    vec = np.real(eigvecs[:, idx])
    vec = np.abs(vec)
    return vec / vec.sum()


def n_step_forecast(P: np.ndarray, n: int) -> np.ndarray:
    """Chapman-Kolmogorov: P^n is the n-step transition matrix."""
    return np.linalg.matrix_power(P, n)


def signal_from_matrix(P: np.ndarray, current_state: int) -> float:
    """Signed signal: P(next=Bull|current) - P(next=Bear|current).

    Positive -> long, negative -> short, magnitude -> conviction.
    """
    return float(P[current_state, 2] - P[current_state, 0])


def walk_forward_backtest(
    close: pd.Series,
    labels: pd.Series,
    min_train: int = 252,
    vol_forecast: pd.Series | None = None,
    target_vol: float = 0.15,
    leverage_cap: float = 2.0,
) -> dict:
    """Walk-forward: at each day t, fit the matrix on labels up to t-1,
    derive the signal from the current state, hold for one day, score.

    No lookahead. No tuning.
    """
    daily_returns = close.pct_change().dropna()
    common_index = labels.index.intersection(daily_returns.index)
    labels = labels.loc[common_index]
    daily_returns = daily_returns.loc[common_index]

    if len(labels) < min_train + 30:
        return {"sharpe": float("nan"), "max_drawdown": float("nan"), "n_trades": 0,
                "win_rate": float("nan"), "equity": [], "equity_index": [],
                "skew": 0.0, "kurtosis": 3.0, "n_obs": 0}

    # Incremental transition counts: refitting the matrix from scratch each day
    # is O(n^2); each step only adds ONE new transition, so carry the counts and
    # update them. Identical results, ~n times less work.
    lab = labels.to_numpy().astype(int)
    rets = daily_returns.to_numpy().astype(float)
    counts = np.zeros((3, 3), dtype=float)
    for i in range(min_train - 1):
        counts[lab[i], lab[i + 1]] += 1.0

    # Optional vol targeting: same signal and same direction, but the *size*
    # shrinks when tomorrow's volatility is forecast high. Aligned by date so a
    # forecast is only ever used on the day it was made for.
    vf = None
    if vol_forecast is not None and len(vol_forecast):
        vf = vol_forecast.reindex(labels.index).to_numpy(dtype=float)

    strategy_returns = []
    vt_returns = []
    leverages = []
    equity_dates = []
    for t in range(min_train, len(lab) - 1):
        if t > min_train:                      # extend the window by one transition
            counts[lab[t - 2], lab[t - 1]] += 1.0
        row = counts[lab[t]]
        total = row.sum()
        signal = float((row[2] - row[0]) / total) if total else 0.0
        position = float(np.sign(signal))      # +1 / 0 / -1 — simple sign
        r_next = float(rets[t + 1])
        strategy_returns.append(position * r_next)
        if vf is not None:
            sigma = vf[t + 1]
            scale = 1.0
            if np.isfinite(sigma) and sigma > 0:
                scale = min(target_vol / sigma, leverage_cap)
            leverages.append(scale)
            vt_returns.append(position * scale * r_next)
        equity_dates.append(labels.index[t + 1])

    sr = np.array(strategy_returns, dtype=float)
    if sr.std(ddof=1) == 0 or not np.isfinite(sr.std(ddof=1)):
        sharpe = float("nan")
    else:
        sharpe = float(sr.mean() / sr.std(ddof=1) * np.sqrt(252))

    equity = (1.0 + sr).cumprod()
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) else float("nan")

    acted = sr[sr != 0.0]  # only days the strategy took a position
    win_rate = float((acted > 0).mean()) if len(acted) else float("nan")

    # Shape of the return distribution — needed to judge whether the Sharpe is
    # believable. Fat tails and negative skew make the same Sharpe weaker.
    sd = sr.std(ddof=1)
    if len(sr) > 2 and sd > 0 and np.isfinite(sd):
        z = (sr - sr.mean()) / sd
        skew = float((z ** 3).mean())
        kurt = float((z ** 4).mean())
    else:
        skew, kurt = 0.0, 3.0

    out = {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": int(len(sr)),
        "win_rate": win_rate,
        "equity": equity.tolist(),
        "equity_index": [d.strftime("%Y-%m-%d") for d in equity_dates],
        "skew": skew,
        "kurtosis": kurt,
        "n_obs": int(len(sr)),
    }

    # Vol-targeted variant, reported alongside rather than replacing anything —
    # the point is to be able to compare the two curves on your own data.
    if vt_returns:
        vt = np.array(vt_returns, dtype=float)
        vt_sd = vt.std(ddof=1)
        vt_equity = (1.0 + vt).cumprod()
        vt_run_max = np.maximum.accumulate(vt_equity)
        vt_dd = (vt_equity - vt_run_max) / vt_run_max
        vt_acted = vt[vt != 0.0]
        out["vt"] = {
            "sharpe": (float(vt.mean() / vt_sd * np.sqrt(252))
                       if vt_sd > 0 and np.isfinite(vt_sd) else float("nan")),
            "max_drawdown": float(vt_dd.min()) if len(vt_dd) else float("nan"),
            "win_rate": float((vt_acted > 0).mean()) if len(vt_acted) else float("nan"),
            "equity": vt_equity.tolist(),
            "equity_index": out["equity_index"],
            "avg_leverage": float(np.mean(leverages)) if leverages else float("nan"),
            "target_vol": float(target_vol),
            "leverage_cap": float(leverage_cap),
        }
    return out


_REGIME_NAMES = {0: "bear", 1: "sideways", 2: "bull"}


def regime_performance(close: pd.Series, labels: pd.Series, min_train: int = 252) -> dict:
    """Walk-forward strategy return bucketed by the regime in force each day.

    Same no-lookahead loop as the backtest, but records which regime the market
    was in when each day's return was earned. Answers the differentiator
    question: *in which regime does the strategy actually make money?*
    """
    daily_returns = close.pct_change().dropna()
    common = labels.index.intersection(daily_returns.index)
    labels = labels.loc[common]
    daily_returns = daily_returns.loc[common]

    buckets: dict[int, list[float]] = {0: [], 1: [], 2: []}
    if len(labels) >= min_train + 30:
        lab = labels.to_numpy().astype(int)
        rets = daily_returns.to_numpy().astype(float)
        counts = np.zeros((3, 3), dtype=float)      # incremental — see backtest
        for i in range(min_train - 1):
            counts[lab[i], lab[i + 1]] += 1.0
        for t in range(min_train, len(lab) - 1):
            if t > min_train:
                counts[lab[t - 2], lab[t - 1]] += 1.0
            state = int(lab[t])
            row = counts[state]
            total = row.sum()
            signal = float((row[2] - row[0]) / total) if total else 0.0
            buckets[state].append(float(np.sign(signal)) * float(rets[t + 1]))

    out = {}
    for state, name in _REGIME_NAMES.items():
        arr = np.array(buckets[state], dtype=float)
        acted = arr[arr != 0.0]
        out[name] = {
            "days": int(len(arr)),
            "traded": int(len(acted)),
            "winRate": float((acted > 0).mean()) if len(acted) else None,
            "avgReturn": float(arr.mean()) if len(arr) else None,
            "totalReturn": float(arr.sum()) if len(arr) else 0.0,
        }
    return out
