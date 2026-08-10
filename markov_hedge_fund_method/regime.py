"""Observable Markov regime model.

Labels each day Bull (1), Bear (-1), or Sideways (0) using a rolling
return threshold, then builds a 3x3 transition matrix via MLE counting,
solves for the stationary distribution, and runs a walk-forward backtest.
"""

from __future__ import annotations

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
    n = 3
    counts = np.zeros((n, n), dtype=float)
    arr = labels.to_numpy()[:: max(1, stride)]
    for i in range(len(arr) - 1):
        counts[arr[i], arr[i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid divide-by-zero on empty rows
    return counts / row_sums


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
                "win_rate": float("nan"), "equity": [], "equity_index": []}

    # Incremental transition counts: refitting the matrix from scratch each day
    # is O(n^2); each step only adds ONE new transition, so carry the counts and
    # update them. Identical results, ~n times less work.
    lab = labels.to_numpy().astype(int)
    rets = daily_returns.to_numpy().astype(float)
    counts = np.zeros((3, 3), dtype=float)
    for i in range(min_train - 1):
        counts[lab[i], lab[i + 1]] += 1.0

    strategy_returns = []
    equity_dates = []
    for t in range(min_train, len(lab) - 1):
        if t > min_train:                      # extend the window by one transition
            counts[lab[t - 2], lab[t - 1]] += 1.0
        row = counts[lab[t]]
        total = row.sum()
        signal = float((row[2] - row[0]) / total) if total else 0.0
        position = float(np.sign(signal))      # +1 / 0 / -1 — simple sign
        strategy_returns.append(position * float(rets[t + 1]))
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

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": int(len(sr)),
        "win_rate": win_rate,
        "equity": equity.tolist(),
        "equity_index": [d.strftime("%Y-%m-%d") for d in equity_dates],
    }


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
