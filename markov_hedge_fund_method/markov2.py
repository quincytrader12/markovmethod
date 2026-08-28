"""Markov 2.0 — the three corrections layered on the observable model.

FIX 1  Stride sampling. Overlapping rolling-window labels fake persistence on
       the diagonal. `compare_matrices` builds BOTH the legacy (overlapping)
       and the stride-sampled (non-overlapping) matrix so the difference is
       visible, and `stickiness` quantifies the inflated diagonal.

FIX 2  Label verification. `verify_labels` programmatically checks the
       Bull/Bear/Sideways mapping against the data's own extremes (the
       highest-return window must be Bull, the lowest Bear, the flattest
       Sideways). A swapped display fails this before the user ever sees it.

FIX 3  Two explicit modes. `desired_position` implements FILTER (regime gates
       a strategy: act only when |signal| clears a threshold) and STANDALONE
       (trade the signal directly, size proportional to |signal| up to a cap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from .regime import (
    STATES,
    build_transition_matrix,
    label_regimes,
    signal_from_matrix,
)


class Strategy(str, Enum):
    FILTER = "filter"          # regime gates an existing strategy
    STANDALONE = "standalone"  # trade the signal directly, sized by conviction


# ── FIX 1 — stride sampling ──────────────────────────────────────────────────
def stickiness(P: np.ndarray) -> np.ndarray:
    """The persistence diagonal — P(stay in state | in state)."""
    return np.diag(P).copy()


@dataclass
class MatrixComparison:
    legacy: np.ndarray          # overlapping (stride=1) — flatters the diagonal
    honest: np.ndarray          # non-overlapping (stride=window) — the truth
    window: int
    n_legacy: int               # transitions counted (overlapping)
    n_honest: int               # transitions counted (stride-sampled)

    @property
    def stickiness_legacy(self) -> np.ndarray:
        return stickiness(self.legacy)

    @property
    def stickiness_honest(self) -> np.ndarray:
        return stickiness(self.honest)

    @property
    def inflation(self) -> np.ndarray:
        """How many percentage points the overlapping diagonal over-states."""
        return (self.stickiness_legacy - self.stickiness_honest) * 100.0


def compare_matrices(labels: pd.Series, window: int) -> MatrixComparison:
    """Build the legacy and stride-sampled matrices side by side."""
    legacy = build_transition_matrix(labels, stride=1)
    honest = build_transition_matrix(labels, stride=window)
    n_legacy = max(len(labels) - 1, 0)
    n_honest = max(len(labels[::window]) - 1, 0)
    return MatrixComparison(legacy, honest, window, n_legacy, n_honest)


# ── FIX 2 — label verification ───────────────────────────────────────────────
@dataclass
class LabelCheck:
    description: str
    ok: bool
    detail: str


@dataclass
class Verification:
    passed: bool
    checks: list[LabelCheck] = field(default_factory=list)


def verify_labels(close: pd.Series, window: int = 20, threshold: float = 0.02) -> Verification:
    """Self-check the label mapping against the data's own extremes.

    Independent of any hand-written date: the window with the largest positive
    cumulative return MUST be labelled Bull, the most negative Bear, and the
    flattest Sideways. If STATES or a display were swapped, this fails.
    """
    roll = close.pct_change(window)
    labels = label_regimes(close, window=window, threshold=threshold)
    roll = roll.loc[labels.index]

    checks: list[LabelCheck] = []

    imax = roll.idxmax()
    ok_bull = int(labels.loc[imax]) == STATES.index("Bull")
    checks.append(LabelCheck(
        "peak-return window is labelled Bull", ok_bull,
        f"{imax.date()} rolled {roll.loc[imax]*100:+.1f}% → {STATES[int(labels.loc[imax])]}",
    ))

    imin = roll.idxmin()
    ok_bear = int(labels.loc[imin]) == STATES.index("Bear")
    checks.append(LabelCheck(
        "trough-return window is labelled Bear", ok_bear,
        f"{imin.date()} rolled {roll.loc[imin]*100:+.1f}% → {STATES[int(labels.loc[imin])]}",
    ))

    izero = roll.abs().idxmin()
    ok_flat = int(labels.loc[izero]) == STATES.index("Sideways")
    checks.append(LabelCheck(
        "flattest window is labelled Sideways", ok_flat,
        f"{izero.date()} rolled {roll.loc[izero]*100:+.1f}% → {STATES[int(labels.loc[izero])]}",
    ))

    return Verification(passed=all(c.ok for c in checks), checks=checks)


# ── FIX 3 — two explicit modes ───────────────────────────────────────────────
def desired_position(
    signal: float,
    strategy: Strategy,
    *,
    threshold: float = 0.15,
    cap: float = 1.0,
    scale: float = 0.5,
) -> float:
    """Map a signal to a target position under the chosen strategy.

    FILTER     : gate. +cap when signal > +threshold, -cap when < -threshold,
                 else 0 (flat in chop). The regime decides WHEN to act.
    STANDALONE : continuous. position = clip(signal / scale, -cap, +cap), so
                 size grows with conviction |signal| up to the cap.
    """
    if strategy is Strategy.FILTER:
        if signal > threshold:
            return cap
        if signal < -threshold:
            return -cap
        return 0.0
    return float(np.clip(signal / scale, -cap, cap))


# ── upgraded walk-forward with honest metrics ────────────────────────────────
@dataclass
class BacktestResult:
    label: str
    stride: int
    sharpe: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    n_trades: int
    exposure: float                 # fraction of days with a non-zero position
    equity: np.ndarray              # equity curve, starts at 1.0
    equity_index: pd.DatetimeIndex


def walk_forward(
    close: pd.Series,
    *,
    window: int = 20,
    threshold: float = 0.02,
    stride: int = 1,
    strategy: Strategy = Strategy.STANDALONE,
    signal_threshold: float = 0.15,
    cap: float = 1.0,
    scale: float = 0.5,
    min_train: int = 252,
    label: str = "",
) -> BacktestResult:
    """Walk forward with no lookahead: at each day t, estimate the matrix on
    labels < t (using `stride` sampling), size the position from the signal
    under `strategy`, and score against the next day's return.
    """
    labels = label_regimes(close, window=window, threshold=threshold)
    daily = close.pct_change().dropna()
    idx = labels.index.intersection(daily.index)
    labels, daily = labels.loc[idx], daily.loc[idx]

    rets: list[float] = []
    dates: list[pd.Timestamp] = []
    if len(labels) >= min_train + 30:
        for t in range(min_train, len(labels) - 1):
            P_t = build_transition_matrix(labels.iloc[:t], stride=stride)
            sig = signal_from_matrix(P_t, int(labels.iloc[t]))
            pos = desired_position(sig, strategy, threshold=signal_threshold, cap=cap, scale=scale)
            rets.append(pos * float(daily.iloc[t + 1]))
            dates.append(labels.index[t + 1])

    sr = np.asarray(rets, dtype=float)
    if sr.size == 0:
        return BacktestResult(label or strategy.value, stride, float("nan"), float("nan"),
                              float("nan"), float("nan"), 0, 0.0,
                              np.array([1.0]), pd.DatetimeIndex(dates))

    active = sr[sr != 0.0]
    std = active.std(ddof=1) if active.size > 1 else 0.0
    sharpe = float(active.mean() / std * np.sqrt(252)) if std else float("nan")
    wins = active[active > 0]
    losses = active[active < 0]
    win_rate = float(len(wins) / len(active)) if active.size else float("nan")
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    equity = np.cumprod(1.0 + sr)
    running_max = np.maximum.accumulate(equity)
    max_dd = float(((equity - running_max) / running_max).min())

    return BacktestResult(
        label=label or strategy.value,
        stride=stride,
        sharpe=sharpe,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        n_trades=int(active.size),
        exposure=float(active.size / sr.size),
        equity=equity,
        equity_index=pd.DatetimeIndex(dates),
    )
