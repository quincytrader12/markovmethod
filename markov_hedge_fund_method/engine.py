"""Pure analysis step: a close-price series in, a Snapshot out.

No I/O, no side effects — this is the same math as the CLI skill
(`regime.py`), packaged as a single immutable snapshot the TUI renders and,
later, an executor can act on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .regime import (
    STATES,
    build_transition_matrix,
    label_regimes,
    signal_from_matrix,
    stationary_distribution,
)
from .markov2 import (
    MatrixComparison,
    Strategy,
    Verification,
    compare_matrices,
    desired_position,
    verify_labels,
)


@dataclass
class Snapshot:
    ticker: str
    n_rows: int
    start: str
    end: str
    matrix: np.ndarray          # 3x3 transition matrix P
    stationary: np.ndarray      # length-3 long-run mix
    current_state: int          # 0=Bear, 1=Sideways, 2=Bull (regime.STATES order)
    signal: float               # P(Bull next) - P(Bear next) from current state
    desired_position: int       # sign of signal: -1 short / 0 flat / +1 long
    last_price: float

    @property
    def current_state_name(self) -> str:
        return STATES[self.current_state]

    @property
    def desired_label(self) -> str:
        return {1: "LONG", 0: "FLAT", -1: "SHORT"}[self.desired_position]


def analyze(close: pd.Series, ticker: str, window: int = 20, threshold: float = 0.02) -> Snapshot:
    labels = label_regimes(close, window=window, threshold=threshold)
    P = build_transition_matrix(labels)
    pi = stationary_distribution(P)
    current = int(labels.iloc[-1])
    signal = signal_from_matrix(P, current)
    return Snapshot(
        ticker=ticker,
        n_rows=int(len(close)),
        start=str(close.index.min().date()),
        end=str(close.index.max().date()),
        matrix=P,
        stationary=pi,
        current_state=current,
        signal=signal,
        desired_position=int(np.sign(signal)),
        last_price=float(close.iloc[-1]),
    )


@dataclass
class SnapshotV2:
    """Markov 2.0 snapshot: honest (stride-sampled) matrix + the three fixes."""
    ticker: str
    n_rows: int
    start: str
    end: str
    honest_matrix: np.ndarray       # stride-sampled — the truthful one
    stationary: np.ndarray
    comparison: MatrixComparison    # legacy vs honest (Fix 1)
    verification: Verification      # label self-check (Fix 2)
    current_state: int
    signal: float                   # from the HONEST matrix, not the inflated one
    strategy: Strategy              # FILTER / STANDALONE (Fix 3)
    target_position: float          # position under that strategy
    last_price: float

    @property
    def current_state_name(self) -> str:
        return STATES[self.current_state]

    @property
    def target_label(self) -> str:
        if self.strategy is Strategy.FILTER:
            return {1.0: "LONG (allowed)", 0.0: "FLAT (gated off)", -1.0: "SHORT (allowed)"}.get(
                self.target_position, f"{self.target_position:+.2f}")
        return f"{self.target_position:+.2f}x"


def analyze2(
    close: pd.Series,
    ticker: str,
    *,
    window: int = 20,
    threshold: float = 0.02,
    strategy: Strategy = Strategy.FILTER,
    signal_threshold: float = 0.15,
    size_cap: float = 1.0,
    size_scale: float = 0.5,
) -> SnapshotV2:
    labels = label_regimes(close, window=window, threshold=threshold)
    comparison = compare_matrices(labels, window)
    verification = verify_labels(close, window=window, threshold=threshold)
    honest = comparison.honest
    pi = stationary_distribution(honest)
    current = int(labels.iloc[-1])
    signal = signal_from_matrix(honest, current)  # honest signal drives the trade
    target = desired_position(
        signal, strategy, threshold=signal_threshold, cap=size_cap, scale=size_scale
    )
    return SnapshotV2(
        ticker=ticker,
        n_rows=int(len(close)),
        start=str(close.index.min().date()),
        end=str(close.index.max().date()),
        honest_matrix=honest,
        stationary=pi,
        comparison=comparison,
        verification=verification,
        current_state=current,
        signal=signal,
        strategy=strategy,
        target_position=target,
        last_price=float(close.iloc[-1]),
    )
