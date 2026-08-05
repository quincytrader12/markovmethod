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
