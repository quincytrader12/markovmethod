"""Proof, not promises.

Runs the before-fix / after-fix comparison and renders a single figure:
  (left)  walk-forward equity curves — legacy overlapping vs stride-fixed
  (right) persistence diagonal — the fake (overlapping) vs true (stride) bars

Also returns a text summary with win rate, profit factor, and max drawdown.
Everything is walk-forward (the matrix never sees the day it trades).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .regime import STATES, label_regimes
from .markov2 import (
    BacktestResult,
    Strategy,
    compare_matrices,
    verify_labels,
    walk_forward,
)

_ROSE = "#c57f86"   # legacy / fake
_GREEN = "#84bba1"  # fixed / true
_GREY = "#a4abb7"


@dataclass
class Proof:
    ticker: str
    comparison: object
    verification: object
    legacy: BacktestResult
    fixed: BacktestResult
    image_path: str | None


def _fmt(bt: BacktestResult) -> str:
    pf = "inf" if bt.profit_factor == float("inf") else f"{bt.profit_factor:.2f}"
    return (f"  {bt.label:<22} Sharpe {bt.sharpe:5.2f} | win {bt.win_rate*100:4.1f}% | "
            f"PF {pf:>4} | maxDD {bt.max_drawdown*100:6.1f}% | trades {bt.n_trades}")


def generate_proof(
    close: pd.Series,
    ticker: str,
    *,
    window: int = 20,
    threshold: float = 0.02,
    strategy: Strategy = Strategy.STANDALONE,
    image_path: str | None = None,
    synthetic: bool = False,
) -> Proof:
    labels = label_regimes(close, window=window, threshold=threshold)
    comparison = compare_matrices(labels, window)
    verification = verify_labels(close, window=window, threshold=threshold)

    legacy = walk_forward(close, window=window, threshold=threshold, stride=1,
                          strategy=strategy, label="legacy (overlapping)")
    fixed = walk_forward(close, window=window, threshold=threshold, stride=window,
                         strategy=strategy, label="fixed (stride-sampled)")

    if image_path:
        _render(ticker, comparison, legacy, fixed, image_path, synthetic)

    return Proof(ticker, comparison, verification, legacy, fixed, image_path)


def _render(ticker, comparison, legacy, fixed, path, synthetic) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#0b0f0d", "axes.facecolor": "#0b0f0d",
        "savefig.facecolor": "#0b0f0d", "text.color": "#d8dee9",
        "axes.labelcolor": "#d8dee9", "xtick.color": "#a4abb7",
        "ytick.color": "#a4abb7", "axes.edgecolor": "#3a4048", "font.size": 10,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    tag = "  [SYNTHETIC DATA]" if synthetic else ""
    fig.suptitle(f"Markov 2.0 — before-fix vs after-fix · {ticker}{tag}",
                 color="#84bba1", fontsize=14, fontweight="bold")

    # left — equity curves
    ax1.plot(legacy.equity_index, legacy.equity, color=_ROSE, lw=1.6,
             label=f"legacy (overlap)  Sharpe {legacy.sharpe:.2f}")
    ax1.plot(fixed.equity_index, fixed.equity, color=_GREEN, lw=1.6,
             label=f"fixed (stride)    Sharpe {fixed.sharpe:.2f}")
    ax1.axhline(1.0, color="#3a4048", lw=0.8)
    ax1.set_title("Walk-forward equity (no lookahead)", color="#d8dee9")
    ax1.set_ylabel("growth of $1")
    ax1.legend(facecolor="#0b0f0d", edgecolor="#3a4048", labelcolor="#d8dee9", fontsize=9)
    ax1.grid(True, color="#1a1f1c", lw=0.6)

    # right — stickiness bars
    x = np.arange(3)
    w = 0.36
    ax2.bar(x - w / 2, comparison.stickiness_legacy * 100, w, color=_ROSE,
            label="overlapping (fake persistence)")
    ax2.bar(x + w / 2, comparison.stickiness_honest * 100, w, color=_GREEN,
            label="stride-sampled (true)")
    for xi, (a, b) in enumerate(zip(comparison.stickiness_legacy, comparison.stickiness_honest)):
        ax2.text(xi - w / 2, a * 100 + 1, f"{a*100:.0f}", ha="center", color=_ROSE, fontsize=9)
        ax2.text(xi + w / 2, b * 100 + 1, f"{b*100:.0f}", ha="center", color=_GREEN, fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(STATES)
    ax2.set_ylabel("persistence  P(stay) %")
    ax2.set_ylim(0, 100)
    ax2.set_title("Diagonal: overlapping inflates persistence", color="#d8dee9")
    ax2.legend(facecolor="#0b0f0d", edgecolor="#3a4048", labelcolor="#d8dee9", fontsize=9)
    ax2.grid(True, axis="y", color="#1a1f1c", lw=0.6)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    """CLI: `markov-proof --ticker SPY [--demo] [--image out.png]`."""
    import argparse

    from .config import load_settings
    from .market_data import get_history, synthetic_close

    ap = argparse.ArgumentParser(prog="markov-proof")
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--strategy", choices=[s.value for s in Strategy], default=Strategy.STANDALONE.value)
    ap.add_argument("--demo", action="store_true", help="Offline synthetic data")
    ap.add_argument("--image", default="markov_proof.png", help="Output PNG path")
    args = ap.parse_args()

    if args.demo:
        close, ticker, synthetic = synthetic_close(), f"{args.ticker} (demo)", True
    else:
        close = get_history(load_settings(ticker=args.ticker, years=args.years))
        ticker, synthetic = args.ticker, False

    proof = generate_proof(
        close, ticker, window=args.window, threshold=args.threshold,
        strategy=Strategy(args.strategy), image_path=args.image, synthetic=synthetic,
    )
    print(summary_text(proof))
    print(f"\nfigure written to {args.image}")
    return 0


def summary_text(proof: Proof) -> str:
    lines = [f"Markov 2.0 proof — {proof.ticker}", ""]
    lines.append("Persistence diagonal (Bear / Sideways / Bull):")
    lines.append(f"  overlapping (fake): {np.round(proof.comparison.stickiness_legacy*100,1)}")
    lines.append(f"  stride  (true)    : {np.round(proof.comparison.stickiness_honest*100,1)}")
    lines.append(f"  inflation (pp)    : {np.round(proof.comparison.inflation,1)}")
    lines.append("")
    lines.append(f"Label verification: {'PASSED' if proof.verification.passed else 'FAILED'}")
    lines.append("")
    lines.append("Walk-forward backtest:")
    lines.append(_fmt(proof.legacy))
    lines.append(_fmt(proof.fixed))
    lines.append("")
    lines.append('"Backtests flatter. The fixed matrix shows uglier, truer numbers —')
    lines.append(' those are the only ones worth trading."')
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.exit(main())
