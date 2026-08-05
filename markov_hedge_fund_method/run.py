"""CLI entry point: fetch -> label -> matrix -> stationary -> walk-forward.

Usage:
    uv run python -m markov_hedge_fund_method.run --ticker SPY --years 10 --window 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .regime import (
    STATES,
    label_regimes,
    build_transition_matrix,
    stationary_distribution,
    walk_forward_backtest,
)

HMM_FLAG_FILE = Path(__file__).resolve().parent.parent / ".hmm_available"


def _hmm_available() -> bool:
    if HMM_FLAG_FILE.exists():
        return HMM_FLAG_FILE.read_text().strip().lower() == "true"
    try:
        import hmmlearn  # noqa: F401
        return True
    except ImportError:
        return False


def _fetch_with_retry(ticker: str, years: int) -> pd.DataFrame:
    """Fetch via yfinance with one retry; raise on persistent empty."""
    import yfinance as yf

    end = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    start = end - pd.DateOffset(years=years)

    for attempt in (1, 2):
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ! yfinance error on attempt {attempt}: {exc}")
            df = pd.DataFrame()

        if not df.empty:
            return df

        if attempt == 1:
            print("  ! yfinance returned empty data — retrying in 30s.")
            time.sleep(30)

    raise RuntimeError(
        f"yfinance returned empty data for {ticker} after retry. "
        "Yahoo may be rate-limiting. Try again in a few minutes."
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="markov-hedge-fund-method")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--window", type=int, default=20, help="Rolling-return window in trading days")
    parser.add_argument("--threshold", type=float, default=0.02, help="Regime label threshold on rolling return")
    parser.add_argument("--no-hmm", action="store_true", help="Skip HMM fit even if hmmlearn is available")
    args = parser.parse_args()

    print(f"\nmarkov-hedge-fund-method — ticker={args.ticker} years={args.years} window={args.window}")
    print(f"  fetching {args.ticker} from Yahoo Finance...")
    df = _fetch_with_retry(args.ticker, args.years)

    # Robust to yfinance returning a MultiIndex column frame on some installs.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].dropna()
    print(f"  fetched {len(close)} rows | {close.index.min().date()} -> {close.index.max().date()}")

    labels = label_regimes(close, window=args.window, threshold=args.threshold)
    P = build_transition_matrix(labels)
    pi = stationary_distribution(P)

    print("\nTransition matrix (rows = from, cols = to):")
    print(f"            {STATES[0]:>9s} {STATES[1]:>9s} {STATES[2]:>9s}")
    for i, from_state in enumerate(STATES):
        row = "  ".join(f"{P[i, j]*100:7.2f}%" for j in range(3))
        print(f"  {from_state:>9s}  {row}")

    print("\nPersistence diagonal:")
    print(f"  {STATES[0]} -> {STATES[0]}: {P[0,0]*100:.2f}%")
    print(f"  {STATES[1]} -> {STATES[1]}: {P[1,1]*100:.2f}%")
    print(f"  {STATES[2]} -> {STATES[2]}: {P[2,2]*100:.2f}%")

    print("\nStationary distribution (long-run regime mix):")
    for s, p in zip(STATES, pi):
        print(f"  {s:>9s}: {p*100:.2f}%")

    print("\nWalk-forward backtest (re-estimating matrix at every step, no lookahead)...")
    result = walk_forward_backtest(close, labels)
    sharpe = result["sharpe"]
    mdd = result["max_drawdown"]
    if np.isfinite(sharpe):
        print(f"  Sharpe (annualised, walk-forward): {sharpe:.3f}")
    else:
        print("  Sharpe: NaN (insufficient data — try a longer history or different ticker)")
    if np.isfinite(mdd):
        print(f"  Max drawdown:                       {mdd*100:.2f}%")
    else:
        print("  Max drawdown: NaN")
    print(f"  Trades evaluated: {result['n_trades']}")

    if not args.no_hmm and _hmm_available():
        print("\nFitting Hidden Markov Model (Baum-Welch + Viterbi via hmmlearn)...")
        try:
            from .hmm_extension import fit_hmm
            returns = close.pct_change().dropna()
            model, hidden = fit_hmm(returns, n_components=3)
            if model is None:
                print("  HMM extension skipped (hmmlearn import failed at runtime).")
            else:
                means = np.array([model.means_[k][0] for k in range(model.n_components)])
                order = np.argsort(means)
                labels_for_hmm = ["Bear (lowest mean return)", "Sideways", "Bull (highest mean return)"]
                print("  HMM regime mean daily returns (sorted):")
                for rank, k in enumerate(order):
                    print(f"    {labels_for_hmm[rank]:<30s} state {k}: {means[k]*100:+.3f}% per day")
                print("  Note: Baum-Welch finds local maxima. For production fit several random_state values.")
        except Exception as exc:  # noqa: BLE001
            print(f"  HMM extension skipped at runtime: {exc}")
    else:
        print("\nHMM extension skipped (optional); observable Markov model installed successfully.")

    print("\n----------------------------------------------------------------")
    print(" Framework: Roan (@RohOnChain). Installed as a Claude Code skill")
    print(" by Lewis Jackson. Backtests are historical, not forward-looking.")
    print("----------------------------------------------------------------\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
