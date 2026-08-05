"""Entry point for the terminal app (`markov-terminal`).

    markov-terminal --ticker SPY                 # dashboard, data-only or Alpaca-backed
    markov-terminal --ticker AAPL --demo         # offline synthetic data, no network
    markov-terminal --ticker SPY --mode paper    # enable paper auto-trading (needs keys)
    markov-terminal --save-keys KEYID SECRET     # store Alpaca keys in the OS keychain

Mode is the safety gate. It defaults to `dashboard` (read-only); `paper` and
`live` require credentials and are the only modes that can place orders.
"""

from __future__ import annotations

import argparse
import sys

from .config import Mode, load_settings, save_keys
from .markov2 import Strategy


def main() -> int:
    parser = argparse.ArgumentParser(prog="markov-terminal")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.DASHBOARD.value)
    parser.add_argument("--strategy", choices=[s.value for s in Strategy], default=Strategy.FILTER.value,
                        help="filter = regime gates a strategy; standalone = trade the signal directly")
    parser.add_argument("--signal-threshold", type=float, default=0.15, dest="signal_threshold",
                        help="FILTER: |signal| must clear this to act")
    parser.add_argument("--cap", type=float, default=1.0, dest="size_cap",
                        help="STANDALONE: max |position|")
    parser.add_argument("--poll", type=int, default=60, dest="poll_seconds",
                        help="Seconds between refreshes")
    parser.add_argument("--demo", action="store_true",
                        help="Run on offline synthetic data (no network, no broker)")
    parser.add_argument("--save-keys", nargs=2, metavar=("KEY_ID", "SECRET"),
                        help="Store Alpaca credentials in the OS keychain and exit")
    args = parser.parse_args()

    if args.save_keys:
        save_keys(*args.save_keys)
        print("Alpaca credentials saved to the OS keychain.")
        return 0

    settings = load_settings(
        ticker=args.ticker,
        years=args.years,
        window=args.window,
        threshold=args.threshold,
        mode=Mode(args.mode),
        strategy=Strategy(args.strategy),
        signal_threshold=args.signal_threshold,
        size_cap=args.size_cap,
        poll_seconds=args.poll_seconds,
    )

    if settings.can_trade and not settings.has_credentials:
        print(
            f"Mode '{settings.mode.value}' needs Alpaca credentials, but none were found.\n"
            "Set ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY, or run:\n"
            "  markov-terminal --save-keys <KEY_ID> <SECRET>",
            file=sys.stderr,
        )
        return 2

    if settings.mode is Mode.LIVE:
        print("!! LIVE mode places REAL-MONEY orders. Ctrl-C now to abort.")

    # Import the app lazily so `--save-keys` works without a TTY / textual.
    from .tui import TerminalApp

    TerminalApp(settings, demo=args.demo).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
