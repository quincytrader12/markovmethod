"""Entry point for the Mamba Terminal app (`mamba-terminal`).

    mamba-terminal --ticker SPY                 # dashboard, data-only or Alpaca-backed
    mamba-terminal --ticker AAPL --demo         # offline synthetic data, no network
    mamba-terminal --ticker SPY --mode paper    # enable paper auto-trading (needs keys)
    mamba-terminal --save-keys KEYID SECRET --account swing        # save keys to a profile
    mamba-terminal --save-keys KEYID SECRET --account live1 --live # save a LIVE-key profile
    mamba-terminal --account live1 --mode live  # run against a named account
    mamba-terminal --list-accounts              # show saved account profiles

Mode is the safety gate. It defaults to `dashboard` (read-only); `paper` and
`live` require credentials and are the only modes that can place orders.

Multiple portfolios: each Alpaca account is a named *profile* (keys in the OS
keychain, paper/live flag in a local registry). Switch between them at runtime
from the Accounts screen (press `a`) or with --account on the CLI.
"""

from __future__ import annotations

import argparse
import sys

from .config import Mode, load_settings
from .markov2 import Strategy


def main() -> int:
    parser = argparse.ArgumentParser(prog="mamba-terminal")
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
                        help="Store Alpaca credentials in a named account profile and exit")
    parser.add_argument("--account", metavar="NAME",
                        help="Use (or, with --save-keys, name) an Alpaca account profile")
    parser.add_argument("--live", action="store_true",
                        help="With --save-keys: mark the saved keys as LIVE (default: paper)")
    parser.add_argument("--list-accounts", action="store_true",
                        help="List saved account profiles and exit")
    parser.add_argument("--remove-account", metavar="NAME",
                        help="Delete a saved account profile and exit")
    args = parser.parse_args()

    from .accounts import AccountStore
    store = AccountStore()

    if args.list_accounts:
        profiles = store.list()
        if not profiles:
            print("No account profiles saved. Add one with:\n"
                  "  mamba-terminal --save-keys <KEY_ID> <SECRET> --account <name>")
        else:
            print("Saved Alpaca accounts (* = active):")
            for p in profiles:
                tag = "paper" if p.paper else "LIVE"
                print(f"  {'*' if p.active else ' '} {p.name:<16s} [{tag}]")
        return 0

    if args.remove_account:
        try:
            store.remove(args.remove_account)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"Removed account {args.remove_account!r}.")
        return 0

    if args.save_keys:
        name = args.account or "default"
        try:
            store.add(name, args.save_keys[0], args.save_keys[1], paper=not args.live)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not save keys: {exc}", file=sys.stderr)
            return 2
        tag = "LIVE" if args.live else "paper"
        print(f"Saved Alpaca keys to account {name!r} [{tag}] — now the active account.")
        return 0

    settings = load_settings(
        account=args.account,
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
            "Set ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY, add an account:\n"
            "  mamba-terminal --save-keys <KEY_ID> <SECRET> --account <name>\n"
            "or add/switch accounts live from the Accounts screen (press 'a').",
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
