"""Runtime configuration for the terminal: ticker, model params, execution
mode, and Alpaca credentials.

Keys are NEVER hardcoded or bundled. They are read, in order, from:
  1. environment variables ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY
  2. the OS keychain via `keyring` (service name below)

Execution mode is the safety gate:
  DASHBOARD -> read-only. Data + account/positions shown, NO orders placed.
  PAPER     -> orders placed against an Alpaca *paper* account (fake money).
  LIVE      -> orders placed against a real-money account.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from .markov2 import Strategy

SERVICE = "markov-hedge-fund-method"


class Mode(str, Enum):
    DASHBOARD = "dashboard"  # read-only, no orders
    BACKTEST = "backtest"    # the strategy lab; read-only like dashboard
    PAPER = "paper"          # orders on paper account
    LIVE = "live"            # orders on live account


@dataclass
class Settings:
    ticker: str = "SPY"
    window: int = 20
    threshold: float = 0.02
    years: int = 10
    mode: Mode = Mode.DASHBOARD
    poll_seconds: int = 60
    # Fraction of buying power to deploy per side when auto-trading is enabled.
    target_notional_pct: float = 0.10
    # Markov 2.0 — Fix 3: how the signal becomes a position.
    strategy: Strategy = Strategy.FILTER
    signal_threshold: float = 0.15   # FILTER: |signal| must clear this to act
    size_cap: float = 1.0            # STANDALONE: max |position|
    size_scale: float = 0.5          # STANDALONE: signal/scale before the cap
    api_key: str | None = None
    api_secret: str | None = None
    # Where intraday bars come from. Execution is always Alpaca — the venue you
    # trade on and the feed you analyse are unrelated choices, and Alpaca's order
    # API neither knows nor cares which data led to the ticket.
    #
    #   "auto"   Yahoo when it works, Alpaca otherwise. The default, because
    #            Alpaca's free tier is IEX-only — roughly 2-3% of consolidated
    #            volume — so having credentials would otherwise make intraday
    #            bars *worse* than not having them.
    #   "yahoo"  consolidated tape, free, ~15 min delayed, unofficial endpoint.
    #   "alpaca" whatever the account's data plan entitles it to: IEX on the free
    #            tier, full SIP on a paid one, and real-time either way.
    intraday_source: str = "auto"
    # Active Alpaca profile (see accounts.py); None = legacy env/keychain keys.
    account: str | None = None
    # Whether the active profile's keys are paper (True) or live (False/None).
    account_paper: bool | None = None

    @property
    def paper(self) -> bool:
        """Alpaca TradingClient `paper` flag — the connected profile decides the
        endpoint (paper vs live); without a profile, anything but LIVE mode is
        paper."""
        if self.account_paper is not None:
            return self.account_paper
        return self.mode is not Mode.LIVE

    @property
    def can_trade(self) -> bool:
        """True only when the mode explicitly permits order placement."""
        return self.mode in (Mode.PAPER, Mode.LIVE)

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)


def _load_keys() -> tuple[str | None, str | None]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    sec = os.environ.get("ALPACA_API_SECRET_KEY")
    if key and sec:
        return key, sec
    try:
        import keyring  # optional dependency
        key = key or keyring.get_password(SERVICE, "api_key_id")
        sec = sec or keyring.get_password(SERVICE, "api_secret_key")
    except BaseException:  # noqa: BLE001 — a broken keyring backend can panic
        # Missing/misconfigured keychain must never crash startup — degrade to
        # "no stored keys" and let the user connect an account from the UI.
        pass
    return key, sec


def load_settings(account: str | None = None, **overrides) -> Settings:
    """Build Settings, resolving credentials from (in order):

      1. a named/active account profile (see accounts.py), else
      2. environment variables / the legacy single-account keychain slot.

    `None` overrides are ignored so callers can pass argparse defaults freely.
    """
    key = sec = None
    acct_name = acct_paper = None
    try:
        from .accounts import AccountStore
        resolved = AccountStore().resolve(account)
    except BaseException:  # noqa: BLE001 — accounts optional; keyring may panic
        resolved = None
    if resolved is not None:
        key, sec = resolved.key_id, resolved.secret
        acct_name, acct_paper = resolved.name, resolved.paper
    else:
        key, sec = _load_keys()

    s = Settings(api_key=key, api_secret=sec, account=acct_name, account_paper=acct_paper)
    for name, value in overrides.items():
        if value is not None and hasattr(s, name):
            setattr(s, name, value)
    return s


def save_keys(api_key: str, api_secret: str) -> None:
    """Persist credentials to the OS keychain (never to disk in plaintext)."""
    import keyring
    keyring.set_password(SERVICE, "api_key_id", api_key)
    keyring.set_password(SERVICE, "api_secret_key", api_secret)
