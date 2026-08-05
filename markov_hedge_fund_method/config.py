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

SERVICE = "markov-hedge-fund-method"


class Mode(str, Enum):
    DASHBOARD = "dashboard"  # read-only, no orders
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
    api_key: str | None = None
    api_secret: str | None = None

    @property
    def paper(self) -> bool:
        """Alpaca TradingClient `paper` flag — anything but LIVE is paper."""
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
    except Exception:
        pass
    return key, sec


def load_settings(**overrides) -> Settings:
    """Build Settings from env/keychain, with explicit overrides winning.

    `None` overrides are ignored so callers can pass argparse defaults freely.
    """
    key, sec = _load_keys()
    s = Settings(api_key=key, api_secret=sec)
    for name, value in overrides.items():
        if value is not None and hasattr(s, name):
            setattr(s, name, value)
    return s


def save_keys(api_key: str, api_secret: str) -> None:
    """Persist credentials to the OS keychain (never to disk in plaintext)."""
    import keyring
    keyring.set_password(SERVICE, "api_key_id", api_key)
    keyring.set_password(SERVICE, "api_secret_key", api_secret)
