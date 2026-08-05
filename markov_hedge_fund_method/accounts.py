"""Multiple named Alpaca accounts (profiles) for different portfolios.

Each **profile** stores its own API key id + secret (in the OS keychain via
`keyring`) plus whether those are *paper* or *live* credentials. A small JSON
registry — names and paper/live flags only, never secrets — records the
profiles and which one is active. This lets one terminal connect several
portfolios (e.g. "swing", "longterm", "live-main") and switch between them at
runtime.

Secrets live ONLY in the keychain. The store takes an injectable keyring and
config directory, so it is fully unit-testable without touching the real
keychain or the user's home directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .config import SERVICE


@dataclass
class AccountProfile:
    name: str
    paper: bool
    active: bool


@dataclass
class ResolvedAccount:
    name: str
    key_id: str
    secret: str
    paper: bool


class KeyringUnavailable(RuntimeError):
    """Raised when credential storage is needed but `keyring` is not installed."""


def default_config_dir() -> str:
    """Cross-platform per-user config dir for the terminal's account registry."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "mamba-terminal")


class AccountStore:
    """Registry of named Alpaca profiles; secrets in keyring, metadata in JSON."""

    def __init__(self, keyring=None, config_dir: str | None = None):
        self._keyring = keyring
        self.config_dir = config_dir or default_config_dir()
        self.registry_path = os.path.join(self.config_dir, "accounts.json")

    # ── keyring (lazy, injectable) ───────────────────────────────────────────
    @property
    def keyring(self):
        if self._keyring is None:
            try:
                import keyring as _k
            except Exception as exc:  # noqa: BLE001
                raise KeyringUnavailable(
                    "The `keyring` package is required to store account credentials. "
                    'Install the terminal extra: pip install -e ".[terminal]".'
                ) from exc
            self._keyring = _k
        return self._keyring

    # ── registry i/o ─────────────────────────────────────────────────────────
    def _read_registry(self) -> dict:
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"active": None, "profiles": {}}
        data.setdefault("active", None)
        data.setdefault("profiles", {})
        return data

    def _write_registry(self, data: dict) -> None:
        os.makedirs(self.config_dir, exist_ok=True)
        tmp = self.registry_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.registry_path)  # atomic on the same filesystem

    @staticmethod
    def _u_key(name: str) -> str:
        return f"{name}/api_key_id"

    @staticmethod
    def _u_sec(name: str) -> str:
        return f"{name}/api_secret_key"

    # ── public API ───────────────────────────────────────────────────────────
    def list(self) -> list[AccountProfile]:
        reg = self._read_registry()
        active = reg.get("active")
        return [
            AccountProfile(name=n, paper=bool(p.get("paper", True)), active=(n == active))
            for n, p in sorted(reg["profiles"].items())
        ]

    def active(self) -> str | None:
        return self._read_registry().get("active")

    def exists(self, name: str) -> bool:
        return name in self._read_registry()["profiles"]

    def add(self, name: str, key_id: str, secret: str, paper: bool = True,
            make_active: bool = True) -> None:
        name = (name or "").strip()
        if not name:
            raise ValueError("Account name is required.")
        if "/" in name:
            raise ValueError("Account name cannot contain '/'.")
        if not key_id or not secret:
            raise ValueError("Both API key id and secret are required.")
        self.keyring.set_password(SERVICE, self._u_key(name), key_id)
        self.keyring.set_password(SERVICE, self._u_sec(name), secret)
        reg = self._read_registry()
        reg["profiles"][name] = {"paper": bool(paper)}
        if make_active or reg.get("active") is None:
            reg["active"] = name
        self._write_registry(reg)

    def remove(self, name: str) -> None:
        reg = self._read_registry()
        if name not in reg["profiles"]:
            raise KeyError(f"No account named {name!r}.")
        del reg["profiles"][name]
        if reg.get("active") == name:
            reg["active"] = next(iter(reg["profiles"]), None)
        self._write_registry(reg)
        for username in (self._u_key(name), self._u_sec(name)):
            try:
                self.keyring.delete_password(SERVICE, username)
            except Exception:  # noqa: BLE001 — best-effort secret cleanup
                pass

    def set_active(self, name: str) -> None:
        reg = self._read_registry()
        if name not in reg["profiles"]:
            raise KeyError(f"No account named {name!r}.")
        reg["active"] = name
        self._write_registry(reg)

    def credentials(self, name: str) -> tuple[str | None, str | None]:
        return (
            self.keyring.get_password(SERVICE, self._u_key(name)),
            self.keyring.get_password(SERVICE, self._u_sec(name)),
        )

    def resolve(self, name: str | None = None) -> ResolvedAccount | None:
        """Return the resolved (name, key, secret, paper) for `name` or the
        active profile, or None if there is no usable profile."""
        reg = self._read_registry()
        name = name or reg.get("active")
        if not name or name not in reg["profiles"]:
            return None
        key, sec = self.credentials(name)
        if not key or not sec:
            return None
        paper = bool(reg["profiles"][name].get("paper", True))
        return ResolvedAccount(name=name, key_id=key, secret=sec, paper=paper)
