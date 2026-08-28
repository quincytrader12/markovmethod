"""AccountStore: named multi-profile credential storage (fake keyring + tmp dir)."""

from __future__ import annotations

import pytest

from markov_hedge_fund_method.accounts import AccountStore
from markov_hedge_fund_method.config import SERVICE


class FakeKeyring:
    """In-memory stand-in for the `keyring` module."""

    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}

    def set_password(self, service, username, password):
        self.store[(service, username)] = password

    def get_password(self, service, username):
        return self.store.get((service, username))

    def delete_password(self, service, username):
        self.store.pop((service, username), None)


def _store(tmp_path):
    return AccountStore(keyring=FakeKeyring(), config_dir=str(tmp_path))


def test_add_lists_and_activates(tmp_path):
    s = _store(tmp_path)
    assert s.list() == []
    s.add("swing", "KID1", "SEC1", paper=True)
    profiles = s.list()
    assert [p.name for p in profiles] == ["swing"]
    assert profiles[0].paper is True
    assert profiles[0].active is True          # first account becomes active
    assert s.active() == "swing"


def test_multiple_accounts_switch(tmp_path):
    s = _store(tmp_path)
    s.add("swing", "K1", "S1", paper=True)
    s.add("longterm", "K2", "S2", paper=True, make_active=False)
    s.add("live-main", "K3", "S3", paper=False, make_active=False)
    names = {p.name for p in s.list()}
    assert names == {"swing", "longterm", "live-main"}
    assert s.active() == "swing"               # still the first
    s.set_active("live-main")
    assert s.active() == "live-main"
    resolved = s.resolve()                      # active resolves to live-main
    assert resolved.name == "live-main"
    assert resolved.key_id == "K3" and resolved.secret == "S3"
    assert resolved.paper is False


def test_resolve_named_vs_active(tmp_path):
    s = _store(tmp_path)
    s.add("a", "KA", "SA", paper=True)
    s.add("b", "KB", "SB", paper=False, make_active=False)
    assert s.resolve("b").key_id == "KB"
    assert s.resolve().name == "a"             # active is still 'a'
    assert s.resolve("missing") is None


def test_secrets_are_isolated_per_account(tmp_path):
    kr = FakeKeyring()
    s = AccountStore(keyring=kr, config_dir=str(tmp_path))
    s.add("swing", "K1", "S1")
    s.add("live", "K2", "S2", make_active=False)
    # Stored under per-account usernames, never colliding.
    assert kr.get_password(SERVICE, "swing/api_key_id") == "K1"
    assert kr.get_password(SERVICE, "live/api_secret_key") == "S2"


def test_remove_clears_secrets_and_reassigns_active(tmp_path):
    kr = FakeKeyring()
    s = AccountStore(keyring=kr, config_dir=str(tmp_path))
    s.add("swing", "K1", "S1")
    s.add("longterm", "K2", "S2", make_active=False)
    s.set_active("swing")
    s.remove("swing")
    assert kr.get_password(SERVICE, "swing/api_key_id") is None   # secret gone
    assert [p.name for p in s.list()] == ["longterm"]
    assert s.active() == "longterm"            # active reassigned
    s.remove("longterm")
    assert s.list() == []
    assert s.active() is None


def test_validation(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(ValueError):
        s.add("", "K", "S")
    with pytest.raises(ValueError):
        s.add("bad/name", "K", "S")
    with pytest.raises(ValueError):
        s.add("ok", "", "S")
    with pytest.raises(KeyError):
        s.remove("nope")
    with pytest.raises(KeyError):
        s.set_active("nope")


def test_registry_persists_across_instances(tmp_path):
    kr = FakeKeyring()
    AccountStore(keyring=kr, config_dir=str(tmp_path)).add("swing", "K1", "S1")
    # A fresh store over the same dir + keyring sees the saved profile.
    s2 = AccountStore(keyring=kr, config_dir=str(tmp_path))
    assert [p.name for p in s2.list()] == ["swing"]
    assert s2.resolve("swing").secret == "S1"
