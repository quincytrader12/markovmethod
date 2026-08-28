"""Accounts screen + live account switching, driven headlessly (fake broker)."""

from __future__ import annotations

import asyncio

from textual.widgets import Checkbox, Input, Static

from markov_hedge_fund_method import tui as tuimod
from markov_hedge_fund_method.accounts import AccountStore
from markov_hedge_fund_method.broker import Account
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.tui import TerminalApp


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def set_password(self, s, u, p):
        self.store[(s, u)] = p

    def get_password(self, s, u):
        return self.store.get((s, u))

    def delete_password(self, s, u):
        self.store.pop((s, u), None)


class FakeBroker:
    def __init__(self, settings):
        self.settings = settings

    def get_account(self):
        return Account(cash=1.0, equity=1.0, buying_power=2.0, status="ACTIVE")

    def get_position(self, symbol):
        return None

    def list_open_orders(self):
        return []


async def _until(pilot, cond, tries=100):
    for _ in range(tries):
        if cond():
            return
        await pilot.pause()


def _store(tmp_path):
    s = AccountStore(keyring=FakeKeyring(), config_dir=str(tmp_path))
    s.add("swing", "K1", "S1", paper=True)
    s.add("livemain", "K2", "S2", paper=False, make_active=False)
    return s


def _screen_has(app, selector):
    scr = app.screen
    return scr.__class__.__name__ == "AccountsScreen" and bool(scr.query(selector))


def _app(monkeypatch, store):
    monkeypatch.setattr(tuimod, "get_history", lambda settings: tuimod.synthetic_close())
    monkeypatch.setattr(tuimod, "make_broker", lambda settings: FakeBroker(settings))
    app = TerminalApp(Settings(ticker="SPY", mode=Mode.PAPER), demo=False)
    app.accounts = store
    return app


def test_switch_account_rebuilds_broker(monkeypatch, tmp_path):
    app = _app(monkeypatch, _store(tmp_path))

    async def scenario():
        async with app.run_test(size=(140, 50)) as pilot:
            app._apply_account_switch("livemain")   # synchronous state change
            await pilot.pause()

    asyncio.run(scenario())
    assert app.settings.account == "livemain"
    assert app.settings.account_paper is False
    assert app.settings.api_key == "K2" and app.settings.api_secret == "S2"
    assert app.settings.paper is False           # live endpoint follows the profile
    assert isinstance(app.broker, FakeBroker)
    assert app.broker.settings.api_key == "K2"


def test_accounts_screen_lists_profiles(monkeypatch, tmp_path):
    app = _app(monkeypatch, _store(tmp_path))
    captured = {}

    async def scenario():
        async with app.run_test(size=(140, 50)) as pilot:
            app.action_accounts()
            await _until(pilot, lambda: _screen_has(app, "#acct-list"))
            listing = app.screen.query_one("#acct-list", Static).render()
            captured["text"] = getattr(listing, "plain", str(listing))

    asyncio.run(scenario())
    assert "swing" in captured["text"]
    assert "livemain" in captured["text"]


def test_add_account_via_screen(monkeypatch, tmp_path):
    store = AccountStore(keyring=FakeKeyring(), config_dir=str(tmp_path))
    app = _app(monkeypatch, store)

    async def scenario():
        async with app.run_test(size=(140, 50)) as pilot:
            app.action_accounts()
            await _until(pilot, lambda: _screen_has(app, "#acct_name"))
            app.screen.query_one("#acct_name", Input).value = "newport"
            app.screen.query_one("#acct_key", Input).value = "KX"
            app.screen.query_one("#acct_secret", Input).value = "SX"
            app.screen.query_one("#acct_paper", Checkbox).value = True
            await pilot.click("#acct_add")
            await _until(pilot, lambda: bool(store.list()))

    asyncio.run(scenario())
    assert [p.name for p in store.list()] == ["newport"]
    assert app.settings.account == "newport"     # newly added becomes active + connected
    assert app.settings.api_key == "KX"
