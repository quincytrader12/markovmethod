"""End-to-end wiring test for the execution panel via Textual's headless pilot.

Uses a fake broker (no network, no Alpaca account) but a REAL
`build_order_request`, so a submitted ticket is proven to construct a valid
alpaca-py order. No pytest-asyncio needed — scenarios run under asyncio.run.
"""

from __future__ import annotations

import asyncio

from textual.widgets import Checkbox, Input, Select

from markov_hedge_fund_method import tui as tuimod
from markov_hedge_fund_method.broker import Account, OrderResult
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.orders import build_order_request
from markov_hedge_fund_method.tui import TerminalApp


class FakeBroker:
    def __init__(self):
        self.submitted = []
        self.cancelled = 0

    def get_account(self):
        return Account(cash=1000.0, equity=1000.0, buying_power=2000.0, status="ACTIVE")

    def get_position(self, symbol):
        return None

    def list_open_orders(self):
        return []

    def submit_ticket(self, ticket):
        req = build_order_request(ticket)  # real construction — proves validity
        self.submitted.append((ticket, req))
        return OrderResult(id="fake-1", status="accepted", summary="ok")

    def cancel_all_orders(self):
        self.cancelled += 1
        return 3


def _make_app(mode: Mode, monkeypatch_target):
    # Avoid the network: feed synthetic prices in place of get_history.
    monkeypatch_target.setattr(tuimod, "get_history", lambda settings: tuimod.synthetic_close())
    settings = Settings(ticker="SPY", mode=mode, api_key="k", api_secret="s")
    app = TerminalApp(settings, demo=False)
    app.broker = FakeBroker()
    return app


def _run(coro):
    asyncio.run(coro)


def test_submit_market_order_reaches_broker(monkeypatch):
    app = _make_app(Mode.PAPER, monkeypatch)

    async def scenario():
        async with app.run_test(size=(140, 45)) as pilot:
            app.query_one("#exec_type", Select).value = "market"
            app.query_one("#exec_qty", Input).value = "5"
            await pilot.click("#exec_submit")
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario())
    assert len(app.broker.submitted) == 1
    ticket, req = app.broker.submitted[0]
    assert ticket.symbol == "SPY"
    assert getattr(req.type, "value", req.type) == "market"
    assert float(req.qty) == 5


def test_submit_bracket_order_builds_legs(monkeypatch):
    app = _make_app(Mode.PAPER, monkeypatch)

    async def scenario():
        async with app.run_test(size=(140, 45)) as pilot:
            app.query_one("#exec_type", Select).value = "limit"
            app.query_one("#exec_class", Select).value = "bracket"
            app.query_one("#exec_tif", Select).value = "gtc"
            app.query_one("#exec_qty", Input).value = "10"
            app.query_one("#exec_limit", Input).value = "100"
            app.query_one("#exec_tp", Input).value = "110"
            app.query_one("#exec_slstop", Input).value = "95"
            await pilot.click("#exec_submit")
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario())
    assert len(app.broker.submitted) == 1
    _, req = app.broker.submitted[0]
    assert getattr(req.order_class, "value", req.order_class) == "bracket"
    assert float(req.take_profit.limit_price) == 110
    assert float(req.stop_loss.stop_price) == 95


def test_cancel_all_reaches_broker(monkeypatch):
    app = _make_app(Mode.PAPER, monkeypatch)

    async def scenario():
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.click("#exec_cancel")
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario())
    assert app.broker.cancelled == 1


def test_dashboard_mode_blocks_submission(monkeypatch):
    app = _make_app(Mode.DASHBOARD, monkeypatch)

    async def scenario():
        async with app.run_test(size=(140, 45)) as pilot:
            app.query_one("#exec_qty", Input).value = "5"
            await pilot.click("#exec_submit")
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario())
    assert app.broker.submitted == []  # read-only mode never reached the broker


def test_invalid_ticket_does_not_reach_broker(monkeypatch):
    # limit order with no limit price -> validation error, broker untouched.
    app = _make_app(Mode.PAPER, monkeypatch)

    async def scenario():
        async with app.run_test(size=(140, 45)) as pilot:
            app.query_one("#exec_type", Select).value = "limit"
            app.query_one("#exec_qty", Input).value = "5"
            await pilot.click("#exec_submit")
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario())
    assert app.broker.submitted == []
