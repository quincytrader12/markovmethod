"""Web HUD backend (FastAPI) — endpoints exercised with TestClient.

Demo mode drives the data endpoints offline (synthetic prices); a fake broker
+ injected account store cover portfolio/orders/accounts with no network.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from markov_hedge_fund_method.accounts import AccountStore
from markov_hedge_fund_method.broker import Account, OrderResult, Position
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.orders import build_order_request
from markov_hedge_fund_method.web import AppState, create_app


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
    def __init__(self):
        self.submitted = []
        self.cancelled = 0

    def get_account(self):
        return Account(cash=1000.0, equity=2500.0, buying_power=5000.0, status="ACTIVE")

    def get_position(self, symbol):
        return Position(symbol=symbol, qty=3, market_value=900.0, unrealized_pl=42.0, side="long")

    def list_open_orders(self):
        return []

    def submit_ticket(self, ticket):
        build_order_request(ticket)  # prove validity
        self.submitted.append(ticket)
        return OrderResult(id="web-1", status="accepted", summary="ok")

    def cancel_all_orders(self):
        self.cancelled += 1
        return 2


def _demo_client():
    state = AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)
    return TestClient(create_app(state)), state


def _paper_client(tmp_path):
    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER), demo=False)
    state.accounts = AccountStore(keyring=FakeKeyring(), config_dir=str(tmp_path))
    state.broker = FakeBroker()
    return TestClient(create_app(state)), state


def test_config_and_state_demo():
    client, _ = _demo_client()
    cfg = client.get("/api/config").json()
    assert cfg["demo"] is True and cfg["mode"] == "dashboard"

    st = client.get("/api/state", params={"symbol": "spy"}).json()
    assert st["ticker"] == "SPY"
    assert st["regime"] in ("bull", "bear", "sideways")
    assert 0 <= st["greedFear"]["score"] <= 100
    assert len(st["chart"]["bars"]) > 50
    assert all(abs(sum(row) - 1.0) < 1e-6 for row in st["matrix"])
    assert "dataSource" in st


def test_index_served():
    client, _ = _demo_client()
    r = client.get("/")
    assert r.status_code == 200
    assert "MAMBA" in r.text and "canvas" in r.text


def test_orders_require_connection():
    client, _ = _demo_client()               # demo → no broker
    r = client.post("/api/orders", json={"symbol": "SPY", "qty": 1})
    assert r.status_code == 403


def test_order_submit_and_validation(tmp_path):
    client, state = _paper_client(tmp_path)
    ok = client.post("/api/orders", json={"symbol": "SPY", "order_type": "market", "qty": 5})
    assert ok.status_code == 200 and ok.json()["id"] == "web-1"
    assert len(state.broker.submitted) == 1

    bad = client.post("/api/orders", json={"symbol": "SPY", "order_type": "limit", "qty": 5})
    assert bad.status_code == 400                       # limit without a price
    assert len(state.broker.submitted) == 1             # never reached the broker

    cancel = client.post("/api/orders/cancel_all")
    assert cancel.status_code == 200 and cancel.json()["cancelled"] == 2


def test_portfolio(tmp_path):
    client, _ = _paper_client(tmp_path)
    pf = client.get("/api/portfolio", params={"symbol": "SPY"}).json()
    assert pf["connected"] is True
    assert pf["account"]["equity"] == 2500.0
    assert pf["position"]["symbol"] == "SPY"


def test_accounts_crud(tmp_path):
    client, state = _paper_client(tmp_path)
    assert client.get("/api/accounts").json()["accounts"] == []

    r = client.post("/api/accounts", json={"name": "swing", "key_id": "K1", "secret": "S1", "paper": True})
    body = r.json()
    assert [a["name"] for a in body["accounts"]] == ["swing"]
    assert body["active"] == "swing"

    client.post("/api/accounts", json={"name": "live1", "key_id": "K2", "secret": "S2", "paper": False})
    active = client.post("/api/accounts/active", json={"name": "live1"}).json()
    assert active["active"] == "live1"
    assert state.settings.account == "live1" and state.settings.account_paper is False

    gone = client.delete("/api/accounts/swing").json()
    assert [a["name"] for a in gone["accounts"]] == ["live1"]


def test_add_account_validation(tmp_path):
    client, _ = _paper_client(tmp_path)
    r = client.post("/api/accounts", json={"name": "bad/name", "key_id": "K", "secret": "S"})
    assert r.status_code == 400


def test_quote_is_cheap_and_shaped():
    client, _ = _demo_client()
    q = client.get("/api/quote", params={"symbol": "nvda"}).json()
    assert q["ticker"] == "NVDA"
    assert q["regime"] in ("bull", "bear", "sideways")
    assert isinstance(q["lastPrice"], (int, float))
    assert "chart" not in q  # the cheap endpoint omits the heavy series


def test_search_prefix_and_custom_ticker():
    client, _ = _demo_client()
    res = client.get("/api/search", params={"q": "aa"}).json()["results"]
    assert "AAPL" in res and all(isinstance(s, str) for s in res)
    # a ticker not in the bundled universe is still offered verbatim
    custom = client.get("/api/search", params={"q": "ZZZZ"}).json()["results"]
    assert custom[0] == "ZZZZ"


def test_state_chart_is_wide_enough_for_timeframes():
    client, _ = _demo_client()
    bars = client.get("/api/state", params={"symbol": "SPY"}).json()["chart"]["bars"]
    assert len(bars) >= 504  # enough history for the 2Y timeframe, sliced client-side


def test_news_demo_sample_feed():
    client, _ = _demo_client()
    body = client.get("/api/news", params={"symbol": "TSLA"}).json()
    items = body["items"]
    assert body["symbol"] == "TSLA" and len(items) >= 1
    assert all(i["sentiment"] in ("bullish", "bearish", "neutral") for i in items)
    assert all(i["sample"] is True for i in items)  # offline → sample feed


def test_news_sentiment_classifier():
    from markov_hedge_fund_method.news import classify
    assert classify("Company beats earnings and shares surge to record") == "bullish"
    assert classify("Stock plunges after downgrade and lawsuit probe") == "bearish"
    assert classify("Company holds annual meeting today") == "neutral"
