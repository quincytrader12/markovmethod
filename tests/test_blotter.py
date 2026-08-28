"""Trading blotter — all positions + open orders, with close/cancel actions."""

from __future__ import annotations

from fastapi.testclient import TestClient

from markov_hedge_fund_method.broker import Account, OpenOrder, Position
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.web import AppState, create_app


class BlotterBroker:
    def __init__(self):
        self.closed: list[str] = []
        self.cancelled: list[str] = []

    def get_account(self):
        return Account(cash=1000.0, equity=2600.0, buying_power=5000.0,
                       status="ACTIVE", last_equity=2500.0)

    def list_positions(self):
        return [
            Position("AAPL", 10, 1900.0, 150.0, "long",
                     avg_entry=175.0, current_price=190.0, unrealized_plpc=0.086),
            Position("TSLA", 5, 1000.0, -80.0, "short",
                     avg_entry=220.0, current_price=200.0, unrealized_plpc=-0.074),
        ]

    def list_open_orders(self):
        return [OpenOrder(id="o1", symbol="NVDA", side="buy", type="limit", qty="3", status="new")]

    def close_position(self, symbol):
        self.closed.append(symbol)
        return f"closing {symbol}"

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)


def _paper():
    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER), demo=False)
    state.broker = BlotterBroker()
    return TestClient(create_app(state)), state


def _demo():
    state = AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)
    return TestClient(create_app(state)), state


def test_blotter_not_connected():
    client, _ = _demo()
    d = client.get("/api/blotter").json()
    assert d["connected"] is False
    assert d["positions"] == [] and d["openOrders"] == []


def test_blotter_positions_orders_and_daypl():
    client, _ = _paper()
    d = client.get("/api/blotter").json()
    assert d["connected"] is True and d["canTrade"] is True
    assert d["account"]["dayPl"] == 100.0                 # 2600 - 2500
    assert round(d["account"]["dayPlPct"], 3) == 0.04
    assert [p["symbol"] for p in d["positions"]] == ["AAPL", "TSLA"]
    aapl = d["positions"][0]
    assert aapl["unrealizedPl"] == 150.0
    assert aapl["avgEntry"] == 175.0 and aapl["currentPrice"] == 190.0
    assert d["openOrders"][0]["symbol"] == "NVDA"


def test_close_position_and_cancel_order():
    client, state = _paper()
    r = client.post("/api/positions/close", json={"symbol": "aapl"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert state.broker.closed == ["AAPL"]                # normalised to upper
    c = client.post("/api/orders/cancel", json={"id": "o1"})
    assert c.status_code == 200 and state.broker.cancelled == ["o1"]


def test_actions_require_connection():
    client, _ = _demo()
    assert client.post("/api/positions/close", json={"symbol": "AAPL"}).status_code == 403
    assert client.post("/api/orders/cancel", json={"id": "x"}).status_code == 403


def test_close_requires_symbol():
    client, _ = _paper()
    assert client.post("/api/positions/close", json={}).status_code == 400


def test_index_has_blotter_ui():
    client, _ = _demo()
    html = client.get("/").text
    assert "Trading Blotter" in html
    assert "openBlotter" in html and "/api/blotter" in html
