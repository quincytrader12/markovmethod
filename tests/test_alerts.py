"""Alerts + risk automation — regime flips, price alerts, kill switch, loss limit."""

from __future__ import annotations

from fastapi.testclient import TestClient

from markov_hedge_fund_method.alerts import AlertEngine
from markov_hedge_fund_method.broker import OrderResult
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.orders import build_order_request
from markov_hedge_fund_method.web import AppState, create_app


class OrderBroker:
    def submit_ticket(self, ticket):
        build_order_request(ticket)
        return OrderResult(id="x1", status="accepted", summary="ok")

    def cancel_all_orders(self):
        return 0

    def list_positions(self):
        return []


# ── engine ───────────────────────────────────────────────────────────────────
def test_regime_flip_only_on_change():
    e = AlertEngine()
    assert e.check_regimes({"SPY": "bull"}) == []          # first sighting seeds
    assert e.check_regimes({"SPY": "bull"}) == []          # unchanged
    evs = e.check_regimes({"SPY": "bear"})                  # flip!
    assert len(evs) == 1
    assert evs[0]["from"] == "bull" and evs[0]["to"] == "bear"
    assert evs[0]["type"] == "regime_flip"


def test_price_alert_one_shot():
    e = AlertEngine()
    e.add_price_alert("AAPL", "above", 200)
    assert e.check_prices({"AAPL": 199}) == []             # not hit
    hit = e.check_prices({"AAPL": 201})                     # crosses
    assert len(hit) == 1 and hit[0]["type"] == "price"
    assert e.check_prices({"AAPL": 205}) == []             # one-shot: already fired


def test_loss_limit_trips_kill():
    e = AlertEngine()
    e.set_loss_limit(500)
    assert e.check_loss_limit(-200) is None and e.halted is False
    ev = e.check_loss_limit(-600)
    assert ev is not None and e.halted is True and ev["type"] == "kill"


# ── endpoints ────────────────────────────────────────────────────────────────
def _demo():
    return TestClient(create_app(AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)))


def test_alerts_endpoint_shape():
    client = _demo()
    d = client.get("/api/alerts", params={"symbols": "SPY,QQQ"}).json()
    assert set(d) >= {"events", "recent", "halted", "lossLimit", "priceAlerts"}
    assert d["halted"] is False


def test_price_alert_crud_endpoints():
    client = _demo()
    r = client.post("/api/alerts/price", json={"symbol": "nvda", "op": "above", "price": 150}).json()
    aid = r["alert"]["id"]
    assert r["alert"]["symbol"] == "NVDA" and r["priceAlerts"]
    bad = client.post("/api/alerts/price", json={"symbol": "X", "op": "sideways", "price": 1})
    assert bad.status_code == 400
    gone = client.post("/api/alerts/price/delete", json={"id": aid}).json()
    assert gone["priceAlerts"] == []


def test_risk_limit_endpoint():
    client = _demo()
    assert client.post("/api/risk/limit", json={"lossLimit": 750}).json()["lossLimit"] == 750.0
    assert client.post("/api/risk/limit", json={"lossLimit": None}).json()["lossLimit"] is None


def test_kill_switch_blocks_orders_until_reset():
    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER), demo=False)
    state.broker = OrderBroker()
    client = TestClient(create_app(state))

    ok = client.post("/api/orders", json={"symbol": "AAPL", "order_type": "market", "qty": 1})
    assert ok.status_code == 200

    k = client.post("/api/kill").json()
    assert k["halted"] is True and k["flattened"] == {"cancelled": 0, "closed": 0}

    blocked = client.post("/api/orders", json={"symbol": "AAPL", "order_type": "market", "qty": 1})
    assert blocked.status_code == 423                       # halted

    client.post("/api/kill/reset")
    again = client.post("/api/orders", json={"symbol": "AAPL", "order_type": "market", "qty": 1})
    assert again.status_code == 200


def test_index_has_alerts_ui():
    client = _demo()
    html = client.get("/").text
    assert "openAlerts" in html and "/api/alerts" in html
    assert "KILL SWITCH" in html and "pollAlerts" in html
