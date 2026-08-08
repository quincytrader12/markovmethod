"""Trade journal (persistent) + performance-by-regime analytics."""

from __future__ import annotations

from fastapi.testclient import TestClient

from markov_hedge_fund_method.broker import OrderResult
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.journal import JournalStore
from markov_hedge_fund_method.orders import build_order_request
from markov_hedge_fund_method.web import AppState, create_app


def _demo(tmp_path):
    state = AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)
    state.journal = JournalStore(config_dir=str(tmp_path))
    return TestClient(create_app(state)), state


class OrderBroker:
    def submit_ticket(self, ticket):
        build_order_request(ticket)
        return OrderResult(id="x1", status="accepted", summary="ok")


def test_journal_store_crud(tmp_path):
    js = JournalStore(config_dir=str(tmp_path))
    assert js.list() == []
    e = js.add(symbol="aapl", side="buy", qty=10, price=150.0,
               regime="bull", tags=["breakout"], notes="test")
    assert e["symbol"] == "AAPL" and e["regime"] == "bull"
    assert js.list()[0]["id"] == e["id"]

    js.update(e["id"], pnl=120.0, rMultiple=1.5, notes="won")
    row = js.list()[0]
    assert row["pnl"] == 120.0 and row["notes"] == "won" and row["rMultiple"] == 1.5

    a = js.analytics()
    assert a["byRegime"]["bull"]["closed"] == 1 and a["byRegime"]["bull"]["winRate"] == 1.0
    assert a["totals"]["pnl"] == 120.0

    assert js.remove(e["id"]) is True and js.list() == []


def test_journal_analytics_by_regime(tmp_path):
    js = JournalStore(config_dir=str(tmp_path))
    js.add(symbol="A", side="buy", regime="bull", pnl=100.0)
    js.add(symbol="B", side="buy", regime="bull", pnl=-40.0)
    js.add(symbol="C", side="buy", regime="bear", pnl=30.0)
    a = js.analytics()
    assert a["byRegime"]["bull"]["closed"] == 2 and a["byRegime"]["bull"]["winRate"] == 0.5
    assert a["byRegime"]["bear"]["winRate"] == 1.0
    assert a["totals"]["closed"] == 3 and a["totals"]["pnl"] == 90.0


def test_journal_endpoints(tmp_path):
    client, _ = _demo(tmp_path)
    assert client.get("/api/journal").json()["entries"] == []

    r = client.post("/api/journal", json={"symbol": "nvda", "side": "buy", "qty": 5,
                                          "price": 120, "tags": ["momo"], "regime": "bull"})
    eid = r.json()["entry"]["id"]
    assert client.get("/api/journal").json()["entries"][0]["symbol"] == "NVDA"

    up = client.post("/api/journal/update", json={"id": eid, "pnl": 75.0}).json()
    assert up["entry"]["pnl"] == 75.0
    assert client.get("/api/journal").json()["analytics"]["totals"]["pnl"] == 75.0

    assert client.post("/api/journal/delete", json={"id": eid}).json()["ok"] is True
    assert client.get("/api/journal").json()["entries"] == []


def test_journal_add_requires_symbol(tmp_path):
    client, _ = _demo(tmp_path)
    assert client.post("/api/journal", json={}).status_code == 400


def test_regime_performance_endpoint(tmp_path):
    client, _ = _demo(tmp_path)
    d = client.get("/api/regime-performance", params={"symbol": "SPY"}).json()
    assert d["symbol"] == "SPY"
    assert set(d["byRegime"]) == {"bear", "sideways", "bull"}
    for stats in d["byRegime"].values():
        assert "winRate" in stats and "days" in stats and "avgReturn" in stats


def test_order_autologs_journal_with_regime(tmp_path):
    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER), demo=False)
    state.broker = OrderBroker()
    state.journal = JournalStore(config_dir=str(tmp_path))
    client = TestClient(create_app(state))

    ok = client.post("/api/orders", json={"symbol": "AAPL", "order_type": "market",
                                          "side": "buy", "qty": 3})
    assert ok.status_code == 200
    entries = client.get("/api/journal").json()["entries"]
    assert len(entries) == 1
    assert entries[0]["symbol"] == "AAPL" and entries[0]["source"] == "order"
    assert entries[0]["regime"] in ("bull", "bear", "sideways", "")


def test_index_has_journal_ui(tmp_path):
    client, _ = _demo(tmp_path)
    html = client.get("/").text
    assert "Trading Journal" in html and "openJournal" in html
    assert "/api/journal" in html and "performance by regime" in html.lower()
