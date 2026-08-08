"""Opportunity scanner — scoring, ranking, rationale and the /api/scan route."""

from __future__ import annotations

from fastapi.testclient import TestClient

from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.scanner import score_payload
from markov_hedge_fund_method.web import AppState, create_app


def _demo_client():
    state = AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)
    return TestClient(create_app(state)), state


def test_score_payload_shape_and_bounds():
    _, state = _demo_client()
    p = state.state_payload("AAPL")
    s = score_payload(p)
    assert s["symbol"] == "AAPL"
    assert 0 <= s["score"] <= 100
    assert s["verdict"] in ("Strong Buy", "Buy", "Watch", "Avoid")
    assert s["regime"] in ("bull", "bear", "sideways")
    assert set(s["factors"]) == {"regime", "forecast", "trend", "momentum", "modelEdge"}
    assert len(s["rationale"]) > 40 and s["name"] in s["rationale"] or s["symbol"] in s["rationale"]


def test_scan_endpoint_ranks_and_limits():
    client, _ = _demo_client()
    d = client.get("/api/scan", params={"top": 6}).json()
    assert d["universe"] == "market"
    assert d["scanned"] >= 6
    results = d["results"]
    assert 1 <= len(results) <= 6
    # sorted by score, descending
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # every card carries a verdict + a why
    assert all(r["verdict"] in ("Strong Buy", "Buy", "Watch", "Avoid") for r in results)
    assert all(r["rationale"] for r in results)
    assert "not investment advice" in d["disclaimer"].lower()


def test_scan_custom_symbols():
    client, _ = _demo_client()
    d = client.get("/api/scan", params={"symbols": "AAPL,MSFT,NVDA", "top": 10}).json()
    assert d["universe"] == "custom"
    syms = {r["symbol"] for r in d["results"]}
    assert syms <= {"AAPL", "MSFT", "NVDA"} and syms
    assert d["scanned"] == 3


def test_scan_watchlist_universe():
    client, _ = _demo_client()
    d = client.get("/api/scan", params={"universe": "watchlist"}).json()
    assert d["universe"] == "watchlist"
    assert d["results"]


def test_index_has_scanner_ui():
    client, _ = _demo_client()
    html = client.get("/").text
    assert "Opportunity Scanner" in html
    assert "openScanner" in html and "/api/scan" in html
