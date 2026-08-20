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


def test_wide_universe_and_groups():
    from markov_hedge_fund_method.web import SCAN_GROUPS, SCAN_UNIVERSE
    assert len(SCAN_UNIVERSE) >= 50, "default sweep should be wide, not just mega caps"
    assert len(set(SCAN_UNIVERSE)) == len(SCAN_UNIVERSE), "no duplicate tickers"
    # The eleven standard sectors, plus the two things that are not sectors.
    assert set(SCAN_GROUPS) >= {
        "communication", "discretionary", "staples", "energy", "financials",
        "health", "industrials", "technology", "materials", "realestate",
        "utilities", "etf", "crypto"}
    # Funds are not a sector and must not dilute the equity sweep.
    assert "SPY" not in SCAN_UNIVERSE and "SPY" in SCAN_GROUPS["etf"]


def test_scan_group_universes():
    client, _ = _demo_client()
    for group in ("communication", "realestate", "utilities", "etf", "crypto"):
        d = client.get("/api/scan", params={"universe": group, "top": 5}).json()
        assert d["universe"] == group and d["universeSize"] > 0
        assert d["results"]


def test_fresh_filter_limits_regime_age():
    client, _ = _demo_client()
    d = client.get("/api/scan", params={"universe": "megacap", "fresh": 5, "top": 50}).json()
    assert d["freshDays"] == 5
    assert all(0 < r["daysInRegime"] <= 5 for r in d["results"])
    assert d["matched"] <= d["scanned"]


def test_proven_filter_requires_track_record():
    client, _ = _demo_client()
    d = client.get("/api/scan", params={"universe": "megacap", "proven": "true", "top": 50}).json()
    assert d["provenOnly"] is True
    assert all(r["sharpe"] > 0 and r["winRate"] > 0.5 for r in d["results"])


def test_sort_by_freshness():
    client, _ = _demo_client()
    d = client.get("/api/scan", params={"universe": "megacap", "sort": "fresh", "top": 20}).json()
    ages = [r["daysInRegime"] for r in d["results"]]
    assert ages == sorted(ages) and d["sort"] == "fresh"


def test_watchlist_scope_uses_supplied_symbols():
    client, _ = _demo_client()
    d = client.get("/api/scan", params={"universe": "watchlist",
                                        "watchlist": "AAPL,MSFT", "top": 10}).json()
    assert d["universe"] == "watchlist"
    assert {r["symbol"] for r in d["results"]} <= {"AAPL", "MSFT"}


def test_scan_cache_makes_repeat_scans_instant():
    import time as _t
    client, _ = _demo_client()
    client.get("/api/scan", params={"universe": "megacap", "top": 5})   # warm
    t0 = _t.time()
    client.get("/api/scan", params={"universe": "megacap", "top": 5, "fresh": 3})
    assert _t.time() - t0 < 0.5, "filter change must reuse the cached scores"
