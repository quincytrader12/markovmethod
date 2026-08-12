"""Watchlist add: honest regimes + a single shared download."""
from __future__ import annotations

import markov_hedge_fund_method.web as web
from fastapi.testclient import TestClient
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.web import AppState, create_app


def _live_state(monkeypatch, fail=False, counter=None):
    """A non-demo AppState whose price fetch we control."""
    import pandas as pd, numpy as np
    def fake_get_ohlc(settings):
        if counter is not None:
            counter.append(settings.ticker)
        if fail:
            raise RuntimeError("network down")
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=800)
        c = pd.Series(np.linspace(100, 200, 800), index=idx)
        return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c})
    monkeypatch.setattr(web, "get_ohlc", fake_get_ohlc)
    return AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=False)


def test_failed_fetch_reports_no_data_not_a_fake_regime(monkeypatch):
    state = _live_state(monkeypatch, fail=True)
    client = TestClient(create_app(state))
    q = client.get("/api/quotes", params={"symbols": "FAKEX"}).json()["quotes"][0]
    assert q["real"] is False
    assert q["regime"] == "unknown"          # NOT bull/bear/sideways from noise
    assert q["dataSource"].startswith("synthetic")


def test_successful_fetch_reports_real_regime(monkeypatch):
    state = _live_state(monkeypatch, fail=False)
    client = TestClient(create_app(state))
    q = client.get("/api/quotes", params={"symbols": "AAPL"}).json()["quotes"][0]
    assert q["real"] is True and q["dataSource"] == "live"
    assert q["regime"] in ("bull", "bear", "sideways")


def test_quote_and_chart_share_one_download(monkeypatch):
    calls = []
    state = _live_state(monkeypatch, fail=False, counter=calls)
    client = TestClient(create_app(state))
    client.get("/api/quotes", params={"symbols": "NVDA"})   # watchlist row
    client.get("/api/state", params={"symbol": "NVDA"})     # chart
    assert calls.count("NVDA") == 1, f"downloaded {calls.count('NVDA')}x, expected 1"


def test_scanner_skips_symbols_with_no_real_data(monkeypatch):
    from markov_hedge_fund_method.scanner import score_symbols
    state = _live_state(monkeypatch, fail=True)
    assert score_symbols(state, ["AAA", "BBB"], workers=2) == []
