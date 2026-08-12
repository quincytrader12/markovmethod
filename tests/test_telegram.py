"""Telegram reports — config, formatting, endpoints (no network)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import markov_hedge_fund_method.telegram as tg
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.telegram import (TelegramError, TelegramNotifier,
                                               format_flip, format_scan)
from markov_hedge_fund_method.web import AppState, create_app


@pytest.fixture
def fake_api(monkeypatch):
    """Record Bot API calls instead of hitting the network."""
    calls = []

    def _call(token, method, params=None):
        calls.append((token, method, params or {}))
        if token == "bad":
            raise TelegramError("Unauthorized")
        if method == "getMe":
            return {"username": "mamba_bot"}
        if method == "getUpdates":
            return [{"message": {"chat": {"id": 424242}}}]
        return {}

    monkeypatch.setattr(tg, "_call", _call)
    return calls


def test_connect_discovers_chat_id_and_persists(tmp_path, fake_api):
    n = TelegramNotifier(config_dir=str(tmp_path))
    assert n.enabled is False
    status = n.connect("tok123")
    assert status["configured"] is True and status["chatId"] == "424242"
    assert n.enabled is True
    # the token is never handed back to the UI
    assert "token" not in status


def test_connect_rejects_bad_token(tmp_path, fake_api):
    n = TelegramNotifier(config_dir=str(tmp_path))
    with pytest.raises(TelegramError):
        n.connect("bad")
    assert n.enabled is False


def test_send_requires_connection(tmp_path, fake_api):
    n = TelegramNotifier(config_dir=str(tmp_path))
    with pytest.raises(TelegramError):
        n.send("hi")
    n.connect("tok123")
    assert n.send("hello") is True
    assert fake_api[-1][1] == "sendMessage"
    assert fake_api[-1][2]["chat_id"] == "424242"


def test_format_scan_filters_by_score():
    rows = [{"symbol": "AAPL", "name": "Apple", "score": 82, "verdict": "Strong Buy",
             "lastPrice": 190, "rationale": "why", "daysInRegime": 2},
            {"symbol": "F", "name": "Ford", "score": 40, "verdict": "Avoid",
             "lastPrice": 10, "rationale": "no", "daysInRegime": 30}]
    text = format_scan(rows, min_score=70)
    assert "AAPL" in text and "Ford" not in text
    assert "🌱 2d new" in text and "not investment advice" in text
    assert format_scan(rows, min_score=95) is None      # nothing qualifies


def test_format_flip_reads_clearly():
    assert "📈" in format_flip({"symbol": "SPY", "from": "bear", "to": "bull"})
    assert "BEAR → " in format_flip({"symbol": "SPY", "from": "bear", "to": "bull"})


def test_endpoints(tmp_path, fake_api):
    state = AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)
    state.telegram = TelegramNotifier(config_dir=str(tmp_path))
    client = TestClient(create_app(state))

    assert client.get("/api/telegram").json()["configured"] is False
    ok = client.post("/api/telegram/connect", json={"token": "tok123"}).json()
    assert ok["configured"] is True

    bad = client.post("/api/telegram/connect", json={"token": "bad"})
    assert bad.status_code == 400

    s = client.post("/api/telegram/settings", json={"sendScans": False, "minScore": 85}).json()
    assert s["sendScans"] is False and s["minScore"] == 85

    assert client.post("/api/telegram/test").json()["ok"] is True
    assert client.post("/api/telegram/disconnect").json()["configured"] is False


def test_send_scan_endpoint_pushes_picks(tmp_path, fake_api):
    state = AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)
    state.telegram = TelegramNotifier(config_dir=str(tmp_path))
    client = TestClient(create_app(state))
    client.post("/api/telegram/connect", json={"token": "tok123"})
    client.post("/api/telegram/settings", json={"minScore": 0})
    r = client.post("/api/telegram/send_scan?universe=megacap&top=3").json()
    assert r["ok"] is True
    assert any(c[1] == "sendMessage" for c in fake_api)
