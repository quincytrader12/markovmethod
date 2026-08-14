"""Background scan watcher — filtering, deduplication, quiet hours, endpoints."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import markov_hedge_fund_method.telegram as tg
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.telegram import TelegramNotifier
from markov_hedge_fund_method.watcher import (DEFAULTS, in_quiet_hours, settings_from,
                                              should_send_now)
from markov_hedge_fund_method.web import AppState, create_app


@pytest.fixture
def fake_api(monkeypatch):
    sent = []

    def _call(token, method, params=None):
        if method == "getMe":
            return {"username": "bot"}
        if method == "getUpdates":
            return [{"message": {"chat": {"id": 1}}}]
        sent.append((params or {}).get("text", ""))
        return {}

    monkeypatch.setattr(tg, "_call", _call)
    return sent


def _app(tmp_path):
    state = AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)
    state.telegram = TelegramNotifier(config_dir=str(tmp_path))
    return state, TestClient(create_app(state))


def _t(hour, wday=2):
    """A struct_time on a Wednesday at `hour`."""
    return time.struct_time((2026, 8, 12, hour, 0, 0, wday, 224, 0))


# ── quiet hours ──────────────────────────────────────────────────────────────
def test_quiet_hours_wrap_past_midnight():
    assert in_quiet_hours(_t(23), 22, 7) is True
    assert in_quiet_hours(_t(3), 22, 7) is True
    assert in_quiet_hours(_t(12), 22, 7) is False
    assert in_quiet_hours(_t(6), 22, 7) is True
    assert in_quiet_hours(_t(7), 22, 7) is False       # end is exclusive


def test_quiet_hours_same_day_window():
    assert in_quiet_hours(_t(13), 12, 14) is True
    assert in_quiet_hours(_t(15), 12, 14) is False
    assert in_quiet_hours(_t(5), 0, 0) is False        # disabled


def test_weekend_is_skipped_when_weekdays_only():
    cfg = dict(DEFAULTS)
    assert should_send_now(cfg, _t(12, wday=5)) is False   # Saturday
    assert should_send_now(cfg, _t(12, wday=2)) is True    # Wednesday
    cfg["weekdaysOnly"] = False
    assert should_send_now(cfg, _t(12, wday=6)) is True


def test_settings_fall_back_to_defaults():
    s = settings_from({"scanIntervalMin": 15})
    assert s["scanIntervalMin"] == 15
    assert s["scanMinDsr"] == DEFAULTS["scanMinDsr"]


# ── filtering + dedupe ───────────────────────────────────────────────────────
def test_only_names_clearing_dsr_and_score_qualify(tmp_path, monkeypatch):
    state, _ = _app(tmp_path)
    rows = [
        {"symbol": "GOOD", "dsr": 0.99, "score": 80, "daysInRegime": 2},
        {"symbol": "WEAKEDGE", "dsr": 0.50, "score": 90, "daysInRegime": 2},
        {"symbol": "LOWSCORE", "dsr": 0.99, "score": 40, "daysInRegime": 2},
    ]
    monkeypatch.setattr(state.watcher, "candidates",
                        lambda cfg: [r for r in rows
                                     if r["dsr"] >= cfg["scanMinDsr"]
                                     and r["score"] >= cfg["scanMinScore"]])
    picks = state.watcher.candidates(state.watcher.config())
    assert [p["symbol"] for p in picks] == ["GOOD"]


def test_each_name_alerts_once_per_cooldown(tmp_path, fake_api, monkeypatch):
    state, _ = _app(tmp_path)
    state.telegram.connect("tok")
    fake_api.clear()
    rows = [{"symbol": "AAPL", "name": "Apple", "dsr": 0.99, "score": 80,
             "verdict": "Strong Buy", "lastPrice": 190, "rationale": "why",
             "daysInRegime": 2}]
    monkeypatch.setattr(state.watcher, "candidates", lambda cfg: rows)

    first = state.watcher.run_once(force=True)
    assert first["sent"] == 1 and first["new"] == ["AAPL"]

    second = state.watcher.run_once(force=True)
    assert second["sent"] == 0 and second["new"] == []      # deduplicated

    # once the cooldown lapses it may alert again
    state.watcher._notified["AAPL"] -= DEFAULTS["cooldownHours"] * 3600 + 1
    third = state.watcher.run_once(force=True)
    assert third["sent"] == 1


def test_quiet_hours_block_sending_but_force_overrides(tmp_path, fake_api, monkeypatch):
    state, _ = _app(tmp_path)
    state.telegram.connect("tok")
    rows = [{"symbol": "NVDA", "name": "Nvidia", "dsr": 0.99, "score": 80,
             "verdict": "Buy", "lastPrice": 120, "rationale": "why", "daysInRegime": 1}]
    monkeypatch.setattr(state.watcher, "candidates", lambda cfg: rows)
    monkeypatch.setattr("markov_hedge_fund_method.watcher.should_send_now", lambda cfg, now=None: False)

    quiet = state.watcher.run_once()
    assert quiet["quiet"] is True and quiet["sent"] == 0
    assert state.watcher._notified == {}          # nothing consumed while muted

    forced = state.watcher.run_once(force=True)
    assert forced["sent"] == 1


# ── endpoints ────────────────────────────────────────────────────────────────
def test_watcher_endpoints(tmp_path, fake_api):
    state, client = _app(tmp_path)
    s = client.get("/api/watcher").json()
    assert s["autoScan"] is False and s["running"] is False

    saved = client.post("/api/watcher", json={"autoScan": True, "scanIntervalMin": 2,
                                              "scanUniverse": "midcap"}).json()
    assert saved["autoScan"] is True
    assert saved["scanIntervalMin"] == 5          # clamped to a sane floor
    assert saved["scanUniverse"] == "midcap"

    # run-now requires Telegram
    assert client.post("/api/watcher/run").status_code == 400
    client.post("/api/telegram/connect", json={"token": "tok"})
    assert client.post("/api/watcher/run").status_code == 200


def test_config_survives_restart(tmp_path, fake_api):
    state, client = _app(tmp_path)
    client.post("/api/watcher", json={"autoScan": True, "scanIntervalMin": 45})
    state2, client2 = _app(tmp_path)              # fresh app, same config dir
    s = client2.get("/api/watcher").json()
    assert s["autoScan"] is True and s["scanIntervalMin"] == 45
