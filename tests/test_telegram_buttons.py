"""Tappable "add to watchlist" buttons on Telegram alerts.

The alert arrives on a phone; the watchlist used to live in a browser tab. These
tests cover the store that closed that gap and the round trip that makes a
button real: send with a keyboard, poll the tap, act, acknowledge, redraw.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.telegram import (
    ADD_PREFIX,
    TelegramNotifier,
    parse_callback,
    watch_buttons,
)
from markov_hedge_fund_method.watchlist import DEFAULTS, MAX_SYMBOLS, WatchlistStore
from markov_hedge_fund_method.web import AppState, create_app


# ── the store the button writes into ────────────────────────────────────────
@pytest.fixture
def store(tmp_path):
    return WatchlistStore(config_dir=str(tmp_path))


def test_a_fresh_install_gets_the_defaults_not_a_blank_screen(store):
    assert store.list() == DEFAULTS


def test_adding_puts_the_newest_first(store):
    store.add("ROKU")
    assert store.list()[0] == "ROKU", "the name you just added is the one you want to see"


def test_adding_is_case_insensitive(store):
    store.add("roku")
    assert "ROKU" in store.list()


def test_adding_twice_is_not_an_error(store):
    """The same name can arrive in two alerts, and a button can be tapped twice."""
    store.add("ROKU")
    second = store.add("ROKU")
    assert second["ok"] is True and second["added"] is False
    assert store.list().count("ROKU") == 1


def test_an_empty_symbol_is_refused(store):
    assert store.add("  ")["ok"] is False


def test_removing_works_and_is_idempotent(store):
    store.add("ROKU")
    assert store.remove("ROKU")["removed"] is True
    assert store.remove("ROKU")["removed"] is False


def test_the_list_is_capped(store):
    store.replace([f"S{i}" for i in range(MAX_SYMBOLS)])
    full = store.add("EXTRA")
    assert full["ok"] is False and "full" in full["reason"]


def test_replace_deduplicates_and_uppercases(store):
    out = store.replace(["aapl", "AAPL", "msft", ""])
    assert out == ["AAPL", "MSFT"]


def test_it_survives_a_restart(tmp_path):
    WatchlistStore(config_dir=str(tmp_path)).add("ROKU")
    assert "ROKU" in WatchlistStore(config_dir=str(tmp_path)).list()


def test_a_corrupt_file_falls_back_to_defaults(tmp_path):
    s = WatchlistStore(config_dir=str(tmp_path))
    s.add("ROKU")
    with open(s.path, "w", encoding="utf-8") as f:
        f.write("{ not json")
    assert WatchlistStore(config_dir=str(tmp_path)).list() == DEFAULTS


# ── the keyboard ────────────────────────────────────────────────────────────
def test_every_symbol_gets_its_own_button():
    """One button for the whole batch would mean taking names you did not want."""
    rows = watch_buttons(["AAPL", "MSFT", "NVDA", "GOOGL"], per_row=3)
    flat = [b for row in rows for b in row]
    assert len(flat) == 4
    assert [d for _, d in flat] == [f"{ADD_PREFIX}{s}" for s in ("AAPL", "MSFT", "NVDA", "GOOGL")]


def test_buttons_wrap_into_rows():
    rows = watch_buttons(["A", "B", "C", "D", "E"], per_row=2)
    assert [len(r) for r in rows] == [2, 2, 1]


def test_a_name_already_held_is_shown_as_taken():
    rows = watch_buttons(["AAPL", "MSFT"], added={"MSFT"})
    labels = {d.split(":")[1]: t for row in rows for t, d in row}
    assert labels["MSFT"].startswith("✓")
    assert labels["AAPL"].startswith("➕")


def test_blank_symbols_do_not_produce_buttons():
    assert watch_buttons(["", "  ", "AAPL"]) == [[("➕ AAPL", "add:AAPL")]]


def test_callback_data_round_trips():
    assert parse_callback("add:AAPL") == ("add", "AAPL")
    assert parse_callback("add:aapl") == ("add", "AAPL")


def test_unrecognised_callback_data_is_ignored():
    """A stale button from an old build must not trigger a mystery action."""
    for junk in ("", "nonsense", "delete:AAPL", "add:", "add:   "):
        assert parse_callback(junk) == ("", "")


# ── the wire format ─────────────────────────────────────────────────────────
class _FakeTelegram:
    """Captures what would have been sent, and replays taps."""

    def __init__(self, tmp_path):
        self.calls = []
        self.updates = []
        self.tmp = tmp_path

    def __call__(self, token, method, params=None):
        self.calls.append((method, params or {}))
        if method == "getUpdates":
            out, self.updates = self.updates, []
            return out
        if method == "sendMessage":
            return {"message_id": 42}
        return {}


@pytest.fixture
def tg(tmp_path, monkeypatch):
    import markov_hedge_fund_method.telegram as tmod

    fake = _FakeTelegram(tmp_path)
    monkeypatch.setattr(tmod, "_call", fake)
    n = TelegramNotifier(config_dir=str(tmp_path))
    n.save(token="t", chatId="99")
    return n, fake


def test_send_attaches_an_inline_keyboard(tg):
    notifier, fake = tg
    notifier.send("hello", buttons=watch_buttons(["AAPL", "MSFT"]))
    method, params = fake.calls[-1]
    assert method == "sendMessage"
    markup = json.loads(params["reply_markup"])
    assert markup["inline_keyboard"][0][0]["callback_data"] == "add:AAPL"


def test_send_without_buttons_sends_no_markup(tg):
    notifier, fake = tg
    notifier.send("hello")
    assert "reply_markup" not in fake.calls[-1][1]


def test_polling_returns_taps_and_advances_the_offset(tg):
    notifier, fake = tg
    fake.updates = [{"update_id": 7, "callback_query": {
        "id": "cb1", "data": "add:AAPL",
        "message": {"message_id": 42, "chat": {"id": 99}}}}]
    taps = notifier.poll_callbacks()
    assert taps[0]["data"] == "add:AAPL" and taps[0]["messageId"] == 42
    # The offset is the acknowledgement: without it Telegram redelivers the tap.
    assert notifier.load()["updateOffset"] == 8


def test_the_offset_is_sent_on_the_next_poll(tg):
    notifier, fake = tg
    notifier.save(updateOffset=8)
    notifier.poll_callbacks()
    assert fake.calls[-1][1]["offset"] == 8


def test_answering_a_callback_stops_the_button_spinning(tg):
    notifier, fake = tg
    assert notifier.answer_callback("cb1", "AAPL added") is True
    method, params = fake.calls[-1]
    assert method == "answerCallbackQuery" and params["callback_query_id"] == "cb1"


# ── the round trip ──────────────────────────────────────────────────────────
def _app(tmp_path, monkeypatch):
    import markov_hedge_fund_method.telegram as tmod

    fake = _FakeTelegram(tmp_path)
    monkeypatch.setattr(tmod, "_call", fake)
    state = AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)
    state.telegram = TelegramNotifier(config_dir=str(tmp_path))
    state.telegram.save(token="t", chatId="99")
    state.watchlist = WatchlistStore(config_dir=str(tmp_path))
    return state, fake


def test_a_tap_adds_the_symbol_to_the_watchlist(tmp_path, monkeypatch):
    state, fake = _app(tmp_path, monkeypatch)
    state.watcher._buttons_for[42] = ["ROKU", "PLTR"]
    fake.updates = [{"update_id": 1, "callback_query": {
        "id": "cb1", "data": "add:ROKU",
        "message": {"message_id": 42, "chat": {"id": 99}}}}]

    handled = state.watcher.handle_taps()
    assert handled == [{"symbol": "ROKU", "added": True}]
    assert "ROKU" in state.watchlist.list()


def test_a_tap_is_acknowledged_and_the_keyboard_redrawn(tmp_path, monkeypatch):
    state, fake = _app(tmp_path, monkeypatch)
    state.watcher._buttons_for[42] = ["ROKU", "PLTR"]
    fake.updates = [{"update_id": 1, "callback_query": {
        "id": "cb1", "data": "add:ROKU",
        "message": {"message_id": 42, "chat": {"id": 99}}}}]
    state.watcher.handle_taps()

    methods = [m for m, _ in fake.calls]
    assert "answerCallbackQuery" in methods, "the button would spin forever"
    assert "editMessageReplyMarkup" in methods, "the tapped name still offers to add"

    edit = [p for m, p in fake.calls if m == "editMessageReplyMarkup"][-1]
    labels = {b["callback_data"].split(":")[1]: b["text"]
              for row in json.loads(edit["reply_markup"])["inline_keyboard"] for b in row}
    assert labels["ROKU"].startswith("✓") and labels["PLTR"].startswith("➕")


def test_tapping_a_name_already_held_says_so_rather_than_failing(tmp_path, monkeypatch):
    state, fake = _app(tmp_path, monkeypatch)
    state.watchlist.add("ROKU")
    state.watcher._buttons_for[42] = ["ROKU"]
    fake.updates = [{"update_id": 1, "callback_query": {
        "id": "cb1", "data": "add:ROKU",
        "message": {"message_id": 42, "chat": {"id": 99}}}}]
    handled = state.watcher.handle_taps()
    assert handled == [{"symbol": "ROKU", "added": False}]
    answer = [p for m, p in fake.calls if m == "answerCallbackQuery"][-1]
    assert "already" in answer["text"].lower()


def test_an_unrecognised_tap_is_ignored(tmp_path, monkeypatch):
    state, fake = _app(tmp_path, monkeypatch)
    before = state.watchlist.list()
    fake.updates = [{"update_id": 1, "callback_query": {
        "id": "cb1", "data": "wipe:everything",
        "message": {"message_id": 42, "chat": {"id": 99}}}}]
    assert state.watcher.handle_taps() == []
    assert state.watchlist.list() == before


def test_polling_is_a_no_op_when_telegram_is_not_connected(tmp_path, monkeypatch):
    state, fake = _app(tmp_path, monkeypatch)
    state.telegram.save(token="", chatId="")
    assert state.watcher.handle_taps() == []


def test_a_poll_failure_does_not_kill_the_watcher(tmp_path, monkeypatch):
    state, fake = _app(tmp_path, monkeypatch)
    state.telegram.poll_callbacks = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    assert state.watcher.handle_taps() == []
    assert "offline" in (state.watcher.last_error or "")


def test_the_alert_digest_carries_a_button_per_name(tmp_path, monkeypatch):
    state, fake = _app(tmp_path, monkeypatch)
    picks = [{"symbol": s, "score": 90, "verdict": "Strong Buy", "dsr": 0.99,
              "regime": "bull", "daysInRegime": 2, "rationale": "x", "lastPrice": 10.0}
             for s in ("AAA", "BBB", "CCC")]
    state.watcher.candidates = lambda cfg: picks
    out = state.watcher.run_once(force=True)

    assert out["sent"] == 3
    send = [p for m, p in fake.calls if m == "sendMessage"][-1]
    markup = json.loads(send["reply_markup"])
    data = [b["callback_data"] for row in markup["inline_keyboard"] for b in row]
    assert data == ["add:AAA", "add:BBB", "add:CCC"]
    assert state.watcher._buttons_for[42] == ["AAA", "BBB", "CCC"]


# ── the browser and the bot share one list ──────────────────────────────────
def _client(tmp_path):
    state = AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)
    state.watchlist = WatchlistStore(config_dir=str(tmp_path))
    return state, TestClient(create_app(state))


def test_the_api_exposes_the_same_list(tmp_path):
    state, client = _client(tmp_path)
    state.watchlist.add("ROKU")
    assert client.get("/api/watchlist").json()["symbols"][0] == "ROKU"


def test_adding_through_the_api_persists(tmp_path):
    state, client = _client(tmp_path)
    client.post("/api/watchlist", json={"symbol": "pltr"})
    assert "PLTR" in WatchlistStore(config_dir=str(tmp_path)).list()


def test_the_api_can_replace_the_whole_list(tmp_path):
    state, client = _client(tmp_path)
    d = client.post("/api/watchlist", json={"symbols": ["aapl", "msft"]}).json()
    assert d["symbols"] == ["AAPL", "MSFT"]


def test_the_api_rejects_a_missing_symbol(tmp_path):
    _, client = _client(tmp_path)
    assert client.post("/api/watchlist", json={}).status_code == 400
    assert client.post("/api/watchlist/remove", json={}).status_code == 400


def test_the_page_syncs_the_stored_list(tmp_path):
    _, client = _client(tmp_path)
    html = client.get("/").text
    assert "syncWatchlist" in html and "/api/watchlist" in html
    assert "added from Telegram" in html, "a name arriving from the phone is not surfaced"
