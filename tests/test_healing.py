"""Self-healing: degraded feeds recover, drifted records reconcile, dead
threads restart — and none of it invents data or places a trade."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.healing import Healer, HealthLog, retry_delay
from markov_hedge_fund_method.journal import JournalStore
from markov_hedge_fund_method.web import AppState, create_app


# ── the retry ladder ─────────────────────────────────────────────────────────
def test_retry_delay_backs_off_then_plateaus():
    delays = [retry_delay(i) for i in range(1, 8)]
    assert delays[0] == 15.0
    assert delays == sorted(delays), "backoff must never shrink"
    assert delays[-1] == delays[-2], "and must plateau rather than grow forever"
    assert retry_delay(0) == 0.0


def test_health_log_is_bounded():
    log = HealthLog(limit=5)
    for i in range(20):
        log.record("data", f"event {i}", healed=False)
    assert len(log.entries) == 5
    assert log.entries[-1]["detail"] == "event 19", "the newest must survive"


def test_health_log_counts_healed_separately():
    log = HealthLog()
    log.record("data", "broke", False)
    log.record("data", "fixed", True)
    assert log.counts() == {"total": 2, "healed": 1, "unhealed": 1}


def test_recent_is_newest_first():
    log = HealthLog()
    log.record("a", "one", True)
    log.record("b", "two", True)
    assert [e["detail"] for e in log.recent()] == ["two", "one"]


# ── data-feed healing ────────────────────────────────────────────────────────
def _state(**kw):
    return AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True, **kw)


def test_a_failed_symbol_is_scheduled_for_retry():
    h = Healer(_state())
    assert h.should_retry("AAPL") is False, "a symbol that never failed is not retried"
    h.note_fetch("AAPL", False, "timeout")
    assert h.should_retry("AAPL") is False, "not yet — the backoff has not elapsed"
    h.data_failures["AAPL"]["nextTry"] = time.monotonic() - 1
    assert h.should_retry("AAPL") is True


def test_recovery_clears_the_failure_and_counts_a_repair():
    h = Healer(_state())
    h.note_fetch("AAPL", False, "timeout")
    assert h.degraded_symbols()[0]["symbol"] == "AAPL"
    h.note_fetch("AAPL", True)
    assert h.degraded_symbols() == []
    assert h.repairs == 1
    assert any(e["healed"] and "recovered" in e["detail"] for e in h.log.entries)


def test_repeated_failures_lengthen_the_wait():
    h = Healer(_state())
    for _ in range(3):
        h.note_fetch("XYZ", False, "down")
    assert h.data_failures["XYZ"]["failures"] == 3
    assert h.degraded_symbols()[0]["retryInSec"] > 60


def test_a_healthy_symbol_is_not_tracked():
    h = Healer(_state())
    h.note_fetch("SPY", True)
    assert h.data_failures == {} and h.repairs == 0


def test_a_failed_fetch_stops_being_cached_forever():
    """The bug this exists to prevent: one bad fetch pinning a symbol to
    fabricated data for the whole cache lifetime."""
    state = _state()
    state.demo = False
    calls = {"n": 0}

    import markov_hedge_fund_method.web as web

    def boom(_settings):
        calls["n"] += 1
        raise RuntimeError("feed down")

    original = web.get_ohlc
    web.get_ohlc = boom
    try:
        _, src = state.ohlc_for("FAIL")
        assert src.startswith("synthetic") and calls["n"] == 1
        state.ohlc_for("FAIL")
        assert calls["n"] == 1, "still inside the backoff — must not hammer the feed"

        state.healer.data_failures["FAIL"]["nextTry"] = time.monotonic() - 1
        state.ohlc_for("FAIL")
        assert calls["n"] == 2, "backoff elapsed — it must try again, not stay broken"
    finally:
        web.get_ohlc = original


def test_recovering_a_symbol_drops_payloads_built_from_fake_prices():
    state = _state()
    state.demo = False
    import markov_hedge_fund_method.web as web
    from markov_hedge_fund_method.market_data import synthetic_ohlc

    original = web.get_ohlc
    web.get_ohlc = lambda _s: (_ for _ in ()).throw(RuntimeError("down"))
    try:
        state.state_payload("FAIL")
        assert "FAIL" in state._state_cache
        state.healer.data_failures["FAIL"]["nextTry"] = time.monotonic() - 1
        web.get_ohlc = lambda _s: synthetic_ohlc(600, seed=1)
        _, src = state.ohlc_for("FAIL")
        assert src == "live"
        assert "FAIL" not in state._state_cache, "stale fake-price payload must be dropped"
    finally:
        web.get_ohlc = original


# ── position reconciliation ──────────────────────────────────────────────────
class _Pos:
    def __init__(self, symbol):
        self.symbol = symbol


class _Broker:
    def __init__(self, symbols):
        self.symbols = list(symbols)

    def list_positions(self):
        return [_Pos(s) for s in self.symbols]


def test_reconcile_closes_journal_entries_the_broker_no_longer_holds(tmp_path):
    state = _state()
    state.journal = JournalStore(config_dir=str(tmp_path))
    state.broker = _Broker(["AAPL"])
    state.journal.add(symbol="AAPL", side="buy", qty=10)     # still open, still held
    state.journal.add(symbol="MSFT", side="buy", qty=5)      # open, but gone

    out = Healer(state).reconcile_positions()
    assert out["closed"] == 1 and out["entries"] == ["MSFT"]
    rows = {e["symbol"]: e for e in state.journal.list()}
    assert rows["AAPL"]["pnl"] is None, "a position still held must be left alone"
    assert rows["MSFT"]["pnl"] == 0.0
    assert "reconciled" in rows["MSFT"]["notes"]


def test_reconcile_never_invents_a_pnl(tmp_path):
    state = _state()
    state.journal = JournalStore(config_dir=str(tmp_path))
    state.broker = _Broker([])
    state.journal.add(symbol="TSLA", side="buy", qty=1)
    Healer(state).reconcile_positions()
    row = state.journal.list()[0]
    assert row["pnl"] == 0.0
    assert "unavailable" in row["notes"], "it must say the number is missing, not guess one"


def test_reconcile_leaves_already_closed_entries_alone(tmp_path):
    state = _state()
    state.journal = JournalStore(config_dir=str(tmp_path))
    state.broker = _Broker([])
    state.journal.add(symbol="NVDA", side="buy", pnl=420.0)
    out = Healer(state).reconcile_positions()
    assert out["closed"] == 0
    assert state.journal.list()[0]["pnl"] == 420.0


def test_reconcile_survives_a_broker_that_raises(tmp_path):
    class Angry:
        def list_positions(self):
            raise RuntimeError("connection reset")

    state = _state()
    state.journal = JournalStore(config_dir=str(tmp_path))
    state.broker = Angry()
    h = Healer(state)
    out = h.reconcile_positions()
    assert out["closed"] == 0 and "error" in out
    assert h.log.entries[-1]["healed"] is False


def test_reconcile_is_a_no_op_without_a_broker():
    assert Healer(_state()).reconcile_positions()["closed"] == 0


# ── threads and connections ──────────────────────────────────────────────────
def test_a_dead_watcher_is_restarted_when_autoscan_is_on():
    state = _state()
    started = {"n": 0}
    state.watcher.status = lambda: {"autoScan": True, "running": False}
    state.watcher.start = lambda: started.__setitem__("n", started["n"] + 1)
    assert Healer(state).restart_watcher_if_dead() is True
    assert started["n"] == 1


def test_a_live_watcher_is_left_alone():
    state = _state()
    state.watcher.status = lambda: {"autoScan": True, "running": True}
    state.watcher.start = lambda: (_ for _ in ()).throw(AssertionError("must not restart"))
    assert Healer(state).restart_watcher_if_dead() is False


def test_watcher_is_not_started_when_autoscan_is_off():
    state = _state()
    state.watcher.status = lambda: {"autoScan": False, "running": False}
    state.watcher.start = lambda: (_ for _ in ()).throw(AssertionError("must not restart"))
    assert Healer(state).restart_watcher_if_dead() is False


def test_demo_mode_never_touches_the_broker():
    assert Healer(_state()).reconnect_broker_if_down() is False


def test_a_working_broker_is_not_reconnected():
    state = _state()
    state.demo = False
    state.settings = Settings(ticker="SPY", mode=Mode.PAPER, api_key="k", api_secret="s")
    state.broker = _Broker([])
    state.reconnect = lambda: (_ for _ in ()).throw(AssertionError("must not reconnect"))
    assert Healer(state).reconnect_broker_if_down() is False


def test_a_dead_broker_is_rebuilt():
    class Dead:
        def list_positions(self):
            raise RuntimeError("socket closed")

    state = _state()
    state.demo = False
    state.settings = Settings(ticker="SPY", mode=Mode.PAPER, api_key="k", api_secret="s")
    state.broker = Dead()
    rebuilt = {"n": 0}

    def fake_reconnect():
        rebuilt["n"] += 1
        state.broker = _Broker([])

    state.reconnect = fake_reconnect
    h = Healer(state)
    assert h.reconnect_broker_if_down() is True
    assert rebuilt["n"] == 1 and h.repairs == 1


# ── the sweep ────────────────────────────────────────────────────────────────
def test_one_broken_repair_cannot_stop_the_others():
    state = _state()
    h = Healer(state)
    h.reconnect_broker_if_down = lambda: (_ for _ in ()).throw(RuntimeError("kaboom"))
    out = h.run()
    assert "error" in out["broker"]
    assert "watcher" in out and "positions" in out, "the sweep must carry on"
    assert h.last_run is not None


def test_status_reports_everything_the_ui_needs():
    h = Healer(_state())
    h.note_fetch("AAPL", False, "timeout")
    s = h.status()
    for key in ("lastRun", "repairs", "running", "intervalSec", "degraded",
                "log", "total", "healed", "unhealed"):
        assert key in s
    assert s["degraded"][0]["symbol"] == "AAPL"


# ── API ──────────────────────────────────────────────────────────────────────
def _client():
    return TestClient(create_app(_state()))


def test_health_endpoint():
    d = _client().get("/api/health").json()
    assert d["repairs"] == 0 and d["degraded"] == []


def test_heal_endpoint_runs_a_sweep():
    c = _client()
    d = c.post("/api/health/heal").json()
    assert d["ok"] is True and d["status"]["lastRun"] is not None


def test_retry_endpoint_requires_a_symbol():
    assert _client().post("/api/health/retry", json={}).status_code == 400


def test_retry_endpoint_refetches():
    d = _client().post("/api/health/retry", json={"symbol": "SPY"}).json()
    assert d["symbol"] == "SPY" and "dataSource" in d
    # Demo mode is synthetic by design, so a retry cannot report live data.
    assert d["ok"] is False and d["dataSource"] == "synthetic (demo)"


def test_index_has_the_health_panel():
    html = _client().get("/").text
    assert "openHealth" in html and "Self-Healing" in html
    assert "/api/health" in html and "retryFeed" in html


# ── the sweep must not drown the log it shares with real faults ─────────────
def test_sweep_failures_are_counted_not_itemised():
    """Reported from live use: the health panel filled with OTC tickers.

    A full-market sweep meets thousands of names with no published price data.
    That is a property of the market, not a fault, and logging each one buries
    the entries that mean something.
    """
    h = Healer(_state())
    for sym in ("AMKBY", "AMIGY", "ALZIF", "ALPMY", "ALIZY"):
        h.note_fetch(sym, False, "no price bars published", quiet=True)
    assert h.log.entries == [], "the sweep is filling the log again"
    assert h.status()["quietFailures"] == 5, "quiet does not mean uncounted"


def test_a_symbol_the_user_cares_about_is_still_logged():
    h = Healer(_state())
    h.note_fetch("AMKBY", False, "no bars", quiet=True)
    h.note_fetch("SPY", False, "connection reset")
    assert [e["detail"] for e in h.log.recent()] == ["SPY fetch failed (connection reset)"]


def test_a_quiet_recovery_is_not_announced():
    h = Healer(_state())
    h.note_fetch("AMKBY", False, "no bars", quiet=True)
    h.note_fetch("AMKBY", True, quiet=True)
    assert h.log.entries == []


def test_the_degraded_list_is_bounded():
    """A panel listing four hundred names tells you less than one listing 25.

    Uses real failures, not quiet ones: quiet failures no longer reach this list
    at all, so the bound has to be proved on the entries that do.
    """
    h = Healer(_state())
    for i in range(400):
        h.note_fetch(f"SYM{i}", False, "connection reset")
    st = h.status()
    assert len(st["degraded"]) <= 25
    assert st["degradedTotal"] == 400, "the true count must still be reported"


def test_a_symbol_with_no_published_data_is_not_called_degraded():
    """Reported from live use: 150+ OTC ADRs filling 'CURRENTLY ON FALLBACK DATA'.

    That panel answers "what that I might be looking at is broken". A name the
    feed has never published is a permanent property of the market, not a fault
    awaiting repair, and three thousand of them made the panel unreadable.
    """
    h = Healer(_state())
    for sym in ("NSRGY", "NTDOY", "DTEGY", "HEINY", "BYDDY"):
        h.note_fetch(sym, False, "no price bars published", quiet=True)
    h.note_fetch("SPY", False, "connection reset")

    st = h.status()
    assert [d["symbol"] for d in st["degraded"]] == ["SPY"]
    assert st["noData"] == 5, "they must still be counted"
    assert st["degradedTotal"] == 1


def test_dead_symbols_are_not_scheduled_for_retry():
    """They were queued for retry forever, which is what produced the churn."""
    h = Healer(_state())
    h.note_fetch("NTDOY", False, "no bars", quiet=True)
    assert h.should_retry("NTDOY") is False
    assert "NTDOY" not in h.data_failures
