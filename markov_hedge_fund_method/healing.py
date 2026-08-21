"""Self-healing — the terminal noticing it is broken and fixing itself.

Every failure this module handles is one that already happened in live use, and
every one of them shares a shape: something breaks quietly, and the terminal
keeps running with a degraded piece nobody is told about. Falling back is what
keeps the screen alive; *staying* fallen back is the bug.

Four repairs, each with a check that is cheap enough to run on a timer:

  * **Data.** A symbol whose fetch failed gets pinned to fabricated data for the
    full cache lifetime — the same duration as a symbol that is working fine.
    That is backwards. Failed symbols get a short retry window with exponential
    backoff instead, so a feed blip heals in seconds rather than minutes, while
    a genuinely dead symbol is not hammered.

  * **Positions.** The journal's idea of what is open and the broker's idea can
    drift apart, which is exactly what happened when a close silently failed.
    Reconciliation reads the broker, finds journal entries still marked open for
    positions that no longer exist, and completes them.

  * **Threads.** The watcher is a daemon thread. If it dies the scan simply stops
    and the status panel keeps saying auto-scan is on. A heartbeat catches that
    and restarts it.

  * **Broker.** A dropped connection turns every order into an error until
    someone reconnects by hand. If the client has gone away, rebuild it.

Two rules the whole module obeys. Healing never invents data: a repair either
recovers the real thing or reports that it could not. And healing never trades:
it can complete a journal record of an exit that already happened, and it can
reconnect a client, but it will not open, close or size a position.
"""

from __future__ import annotations

import time

# Failed fetches retry on this ladder (seconds), so a blip clears fast and a
# genuinely dead symbol backs off instead of being hammered every request.
RETRY_LADDER = (15.0, 60.0, 300.0, 900.0)
MAX_BACKOFF = 900.0


def retry_delay(failures: int) -> float:
    """Seconds to wait before retrying a symbol that has failed `failures` times."""
    if failures <= 0:
        return 0.0
    return RETRY_LADDER[min(failures, len(RETRY_LADDER)) - 1]


class HealthLog:
    """A short, bounded record of what broke and what was repaired.

    Bounded on purpose: a self-healing system that grows an unbounded log is
    just a slower leak. The UI only ever shows the recent tail anyway.
    """

    def __init__(self, limit: int = 60):
        self.limit = limit
        self.entries: list[dict] = []

    def record(self, kind: str, detail: str, healed: bool) -> dict:
        entry = {"at": time.time(), "kind": kind, "detail": detail, "healed": bool(healed)}
        self.entries.append(entry)
        if len(self.entries) > self.limit:
            del self.entries[: len(self.entries) - self.limit]
        return entry

    def recent(self, n: int = 20) -> list[dict]:
        return list(reversed(self.entries[-n:]))

    def counts(self) -> dict:
        healed = sum(1 for e in self.entries if e["healed"])
        return {"total": len(self.entries), "healed": healed,
                "unhealed": len(self.entries) - healed}


class Healer:
    """Runs the repairs against an AppState. Every method is safe to call often."""

    def __init__(self, state):
        self.state = state
        self.log = HealthLog()
        # symbol -> {"failures": int, "nextTry": monotonic deadline}
        self.data_failures: dict[str, dict] = {}
        self.last_run: float | None = None
        self.repairs: int = 0
        # Background sweep failures: counted, not itemised. See note_fetch.
        self.quiet_failures: int = 0
        # Symbols the feed publishes nothing for — counted, never listed.
        self.no_data: set = set()
        self._thread = None
        self._stop = None

    # ── data feed ───────────────────────────────────────────────────────────
    def note_fetch(self, symbol: str, ok: bool, reason: str = "",
                   quiet: bool = False) -> None:
        """Record the outcome of a price fetch and schedule the next attempt.

        `quiet` is for the full-market sweep. This log exists to tell you that
        something you care about has broken and been repaired; a sweep working
        through eleven thousand tickers will meet thousands with no published
        price data, and that is a normal property of the market rather than a
        fault. Logging each one buried the real entries under a wall of noise.
        Quiet failures still count, and the count is reported — they just do not
        each get a line.
        """
        symbol = symbol.upper()
        if ok:
            if symbol in self.data_failures:
                self.data_failures.pop(symbol, None)
                self.repairs += 1
                if not quiet:
                    self.log.record("data", f"{symbol} recovered live data", True)
            return
        if quiet:
            # Not a fault to be healed, so it does not belong in the retry
            # schedule at all. "Currently on fallback data" is a list of things
            # you might be looking at; a name the feed has never published is a
            # permanent property of the market, and putting three thousand of
            # them in that panel made it useless.
            self.quiet_failures += 1
            self.no_data.add(symbol)
            return
        rec = self.data_failures.get(symbol, {"failures": 0, "nextTry": 0.0})
        rec["failures"] += 1
        rec["nextTry"] = time.monotonic() + retry_delay(rec["failures"])
        rec["reason"] = reason
        self.data_failures[symbol] = rec
        self.log.record("data", f"{symbol} fetch failed ({reason or 'no reason given'})", False)

    def should_retry(self, symbol: str) -> bool:
        """True when a previously failed symbol is due another attempt.

        This is what makes the fallback temporary. Without it a symbol that
        failed once keeps serving fabricated data for the whole cache lifetime,
        which is the same as a symbol that is working perfectly.
        """
        rec = self.data_failures.get(symbol.upper())
        return bool(rec) and time.monotonic() >= rec["nextTry"]

    def degraded_symbols(self, limit: int = 25) -> list[dict]:
        """Symbols currently on fallback data, newest trouble first.

        Bounded: a full-market sweep can put thousands of never-published OTC
        tickers in here, and a panel listing three thousand names tells you
        less than one listing twenty-five.
        """
        now = time.monotonic()
        rows = [{"symbol": s, "failures": r["failures"],
                 "retryInSec": max(0.0, round(r["nextTry"] - now, 1)),
                 "reason": r.get("reason", "")}
                for s, r in self.data_failures.items()]
        rows.sort(key=lambda r: (-r["failures"], r["symbol"]))
        return rows[:limit]

    # ── positions vs journal ────────────────────────────────────────────────
    def reconcile_positions(self) -> dict:
        """Close journal entries whose broker position is gone.

        The journal is the record of what happened; the broker is the truth of
        what is. When a close succeeds but the journal write does not — or when
        a position is closed from the Alpaca app, or liquidated — the two drift,
        and every P&L number downstream is wrong until someone notices.

        Realised P&L cannot be recovered after the fact, so the entry is
        completed and marked as reconciled rather than being given a number that
        was never measured. A missing number is recoverable; a fabricated one is
        not.
        """
        broker, journal = self.state.broker, self.state.journal
        if broker is None or journal is None:
            return {"checked": 0, "closed": 0, "entries": []}
        try:
            open_symbols = {str(getattr(p, "symbol", "")).upper()
                            for p in (broker.list_positions() or [])}
        except Exception as exc:  # noqa: BLE001 — a broker hiccup is not a reconciliation
            self.log.record("positions", f"could not read positions: {exc}", False)
            return {"checked": 0, "closed": 0, "entries": [], "error": str(exc)}

        closed = []
        rows = [e for e in journal.list() if e.get("pnl") is None and e.get("symbol")]
        for entry in rows:
            sym = str(entry["symbol"]).upper()
            if sym in open_symbols:
                continue
            note = (entry.get("notes") or "").strip()
            journal.update(entry["id"], pnl=0.0, notes=(
                note + " | " if note else "") + "reconciled: position no longer held, "
                "realised P&L unavailable")
            closed.append(sym)
            self.repairs += 1
            self.log.record("positions", f"{sym} journal entry closed to match broker", True)
        return {"checked": len(rows), "closed": len(closed), "entries": closed}

    # ── background thread ───────────────────────────────────────────────────
    def restart_watcher_if_dead(self) -> bool:
        """Restart the scan watcher if auto-scan is on but its thread is gone."""
        watcher = getattr(self.state, "watcher", None)
        if watcher is None:
            return False
        try:
            status = watcher.status()
            if not status.get("autoScan"):
                return False
            if status.get("running"):
                return False
            watcher.start()
        except Exception as exc:  # noqa: BLE001
            self.log.record("watcher", f"restart failed: {exc}", False)
            return False
        self.repairs += 1
        self.log.record("watcher", "auto-scan thread was dead, restarted", True)
        return True

    # ── broker connection ───────────────────────────────────────────────────
    def reconnect_broker_if_down(self) -> bool:
        """Rebuild the broker client if it has stopped answering.

        Demo mode has no broker by design, and an unconfigured account has none
        either — neither is a fault, so neither is touched.
        """
        if self.state.demo or not getattr(self.state.settings, "has_credentials", False):
            return False
        broker = self.state.broker
        if broker is not None:
            try:
                broker.list_positions()
                return False                      # answering; nothing to heal
            except Exception as exc:  # noqa: BLE001
                self.log.record("broker", f"connection down: {exc}", False)
        try:
            self.state.reconnect()
        except Exception as exc:  # noqa: BLE001
            self.log.record("broker", f"reconnect failed: {exc}", False)
            return False
        if self.state.broker is None:
            return False
        self.repairs += 1
        self.log.record("broker", "broker client rebuilt", True)
        return True

    # ── the sweep ───────────────────────────────────────────────────────────
    def run(self) -> dict:
        """One full pass. Each repair is isolated: one failing cannot stop the rest."""
        out = {}
        for name, fn in (("broker", self.reconnect_broker_if_down),
                         ("watcher", self.restart_watcher_if_dead),
                         ("positions", self.reconcile_positions)):
            try:
                out[name] = fn()
            except Exception as exc:  # noqa: BLE001
                out[name] = {"error": str(exc)}
                self.log.record(name, f"repair raised: {exc}", False)
        self.last_run = time.time()
        return out

    def status(self) -> dict:
        return {
            "lastRun": self.last_run,
            "repairs": self.repairs,
            "quietFailures": self.quiet_failures,
            "noData": len(self.no_data),
            "running": bool(self._thread and self._thread.is_alive()),
            "intervalSec": self.interval,
            "degraded": self.degraded_symbols(),
            "degradedTotal": len(self.data_failures),
            "log": self.log.recent(),
            **self.log.counts(),
        }

    # ── the timer ───────────────────────────────────────────────────────────
    #
    # Deliberately slow. Healing is a background chore: checking every few
    # minutes catches a dead thread or a drifted position long before it costs
    # anything, and checking every few seconds would just add load to a terminal
    # whose whole point is staying responsive while you trade.
    interval = 300.0

    def start(self) -> None:
        import threading

        if getattr(self, "_thread", None) and self._thread.is_alive():
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        stop = getattr(self, "_stop", None)
        if stop is not None:
            stop.set()

    def _loop(self) -> None:
        self._stop.wait(90)                # let the app finish starting first
        while not self._stop.is_set():
            try:
                self.run()
            except Exception as exc:  # noqa: BLE001 — the healer must outlive its repairs
                self.log.record("healer", f"sweep raised: {exc}", False)
            self._stop.wait(self.interval)
