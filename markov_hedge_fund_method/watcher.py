"""Background scan watcher — the terminal hunting while you are not looking.

The scanner only scores when someone clicks it, and the regime-flip alerts only
cover symbols already on the watchlist. Neither one can tell you that a name you
have never looked at just became interesting. This closes that gap: a daemon
thread rescans the universe on an interval, keeps only the names that clear your
thresholds, and pushes a short digest to Telegram.

Three things stop it becoming noise:

  * Deduplication. A name alerts once and then not again for `cooldownHours`,
    so a candidate hovering on the edge of the threshold cannot spam you.
  * Quiet hours and weekday gating, so it does not ping at 3am or on a Sunday.
  * The same DSR bar the UI uses, so a name has to survive the multiple-testing
    correction before it is worth waking you up for.

Runs in the server process rather than the browser, so it keeps working with the
page closed — but note it still needs the app itself to be running.
"""

from __future__ import annotations

import threading
import time

# How often to check for a tapped button. Telegram's getUpdates is cheap and a
# button that takes half an hour to respond may as well not be a button.
TAP_POLL_SEC = 10

DEFAULTS = {
    "autoScan": False,          # off until the user turns it on
    "scanIntervalMin": 30,
    "scanUniverse": "full",      # the whole tradable market, via the sweep
    "scanMinDsr": 0.95,         # same bar as "Proven edge only"
    "scanFreshDays": 0,         # 0 = any age; 5 = only fresh flips
    "scanMinScore": 70,
    "quietStart": 22,           # local hour, inclusive
    "quietEnd": 7,              # local hour, exclusive
    "weekdaysOnly": True,
    "cooldownHours": 24,
}


def settings_from(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    for k in DEFAULTS:
        if cfg.get(k) is not None:
            out[k] = cfg[k]
    return out


def in_quiet_hours(now: time.struct_time, start: int, end: int) -> bool:
    """Quiet window, allowing it to wrap past midnight (e.g. 22 -> 7)."""
    h = now.tm_hour
    if start == end:
        return False
    if start < end:
        return start <= h < end
    return h >= start or h < end


def should_send_now(cfg: dict, now: time.struct_time | None = None) -> bool:
    now = now or time.localtime()
    if cfg["weekdaysOnly"] and now.tm_wday >= 5:      # 5,6 = Sat,Sun
        return False
    return not in_quiet_hours(now, int(cfg["quietStart"]), int(cfg["quietEnd"]))


class ScanWatcher:
    def __init__(self, state):
        self.state = state
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._notified: dict[str, float] = {}     # symbol -> epoch last alerted
        self.last_run: float | None = None
        self.last_sent: int = 0
        self.last_error: str | None = None
        # message id -> the symbols its keyboard offers, so a tap can redraw it
        self._buttons_for: dict[int, list] = {}
        self.last_taps: list = []

    # ── config ──────────────────────────────────────────────────────────────
    def config(self) -> dict:
        return settings_from(self.state.telegram.load())

    def status(self) -> dict:
        cfg = self.config()
        nxt = None
        if cfg["autoScan"] and self.last_run:
            nxt = self.last_run + cfg["scanIntervalMin"] * 60
        return {
            **{k: cfg[k] for k in DEFAULTS},
            "running": bool(self._thread and self._thread.is_alive()),
            "lastRun": self.last_run,
            "nextRun": nxt,
            "lastSent": self.last_sent,
            "lastError": self.last_error,
            "trackedSymbols": len(self._notified),
            "telegramReady": self.state.telegram.enabled,
        }

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        # Let the app finish starting before the first sweep.
        self._stop.wait(60)
        while not self._stop.is_set():
            cfg = self.config()
            if cfg["autoScan"] and self.state.telegram.enabled:
                try:
                    self.run_once()
                except Exception as exc:  # noqa: BLE001 — a bad cycle must not kill the loop
                    self.last_error = str(exc)
            # Buttons are checked far more often than scans run: a tap should
            # take effect in seconds, not on the next half-hourly sweep.
            deadline = time.time() + max(60, int(cfg["scanIntervalMin"]) * 60)
            while not self._stop.is_set() and time.time() < deadline:
                try:
                    self.handle_taps()
                except Exception as exc:  # noqa: BLE001
                    self.last_error = str(exc)
                self._stop.wait(TAP_POLL_SEC)

    # ── one sweep ───────────────────────────────────────────────────────────
    def candidates(self, cfg: dict) -> list[dict]:
        """Score the universe and keep what clears the bar."""
        from .scanner import rank
        from .web import SCAN_GROUPS, SCAN_UNIVERSE

        scope = cfg["scanUniverse"]
        if scope == "full":
            # Read the full-market sweep's leaderboard instead of scanning a
            # list. The sweep is already grinding through every tradable name in
            # the background, so the alert path just reads what it has found —
            # which is the whole point of running it continuously.
            scored = self.state.sweep.results(200)
        else:
            syms = SCAN_GROUPS.get(scope, SCAN_UNIVERSE)
            key = scope if scope in SCAN_GROUPS else "market"
            # Few workers on purpose: this runs unattended in the background and
            # must never make the UI feel sluggish while the user is trading.
            scored = self.state.scored_universe(key, syms, workers=3)
        result = rank(scored, top=50, fresh_days=int(cfg["scanFreshDays"]),
                      proven_only=False, sort="score")
        return [r for r in result["results"]
                if (r.get("dsr") or 0.0) >= float(cfg["scanMinDsr"])
                and (r.get("score") or 0) >= int(cfg["scanMinScore"])]

    # ── button taps coming back from Telegram ───────────────────────────────
    def handle_taps(self) -> list[dict]:
        """Act on buttons tapped in the chat, then tell the phone what happened.

        Three steps per tap and all three matter: do the thing, acknowledge the
        callback so the button stops spinning, and redraw the keyboard so the
        name shows as taken. Skipping the acknowledgement leaves a button that
        looks broken even when the work succeeded.
        """
        from .telegram import parse_callback, watch_buttons

        tg = self.state.telegram
        if not tg.enabled:
            return []
        try:
            taps = tg.poll_callbacks()
        except Exception as exc:  # noqa: BLE001 — a poll failure is not fatal
            self.last_error = str(exc)
            return []

        handled = []
        for tap in taps:
            action, symbol = parse_callback(tap.get("data", ""))
            if action != "add":
                continue
            result = self.state.watchlist.add(symbol)
            note = (f"{symbol} added to your watchlist" if result.get("added")
                    else result.get("reason", "already on the watchlist"))
            tg.answer_callback(tap.get("id"), note)
            # Redraw this message's buttons so the tapped name reads as taken.
            syms = self._buttons_for.get(tap.get("messageId"))
            if syms:
                tg.edit_buttons(tap.get("chatId"), tap.get("messageId"),
                                watch_buttons(syms, added=set(self.state.watchlist.list())))
            handled.append({"symbol": symbol, "added": bool(result.get("added"))})
        if handled:
            self.last_taps = handled
        return handled

    def _is_new(self, symbol: str, cooldown_hours: float, now: float) -> bool:
        last = self._notified.get(symbol)
        return last is None or (now - last) >= cooldown_hours * 3600

    def run_once(self, *, force: bool = False, send: bool = True) -> dict:
        """Scan, filter, deduplicate, notify. Returns what it found and sent."""
        from .telegram import format_scan

        cfg = self.config()
        now = time.time()
        self.last_run = now
        self.last_error = None

        picks = self.candidates(cfg)
        fresh = [p for p in picks if self._is_new(p["symbol"], cfg["cooldownHours"], now)]

        quiet = not should_send_now(cfg) and not force
        sent = 0
        if fresh and send and not quiet and self.state.telegram.enabled:
            from .telegram import watch_buttons

            limit = 6
            text = format_scan(fresh, min_score=int(cfg["scanMinScore"]), limit=limit)
            if text:
                header = (f"🔎 <b>New prospects</b> — {len(fresh)} name"
                          f"{'' if len(fresh) == 1 else 's'} cleared your filters\n\n")
                # One button per name in the digest, so each can be taken or
                # left on its own. Names already held show ticked rather than
                # offering to add something you have.
                shown = [p["symbol"] for p in fresh[:limit]]
                held = set(self.state.watchlist.list())
                msg = self.state.telegram.send(
                    header + text, buttons=watch_buttons(shown, added=held))
                mid = (msg or {}).get("message_id")
                if mid is not None:
                    self._buttons_for[mid] = shown
                    # Bounded: only recent messages can still be tapped usefully.
                    if len(self._buttons_for) > 50:
                        for k in sorted(self._buttons_for)[:-50]:
                            self._buttons_for.pop(k, None)
                sent = len(fresh)
                for p in fresh:
                    self._notified[p["symbol"]] = now
        self.last_sent = sent
        return {
            "scanned": len(picks),
            "new": [p["symbol"] for p in fresh],
            "sent": sent,
            "quiet": quiet,
            "at": time.strftime("%Y-%m-%d %H:%M"),
        }
