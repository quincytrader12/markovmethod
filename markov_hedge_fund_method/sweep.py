"""Full-market sweep — score every tradable symbol, not a shortlist.

The curated groups are a good hunting ground, but they are still a list someone
chose, and a list someone chose is a list of things they already thought of. The
point of a scanner is to find what you were not looking for, which means the
universe has to be everything the broker will actually let you trade.

Three problems stand between that idea and a working sweep, and this module
exists to solve them:

  * **Requests.** Eleven thousand symbols fetched one at a time is eleven
    thousand round-trips. Alpaca's bars endpoint takes a list, so the sweep
    works in chunks and the same universe costs a few dozen requests.

  * **Memory.** Ten years of daily bars for eleven thousand symbols is most of a
    gigabyte, so the sweep cannot simply warm the shared cache and walk away. It
    scores a chunk, keeps the scores, and evicts the bars before moving on. Only
    the leaderboard survives, and a leaderboard is kilobytes.

  * **Patience.** A full pass takes minutes, so it runs as a background cycle
    that resumes where it left off rather than something you wait on. Partial
    results are still results: the leaderboard is live from the first chunk, and
    every chunk improves it.

The sweep never competes with the screen in front of you. It runs few threads,
pauses between chunks, and yields whenever the terminal is busy.
"""

from __future__ import annotations

import threading
import time

CHUNK = 240              # symbols per fetch cycle — sized for the batch endpoint
SLICE = 8                # symbols scored between two checks on the user
PAUSE = 2.0              # seconds between chunks, so the API and UI both breathe
KEEP = 300               # leaderboard size — plenty for any filter downstream
WORKERS = 2              # scoring threads; deliberately modest

# How long the user must be quiet before the sweep will do any work, and how
# long it waits before asking again. Measured, not guessed: scoring on four
# threads with no yielding took a page load from 3.7ms to 172ms — a 46x
# regression, and exactly the stutter this whole design exists to avoid.
IDLE_BEFORE_WORK = 1.5
IDLE_POLL = 0.25

# Names that are tradable but are not companies. Alpaca's asset list carries
# warrants, units, rights and preferred shares, which have no meaningful regime
# and would only dilute the leaderboard.
_SKIP_WORDS = ("warrant", "unit", " right", "rights", "preferred", "depositary",
               "when issued", "when-issued", "notes due", "%")


def is_common_stock(symbol: str, name: str) -> bool:
    """Filter Alpaca's asset list down to things worth ranking."""
    sym = (symbol or "").upper()
    if not sym or len(sym) > 5:
        return False
    if not sym.isalpha():            # excludes BRK.B-style and odd suffixes
        return False
    low = (name or "").lower()
    return not any(w in low for w in _SKIP_WORDS)


class MarketSweep:
    """Rolling scan of the entire tradable universe."""

    def __init__(self, state):
        self.state = state
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.board: dict[str, dict] = {}      # symbol -> scored row
        self.cursor = 0
        self.universe: list[str] = []
        self.cycle = 0
        self.scanned = 0
        self.last_chunk: float | None = None
        self.started: float | None = None
        self.last_error: str | None = None

    # ── universe ────────────────────────────────────────────────────────────
    def build_universe(self) -> list[str]:
        """Every tradable common stock the connected account can reach.

        Falls back to the curated groups when there is no connection, so the
        sweep still does something useful offline instead of nothing.
        """
        from .web import SCAN_ALL

        syms = self.state.alpaca_symbols()
        if not syms:
            return list(SCAN_ALL)
        names = getattr(self.state, "_alpaca_names", {}) or {}
        out = [s for s in sorted(syms) if is_common_stock(s, names.get(s, ""))]
        return out or list(SCAN_ALL)

    # ── deference ───────────────────────────────────────────────────────────
    def yield_to_user(self, timeout: float = 30.0) -> None:
        """Block while the user is actively using the terminal.

        The sweep is the lowest-priority thing in the process. It only works in
        the gaps, and a gap is any moment the user has not asked for something
        in the last second and a half. The timeout stops a busy session from
        starving the sweep forever — after half a minute of waiting it takes one
        small slice anyway, which is short enough not to be felt.
        """
        idle_fn = getattr(self.state, "idle_for", None)
        if idle_fn is None:
            return
        waited = 0.0
        while idle_fn() < IDLE_BEFORE_WORK and waited < timeout:
            if self._stop.is_set():
                return
            time.sleep(IDLE_POLL)
            waited += IDLE_POLL

    # ── one chunk ───────────────────────────────────────────────────────────
    def run_chunk(self) -> dict:
        """Fetch, score and then forget one chunk. Returns what it did."""
        from .scanner import score_symbols

        if not self.universe:
            self.universe = self.build_universe()
            self.cursor = 0
        if not self.universe:
            return {"scored": 0, "cursor": 0, "size": 0}

        batch = self.universe[self.cursor: self.cursor + CHUNK]
        if not batch:
            self.cursor = 0
            self.cycle += 1
            return {"scored": 0, "cursor": 0, "size": len(self.universe), "wrapped": True}

        self.yield_to_user()
        self.state.prefetch_ohlc(batch)

        # Score in small slices, standing aside between each. Scoring the whole
        # chunk in one go is what made the terminal stutter: the sweep has all
        # day, the person using it does not.
        rows = []
        for i in range(0, len(batch), SLICE):
            if self._stop.is_set():
                break
            self.yield_to_user()
            rows.extend(score_symbols(self.state, batch[i: i + SLICE], workers=WORKERS) or [])

        with self._lock:
            for r in rows:
                sym = r.get("symbol") or r.get("ticker")
                if sym:
                    self.board[sym] = r
            self._trim()
            self.scanned += len(batch)
        self._evict(batch)

        self.cursor += len(batch)
        if self.cursor >= len(self.universe):
            self.cursor = 0
            self.cycle += 1
            self.universe = []            # re-enumerate next pass; listings change
        self.last_chunk = time.time()
        return {"scored": len(rows), "cursor": self.cursor, "size": len(self.universe)}

    def _trim(self) -> None:
        """Keep only the best rows. The board is the product; the rest is noise."""
        if len(self.board) <= KEEP:
            return
        best = sorted(self.board.values(), key=lambda r: -(r.get("score") or 0))[:KEEP]
        self.board = {(r.get("symbol") or r.get("ticker")): r for r in best}

    def _evict(self, symbols: list[str]) -> None:
        """Drop a chunk's price history once it has been scored.

        Without this the sweep would hold ten years of bars for the entire
        market at once, which is most of a gigabyte. The scores are what we
        wanted; the bars were only ever a means to them. Anything the user
        actually opens is re-fetched on demand and cached normally.
        """
        keep = {str(s).upper() for s in getattr(self.state, "watchlist_hint", []) or []}
        keep.add(str(self.state.settings.ticker).upper())
        for sym in symbols:
            if sym in keep:
                continue
            self.state._ohlc_cache.pop(sym, None)
            self.state._state_cache.pop(sym, None)
            self.state._state_vol_cache.pop(sym, None)

    # ── results ─────────────────────────────────────────────────────────────
    def results(self, top: int = 50) -> list[dict]:
        with self._lock:
            rows = sorted(self.board.values(), key=lambda r: -(r.get("score") or 0))
        return rows[:top]

    def status(self) -> dict:
        size = len(self.universe) or len(self.board)
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "universeSize": len(self.universe),
            "cursor": self.cursor,
            "progress": round(self.cursor / size, 4) if size else 0.0,
            "cycle": self.cycle,
            "scanned": self.scanned,
            "boardSize": len(self.board),
            "lastChunk": self.last_chunk,
            "started": self.started,
            "lastError": self.last_error,
            "chunk": CHUNK,
        }

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.started = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        self._stop.wait(20)                 # let the dashboard load first
        while not self._stop.is_set():
            try:
                self.run_chunk()
                self.last_error = None
            except Exception as exc:  # noqa: BLE001 — a bad chunk must not end the sweep
                self.last_error = str(exc)
            self._stop.wait(PAUSE)
