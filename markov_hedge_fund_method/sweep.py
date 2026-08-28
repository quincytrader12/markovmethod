"""Full-market sweep — score every tradable symbol, not a shortlist.

The curated groups are a good hunting ground, but they are still a list someone
chose, and a list someone chose is a list of things they already thought of. The
point of a scanner is to find what you were not looking for, which means the
universe has to be everything the broker will actually let you trade.

Three problems stand between that idea and a working sweep, and this module
exists to solve them:

  * **Requests.** Alpaca's bars endpoint takes a list of symbols, so the sweep
    fetches in chunks rather than one name at a time. Be careful about how big
    a win that is: the endpoint pages at 10,000 bars, so a ten-year request for
    240 symbols is around 61 sequential pages inside a single SDK call, not one
    round-trip. Across the whole market it is 2,772 pages batched against
    11,000 individual requests — a 4x saving, which is worth having and is not
    the two-orders-of-magnitude that "one request instead of hundreds" implies.
    The history window is the bigger lever: at three years the same sweep is
    832 pages instead of 2,772.

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
WORKERS = 1              # scoring threads; one, and measured — see below

# One scoring thread, and this is not a concession to responsiveness: it is
# simply faster. Scoring is Python-level work under one interpreter lock, so a
# second thread adds contention and buys no parallelism — measured at 31 vs 25
# symbols a second for one worker against two, and 21 for four.
#
# It is also the whole of the stutter. With two workers a page request's 95th
# percentile was 636ms while the sweep ran; with one it is 18ms at the same
# slice size and the same throughput. The slice was blamed for that at first and
# cut to three, which cost two thirds of the sweep's progress while someone was
# at the terminal and bought nothing — 22ms at the 95th percentile against 18ms
# for a slice of eight. Twelve pushes it back up to 48ms, so eight is the size.

# How long the user must be quiet before the sweep will do any work, and how
# long it waits before asking again. Measured, not guessed: scoring on four
# threads with no yielding took a page load from 3.7ms to 172ms — a 46x
# regression, and exactly the stutter this whole design exists to avoid.
# How long after real input the sweep waits before touching the machine again.
# Raising this to six seconds made no measurable difference to either latency or
# throughput — someone clicking steadily never reaches even a second and a half
# of quiet, and someone who has left is past both thresholds — so it stays at
# the smaller value, which starves the sweep less in the gaps.
IDLE_BEFORE_WORK = 1.5
IDLE_POLL = 0.25

# Names that are tradable but are not companies. Alpaca's asset list carries
# warrants, units, rights and preferred shares, which have no meaningful regime
# and would only dilute the leaderboard.
_SKIP_WORDS = ("warrant", "unit", " right", "rights", "preferred", "depositary",
               "when issued", "when-issued", "notes due", "%")

# Alpaca marks OTC names tradable but publishes no bars for them, so a sweep
# that includes them spends its time collecting guaranteed failures.
_OTC_EXCHANGES = {"OTC", "OTCM", "OTCBB", "PINK", "GREY"}


def is_otc(exchange: str) -> bool:
    return (exchange or "").strip().upper() in _OTC_EXCHANGES


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
    # A share of the universe above which "no data published" is not credible.
    # OTC and delisted names are already filtered out before this applies, so
    # what remains is tradable, listed and expected to report. A third of that
    # going dark at once is not a market condition, it is a bug's residue — the
    # case this was written for had marked 57% of the curated universe dead in a
    # single pass that never fetched anything.
    DEAD_LIMIT = 0.35

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
        self.skipped = 0
        # Symbols the feed publishes nothing for. Held on disk, not just in
        # memory: an in-memory set is forgotten at every launch, and the market
        # carries a few thousand of these, so a restart meant rediscovering
        # every one of them the slow way before any real work got done. What a
        # feed does not publish today it almost certainly will not publish
        # tomorrow, so this is worth keeping.
        self.no_data: set = self._load_no_data()
        self.last_chunk: float | None = None
        self.started: float | None = None
        self.last_error: str | None = None
        self.last_benchmark: dict | None = None

    # ── remembering what has no data ────────────────────────────────────────
    def _no_data_path(self) -> str:
        import os

        from .accounts import default_config_dir

        return os.path.join(default_config_dir(), "no_data.json")

    def _load_no_data(self) -> set:
        import json

        try:
            with open(self._no_data_path(), encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, ValueError, OSError):
            return set()
        syms = data.get("symbols") if isinstance(data, dict) else data
        return {str(x).upper() for x in syms} if isinstance(syms, list) else set()

    def save_no_data(self) -> int:
        """Persist the dead-symbol set. Best-effort — losing it costs time only."""
        import json
        import os

        from .accounts import default_config_dir

        try:
            os.makedirs(default_config_dir(), exist_ok=True)
            path = self._no_data_path()
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"symbols": sorted(self.no_data)}, f)
            os.replace(tmp, path)
        except OSError:
            return 0
        return len(self.no_data)

    # ── universe ────────────────────────────────────────────────────────────
    def build_universe(self) -> list[str]:
        """Every tradable common stock the connected account can reach.

        Two exclusions beyond the name filter, both of which the sweep learned
        the hard way. OTC listings are marked tradable by Alpaca in their
        thousands but carry no published bars, so every one of them was a
        guaranteed failed fetch, retried on every pass forever. And symbols
        already found to have no data are remembered, because the second
        discovery is as worthless as the first.

        Falls back to the curated groups when there is no connection, so the
        sweep still does something useful offline instead of nothing.
        """
        from .web import SCAN_ALL

        syms = self.state.alpaca_symbols()
        if not syms:
            # Offline, the curated groups still make a universe — but the dead
            # list has to apply here too, or a poisoned one is invisible.
            return self._live(list(SCAN_ALL))
        names = getattr(self.state, "_alpaca_names", {}) or {}
        exch = getattr(self.state, "_alpaca_exchange", {}) or {}
        candidates = [s for s in sorted(syms)
                      if is_common_stock(s, names.get(s, ""))
                      and not is_otc(exch.get(s, ""))]
        return self._live(candidates) or self._live(list(SCAN_ALL))

    def _live(self, candidates: list[str]) -> list[str]:
        """Drop the symbols known to publish nothing — unless nearly all of them
        are, which means the list is wrong rather than the market.

        An earlier version of `run_chunk` recorded a dead symbol whenever the
        fetch had not run at all, so one launch without credentials wrote the
        entire universe to disk as unpublishable and the scanner's "Everything"
        view stayed empty from then on. Connecting an account afterwards did not
        help, because the exclusion was already persisted. The fetch bug is
        fixed; this repairs the files it already wrote.
        """
        if not candidates:
            return []
        dead = sum(1 for s in candidates if s in self.no_data)
        if dead >= max(20, int(len(candidates) * self.DEAD_LIMIT)):
            # Not a market in which four hundred companies stopped reporting on
            # the same afternoon.
            self.no_data.clear()
            self.save_no_data()
            self.last_error = (f"cleared {dead} symbols wrongly marked as having "
                               "no data — they will be retried")
            return list(candidates)
        return [s for s in candidates if s not in self.no_data]

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
    # A regime read on a symbol you could never fill is wasted work and, worse,
    # clutters the leaderboard with names that look good and cannot be traded.
    MIN_PRICE = 3.0
    MIN_BARS = 300          # roughly fourteen months — enough to label regimes

    def tradable_enough(self, frame) -> bool:
        """Cheap liquidity screen applied after the bars arrive.

        Price and history length only. Alpaca's daily bars carry volume, but the
        engine does not use it elsewhere and a price floor plus a history floor
        already removes the shells, the sub-dollar names and the recent listings
        that have no past to count.
        """
        if frame is None:
            # Nothing in hand to judge by — demo mode, or a path that fetches
            # per-symbol later. Absence of data is not evidence of illiquidity,
            # so let it through and let the scorer decide.
            return True
        if len(frame) < self.MIN_BARS:
            return False
        try:
            return float(frame["Close"].iloc[-1]) >= self.MIN_PRICE
        except Exception:  # noqa: BLE001
            return False

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
        stats = self.state.prefetch_ohlc(batch) or {}

        # A symbol is only dead if the feed was asked and had nothing for it.
        #
        # This distinction is the whole of a bug that killed the scanner's
        # "Everything" view outright. Without credentials — or during an outage,
        # or a rate limit — prefetch does no work at all and returns having
        # touched nothing. Reading "not in the cache" as "no data published"
        # then marked every symbol in the chunk dead, wrote all of them to
        # no_data.json, and `build_universe` excluded them from then on. One run
        # before connecting an account was enough to blacklist the entire market
        # permanently, and connecting one afterwards did not undo it.
        #
        # So the verdict is only recorded when the fetch actually ran.
        got = sum(1 for sym in batch if sym in self.state._ohlc_cache)
        fetch_ran = bool(stats.get("requested") or stats.get("cached") or got)
        before_dead = len(self.no_data)
        if not fetch_ran:
            self.last_error = ("no price data available — connect an Alpaca "
                               "account under Accounts")
            return {"scored": 0, "cursor": self.cursor, "size": len(self.universe),
                    "blocked": self.last_error}
        if got:
            # And never on a wholesale miss. A batch of two hundred where not
            # one symbol returned is an infrastructure failure, not two hundred
            # delisted companies.
            for sym in batch:
                if sym not in self.state._ohlc_cache:
                    self.no_data.add(sym)

        # Drop the untradable before paying to score them. This is the cheapest
        # possible filter and it runs on data already in hand.
        worth = [s for s in batch
                 if s not in self.no_data
                 and self.tradable_enough((self.state._ohlc_cache.get(s) or (0, None, ""))[1])]
        self.skipped += len(batch) - len(worth)

        # Score in small slices, standing aside between each. Scoring the whole
        # chunk in one go is what made the terminal stutter: the sweep has all
        # day, the person using it does not.
        rows = []
        for i in range(0, len(worth), SLICE):
            if self._stop.is_set():
                break
            self.yield_to_user()
            rows.extend(score_symbols(self.state, worth[i: i + SLICE], workers=WORKERS) or [])

        with self._lock:
            for r in rows:
                sym = r.get("symbol") or r.get("ticker")
                if sym:
                    self.board[sym] = r
            self._trim()
            self.scanned += len(batch)
        self._evict(batch)

        if len(self.no_data) != before_dead:
            self.save_no_data()

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

    # ── measurement ─────────────────────────────────────────────────────────
    #
    # The network half of a sweep cannot be predicted from first principles: it
    # depends on the plan's rate limit, the link, and how Alpaca is feeling. So
    # rather than ship an estimate dressed up as a fact, the sweep can time
    # itself against the real account and report what it actually saw.
    #
    # The arithmetic that *is* knowable is worth stating, because it sets the
    # floor. Alpaca pages at 10,000 bars, so a ten-year request for 240 symbols
    # is roughly 61 sequential pages inside one SDK call, not one round-trip.
    # Batching still wins — 2,772 pages for the whole market against 11,000
    # individual requests — but it is a 4x saving, not the 240x that "one
    # request instead of two hundred and forty" would suggest.
    PAGE_BARS = 10_000
    BARS_PER_YEAR = 252

    def page_projection(self, n_symbols: int, years: int) -> dict:
        """Requests a full pass will cost, and what a shorter window would save.

        Knowable without a connection: it follows from the universe size, the
        history window and Alpaca's page limit. The history window is the only
        lever that moves network cost rather than CPU cost, so it is priced here
        rather than left to be guessed at.
        """
        n = max(int(n_symbols), 1)
        pages = lambda y: max(1, -(-(y * self.BARS_PER_YEAR * n) // self.PAGE_BARS))
        out = {"projectedPagesFullPass": pages(years)}
        for alt in (3, 5):
            out[f"projectedPagesAt{alt}y"] = pages(alt)
        return out

    def benchmark(self, sample: int = 60) -> dict:
        """Time a real fetch-and-score cycle and project a full pass from it.

        Downloads live data on purpose — that is the point. Uses symbols the
        cache does not already hold, so it measures a cold fetch rather than
        the speed of a dictionary lookup.
        """
        from .scanner import score_symbols

        universe = self.universe or self.build_universe()
        years = int(getattr(self.state.settings, "years", 10) or 10)
        bars_each = years * self.BARS_PER_YEAR

        picked, seen = [], set()
        for sym in universe:
            if sym in seen or sym in self.state._ohlc_cache:
                continue
            seen.add(sym)
            picked.append(sym)
            if len(picked) >= sample:
                break
        if not picked:
            return {"ok": False, "reason": "nothing uncached left to measure"}
        pages = self.page_projection(len(universe), years)
        if self.state.demo or not getattr(self.state.settings, "has_credentials", False):
            # Reporting zeros here would look like an immeasurably fast network
            # rather than the absence of one. Say which it is — and still give
            # the half of the answer that does not need a connection at all.
            return {"ok": False, "sampled": 0,
                    "reason": "no live connection — connect an Alpaca account to "
                              "measure fetch speed on your own link",
                    "universeSize": len(universe), "years": years, **pages}

        t0 = time.perf_counter()
        stats = self.state.prefetch_ohlc(picked)
        fetch_s = time.perf_counter() - t0

        got = [s for s in picked if s in self.state._ohlc_cache]
        t1 = time.perf_counter()
        rows = score_symbols(self.state, got, workers=WORKERS) or []
        score_s = time.perf_counter() - t1

        bars = sum(len(self.state._ohlc_cache[s][1]) for s in got) or 1
        self._evict(picked)

        n_uni = len(universe) or 1
        per_symbol = (fetch_s + score_s) / max(len(got), 1)
        pages_sample = max(1, -(-bars // self.PAGE_BARS))

        out = {
            **pages,
            "ok": True,
            "sampled": len(picked),
            "fetched": stats.get("fetched", 0),
            "scored": len(rows),
            "barsDownloaded": bars,
            "fetchSeconds": round(fetch_s, 2),
            "scoreSeconds": round(score_s, 2),
            "secondsPerSymbol": round(per_symbol, 3),
            "symbolsPerSecond": round(len(got) / max(fetch_s + score_s, 1e-6), 1),
            "pagesThisSample": pages_sample,
            "universeSize": n_uni,
            "projectedFullPassMin": round(per_symbol * n_uni / 60.0, 1),
            "years": years,
            "at": time.time(),
        }
        self.last_benchmark = out
        return out

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
            "skipped": self.skipped,
            "noData": len(self.no_data),
            "boardSize": len(self.board),
            "lastChunk": self.last_chunk,
            "started": self.started,
            "lastError": self.last_error,
            "chunk": CHUNK,
            "benchmark": self.last_benchmark,
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
