"""Running the lab across many symbols, and paying the statistical price for it.

Searching one symbol is a search of a hundred and fifty-four. Searching fifty is
a search of seven and a half thousand, and the best result out of it has to clear
a far higher bar than the best result out of any one name. Reporting a winner
chosen across all of them with the deflated Sharpe from its own symbol is the
lab's founding mistake committed one level up — so the pooling in `lab.pool` is
not a nicety here, it is the reason this module is allowed to exist.

Two further traps come with breadth, and both are reported rather than hidden.

A leaderboard from a sweep is usually one idea wearing several tickers. Momentum
on five technology names is a single bet held five times, and it presents itself
as five independent confirmations; `lab.breadth` measures the correlation of the
winners' returns and says how many genuinely different bets are actually there.

And a watchlist is a survivorship-biased sample by construction — it is a list of
names someone already liked. A strategy that made money on one of them may have
done nothing except be long something that went up, so every candidate carries
its Sharpe above buy-and-hold on the same symbol over the same period.

It runs in the background and stands aside for the user exactly as the market
sweep does, for the reason measured there: scoring is CPU-bound work under one
interpreter lock, and a research job has all day while the person watching does
not.
"""

from __future__ import annotations

import threading
import time

from . import lab

# One scoring thread, matching the market sweep — and measured the same way.
# Extra threads add contention under the interpreter lock and buy no
# parallelism; the sweep's own numbers were 31 symbols a second on one worker
# against 25 on two.
WORKERS = 1

# Seconds of genuine quiet before the sweep touches the machine, and how long a
# busy session may starve it before it takes one symbol anyway.
IDLE_BEFORE_WORK = 1.5
IDLE_POLL = 0.25
STARVE_TIMEOUT = 30.0


class LabSweep:
    """Searches a list of symbols, then judges the results as one search."""

    def __init__(self, state, config_dir: str | None = None):
        self.state = state
        self._dir = config_dir
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self.symbols: list[str] = []
        self.done: list[str] = []
        self.skipped: list[dict] = []
        self.results: list = []
        self.pooled: dict = {}
        self.spread: dict = {}
        self.summary_text = ""
        self.started_at = 0.0
        self.finished_at = 0.0
        self.last_error: str | None = None
        self.options: dict = {}

    # ── lifecycle ───────────────────────────────────────────────────────────
    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def config_dir(self) -> str:
        if self._dir:
            return self._dir
        from .accounts import default_config_dir
        return default_config_dir()

    def start(self, symbols: list[str], **options) -> dict:
        if self.running:
            return self.status()
        syms = [s.strip().upper() for s in symbols if s and s.strip()]
        # De-duplicate but keep the order the user gave, so progress reads the
        # way the watchlist looks.
        seen: set[str] = set()
        self.symbols = [s for s in syms if not (s in seen or seen.add(s))]
        self.done, self.skipped, self.results = [], [], []
        self.pooled, self.spread, self.summary_text = {}, {}, ""
        self.last_error = None
        self.options = dict(options)
        self.started_at, self.finished_at = time.time(), 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="lab-sweep")
        self._thread.start()
        return self.status()

    def stop(self) -> None:
        self._stop.set()

    def yield_to_user(self, timeout: float = STARVE_TIMEOUT) -> None:
        """Stand aside while the terminal is being used."""
        idle_fn = getattr(self.state, "idle_for", None)
        if idle_fn is None:
            return
        waited = 0.0
        while idle_fn() < IDLE_BEFORE_WORK and waited < timeout:
            if self._stop.is_set():
                return
            time.sleep(IDLE_POLL)
            waited += IDLE_POLL

    # ── the work ────────────────────────────────────────────────────────────
    def _run(self) -> None:
        try:
            for sym in list(self.symbols):
                if self._stop.is_set():
                    break
                self.yield_to_user()
                if self._stop.is_set():
                    break
                self._one(sym)
            self._finish()
        except Exception as exc:  # noqa: BLE001 — a sweep must not die silently
            self.last_error = str(exc)
            self.finished_at = time.time()

    def _one(self, symbol: str) -> None:
        try:
            df, source = self.state.ohlc_for(symbol)
        except Exception as exc:  # noqa: BLE001
            self._skip(symbol, f"could not load history: {exc}")
            return
        if df is None or df.empty or len(df) < 300:
            self._skip(symbol, "not enough history to search")
            return
        if str(source).startswith("synthetic"):
            # A search over invented prices describes the generator, not a
            # market, and pooling it in would corrupt every other symbol's bar.
            self._skip(symbol, "no real price data")
            return

        close = df["Close"]
        years = int(self.options.get("years") or 0)
        if years > 0:
            close = close.iloc[-int(years * 252):]
        kw = {"symbol": symbol,
              "holdout_frac": float(self.options.get("holdout_frac") or 0.25),
              "long_only": bool(self.options.get("long_only"))}
        if self.options.get("cost_bps"):
            kw["cost_bps"] = float(self.options["cost_bps"])

        try:
            result = lab.search(close, **kw)
        except Exception as exc:  # noqa: BLE001 — one bad symbol, not the sweep
            self._skip(symbol, str(exc))
            return
        if not result.trials:
            self._skip(symbol, "no usable track record")
            return

        with self._lock:
            self.results.append(result)
            self.done.append(symbol)
        # Each symbol's own record is stored as it completes, so stopping a
        # sweep half way still leaves everything it learned.
        lab.save(result, self.config_dir())

    def _skip(self, symbol: str, why: str) -> None:
        with self._lock:
            self.skipped.append({"symbol": symbol, "why": why})

    def _finish(self) -> None:
        with self._lock:
            results = list(self.results)
        if results:
            self.pooled = lab.pool(results)
            self.spread = lab.breadth(results)
            self.summary_text = lab.sweep_summary(results, self.pooled, self.spread)
        self.finished_at = time.time()

    # ── reporting ───────────────────────────────────────────────────────────
    def survivors(self) -> list:
        """Candidates that cleared their holdout, with the pooled DSR attached.

        The pooled number is the one that matters: a strategy is only worth
        adopting if it survives being judged against every attempt the sweep
        made, not just the ones made on its own symbol.
        """
        by_key = {(r["symbol"], r["name"]): r for r in self.pooled.get("ranked", [])}
        out = []
        for res in self.results:
            for h in res.holdout:
                if not h.get("heldUp"):
                    continue
                pooled = by_key.get((res.symbol, h["name"]), {})
                pooled_dsr = pooled.get("dsr")
                out.append({**h, "symbol": res.symbol,
                            "pooledDsr": pooled_dsr,
                            "symbolDsr": h.get("dsr"),
                            # `dsr` is what the playbook gate reads, so it must
                            # be the pooled one. A candidate found in a sweep of
                            # eleven hundred trials has to answer to eleven
                            # hundred, not to the hundred and fifty made on its
                            # own symbol — otherwise the whole pooling is a
                            # display that nothing acts on.
                            "dsr": pooled_dsr if pooled_dsr is not None else h.get("dsr"),
                            "excess": h.get("excess"),
                            "beatBuyAndHold": h.get("beatBuyAndHold", False)})
        out.sort(key=lambda r: (r.get("pooledDsr") or 0), reverse=True)
        return out

    def status(self, top: int = 30) -> dict:
        with self._lock:
            done, skipped = list(self.done), list(self.skipped)
        total = len(self.symbols) or 1
        return {
            "running": self.running,
            "symbols": self.symbols,
            "done": done,
            "skipped": skipped,
            "progress": round(len(done) + len(skipped)) / total,
            "scanned": len(done),
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "seconds": round((self.finished_at or time.time()) - self.started_at, 1)
            if self.started_at else 0.0,
            "lastError": self.last_error,
            "nTrials": self.pooled.get("nTrials", 0),
            "luckBar": self.pooled.get("luckBar"),
            "ranked": self.pooled.get("ranked", [])[:top],
            "breadth": self.spread,
            "survivors": self.survivors() if self.pooled else [],
            "summary": self.summary_text,
            "options": self.options,
        }
