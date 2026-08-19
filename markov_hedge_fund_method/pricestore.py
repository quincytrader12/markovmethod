"""A price store on disk, so the market is downloaded once and not again.

The full-market sweep spent most of its wall-clock re-downloading ten years of
daily bars for eleven thousand symbols, every single pass. That is roughly 900MB
and, on a free Alpaca plan capped at 200 requests a minute, about fourteen
minutes — and almost all of it was history that had not changed since the last
pass. Only the newest day or two is ever new.

So the sweep keeps what it fetched. The first pass is genuinely expensive; every
pass after it asks only for bars since the last one it saw, which is a handful
of rows per symbol instead of two and a half thousand.

Three decisions worth stating:

  * **Sharded by first character.** Eleven thousand individual files punishes
    every filesystem; one giant file has to be rewritten in full to add a single
    symbol. Around three dozen shards is the middle, and a shard loads in
    milliseconds.

  * **float32, not float64.** Prices carry maybe seven significant digits and
    the engine's own labelling threshold is 2%. Half the bytes for no loss that
    reaches a decision.

  * **Corrupt shards are deleted, not repaired.** A truncated shard from a
    killed process is indistinguishable from a valid one until it fails to
    parse. The data is re-downloadable; guessing at its contents is not worth
    the risk of feeding a wrong price into a regime read.
"""

from __future__ import annotations

import gzip
import os
import pickle
import threading

import numpy as np
import pandas as pd

from .accounts import default_config_dir

_COLS = ("Open", "High", "Low", "Close")


def _shard_for(symbol: str) -> str:
    c = (symbol or "_")[0].upper()
    return c if c.isalnum() else "_"


class PriceStore:
    """Persistent daily bars, keyed by symbol, sharded on disk."""

    def __init__(self, config_dir: str | None = None):
        base = config_dir or default_config_dir()
        self.dir = os.path.join(base, "prices")
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}       # shard -> {symbol: payload}
        self._dirty: set[str] = set()

    # ── shard io ────────────────────────────────────────────────────────────
    def _path(self, shard: str) -> str:
        return os.path.join(self.dir, f"{shard}.pkl.gz")

    def _load_shard(self, shard: str) -> dict:
        if shard in self._cache:
            return self._cache[shard]
        data: dict = {}
        path = self._path(shard)
        try:
            with gzip.open(path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001 — truncated or unreadable
            # Re-downloadable data is not worth salvaging badly.
            try:
                os.remove(path)
            except OSError:
                pass
        self._cache[shard] = data
        return data

    def flush(self) -> int:
        """Write every dirty shard. Returns how many were written."""
        with self._lock:
            dirty = list(self._dirty)
            self._dirty.clear()
        if not dirty:
            return 0
        os.makedirs(self.dir, exist_ok=True)
        written = 0
        for shard in dirty:
            data = self._cache.get(shard)
            if data is None:
                continue
            path = self._path(shard)
            tmp = path + ".tmp"
            try:
                with gzip.open(tmp, "wb", compresslevel=4) as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
                written += 1
            except Exception:  # noqa: BLE001 — a failed write must not kill the sweep
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        return written

    # ── reading ─────────────────────────────────────────────────────────────
    def get(self, symbol: str) -> pd.DataFrame | None:
        symbol = symbol.upper()
        with self._lock:
            rec = self._load_shard(_shard_for(symbol)).get(symbol)
        if not rec:
            return None
        idx = pd.to_datetime(np.asarray(rec["t"], dtype="int64"), unit="s")
        frame = pd.DataFrame({c: np.asarray(rec[c], dtype="float64") for c in _COLS},
                             index=idx)
        return frame if not frame.empty else None

    def last_date(self, symbol: str) -> pd.Timestamp | None:
        symbol = symbol.upper()
        with self._lock:
            rec = self._load_shard(_shard_for(symbol)).get(symbol)
        if not rec or not len(rec["t"]):
            return None
        return pd.to_datetime(int(rec["t"][-1]), unit="s")

    def known(self, symbols: list[str]) -> dict[str, pd.Timestamp]:
        """Last stored date for each symbol that the store already holds."""
        out = {}
        for s in symbols:
            d = self.last_date(s)
            if d is not None:
                out[s.upper()] = d
        return out

    # ── writing ─────────────────────────────────────────────────────────────
    def put(self, symbol: str, frame: pd.DataFrame) -> None:
        """Store or extend a symbol's bars, keeping one row per day."""
        symbol = symbol.upper()
        if frame is None or frame.empty or not set(_COLS) <= set(frame.columns):
            return
        existing = self.get(symbol)
        if existing is not None and not existing.empty:
            frame = pd.concat([existing, frame[list(_COLS)]])
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        else:
            frame = frame[list(_COLS)].sort_index()

        # Convert through datetime64[s] rather than dividing a raw integer.
        # pandas 3 stores datetimes at microsecond resolution where pandas 2 used
        # nanoseconds, so any hard-coded divisor is right on one and silently
        # wrong on the other — it turned 2026 dates into 1970, which made every
        # stored symbol look stale and defeated the entire point of the store.
        # This form asks pandas for seconds and does not care how it holds them.
        secs = pd.DatetimeIndex(frame.index).to_numpy().astype("datetime64[s]").astype("int64")
        rec = {"t": secs}
        for c in _COLS:
            rec[c] = frame[c].to_numpy(dtype="float32")
        shard = _shard_for(symbol)
        with self._lock:
            self._load_shard(shard)[symbol] = rec
            self._dirty.add(shard)

    def put_many(self, frames: dict) -> int:
        for sym, df in frames.items():
            self.put(sym, df)
        return len(frames)

    # ── housekeeping ────────────────────────────────────────────────────────
    def stats(self) -> dict:
        symbols = bars = 0
        for shard in self._shards_on_disk():
            data = self._load_shard(shard)
            symbols += len(data)
            bars += sum(len(r["t"]) for r in data.values())
        return {"symbols": symbols, "bars": bars,
                "diskBytes": self.disk_bytes(), "dir": self.dir}

    def _shards_on_disk(self) -> list[str]:
        try:
            return [f.split(".")[0] for f in os.listdir(self.dir) if f.endswith(".pkl.gz")]
        except OSError:
            return []

    def disk_bytes(self) -> int:
        total = 0
        try:
            for f in os.listdir(self.dir):
                if f.endswith(".pkl.gz"):
                    total += os.path.getsize(os.path.join(self.dir, f))
        except OSError:
            pass
        return total

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._dirty.clear()
        try:
            for f in os.listdir(self.dir):
                if f.endswith((".pkl.gz", ".tmp")):
                    os.remove(os.path.join(self.dir, f))
        except OSError:
            pass
