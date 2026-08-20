"""The watchlist, kept on the server rather than in the page.

It lived in the browser's DOM — the list was literally read back out of the
rendered rows — which had two consequences. Refreshing the page lost it, and
nothing outside the browser could add to it. The second one is what a Telegram
button runs into: a message arrives on your phone offering to add a name, and
there is nowhere to put it.

So it moves here: a small ordered list on disk, alongside the account registry,
with the browser and the Telegram bot as two clients of the same store rather
than one owning it and the other locked out.

Order is preserved and newest-first, because a watchlist is a working list and
the thing you just added is the thing you are about to look at.
"""

from __future__ import annotations

import json
import os
import threading

from .accounts import default_config_dir

DEFAULTS = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT", "BTC-USD"]
MAX_SYMBOLS = 200


class WatchlistStore:
    def __init__(self, config_dir: str | None = None):
        self.config_dir = config_dir or default_config_dir()
        self.path = os.path.join(self.config_dir, "watchlist.json")
        self._lock = threading.Lock()

    # ── io ──────────────────────────────────────────────────────────────────
    def _read(self) -> list[str] | None:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, ValueError, OSError):
            return None
        if isinstance(data, dict):
            data = data.get("symbols")
        if not isinstance(data, list):
            return None
        return [str(s).strip().upper() for s in data if str(s).strip()]

    def _write(self, symbols: list[str]) -> None:
        os.makedirs(self.config_dir, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"symbols": symbols}, f, indent=2)
        os.replace(tmp, self.path)

    # ── api ─────────────────────────────────────────────────────────────────
    def list(self) -> list[str]:
        """Current symbols. A first run gets the defaults, not an empty screen."""
        stored = self._read()
        return list(DEFAULTS) if stored is None else stored

    def add(self, symbol: str) -> dict:
        """Add one symbol. Returns whether it was new, so a caller can say so.

        Idempotent on purpose: the Telegram button can be tapped twice, or the
        same name can arrive in two different alerts, and neither should
        duplicate a row or read as a failure.
        """
        sym = (symbol or "").strip().upper()
        if not sym:
            return {"ok": False, "added": False, "reason": "empty symbol"}
        with self._lock:
            current = self.list()
            if sym in current:
                return {"ok": True, "added": False, "symbol": sym,
                        "reason": "already on the watchlist", "symbols": current}
            if len(current) >= MAX_SYMBOLS:
                return {"ok": False, "added": False, "symbol": sym,
                        "reason": f"watchlist is full ({MAX_SYMBOLS})", "symbols": current}
            current = [sym] + current          # newest first: it is what you want to see
            self._write(current)
        return {"ok": True, "added": True, "symbol": sym, "symbols": current}

    def remove(self, symbol: str) -> dict:
        sym = (symbol or "").strip().upper()
        with self._lock:
            current = self.list()
            if sym not in current:
                return {"ok": True, "removed": False, "symbols": current}
            current = [s for s in current if s != sym]
            self._write(current)
        return {"ok": True, "removed": True, "symbol": sym, "symbols": current}

    def replace(self, symbols: list) -> list[str]:
        """Overwrite wholesale — used when the browser reorders or bulk-edits."""
        seen: dict[str, None] = {}
        for raw in symbols or []:
            sym = str(raw).strip().upper()
            if sym:
                seen.setdefault(sym, None)
        out = list(seen)[:MAX_SYMBOLS]
        with self._lock:
            self._write(out)
        return out

    def contains(self, symbol: str) -> bool:
        return (symbol or "").strip().upper() in self.list()
