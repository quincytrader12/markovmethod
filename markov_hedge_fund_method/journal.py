"""Trade journal — a persistent log of trades with regime context.

Every order placed through the terminal is auto-logged here with the regime the
symbol was in at entry, so the journal can answer "how do I do when I trade in
Bull vs Bear vs Sideways?". Entries carry free-form tags + notes, and an
optional realized P&L / R-multiple the user fills in when a trade is closed.

Backed by a JSON file in the same config dir the account registry uses, so it
survives restarts. Pure-Python, no external deps — safe to import anywhere.
"""

from __future__ import annotations

import json
import os
import time
import uuid

from .accounts import default_config_dir

_REGIMES = ("bear", "sideways", "bull")


class JournalStore:
    def __init__(self, config_dir: str | None = None):
        self.config_dir = config_dir or default_config_dir()
        self.path = os.path.join(self.config_dir, "journal.json")

    # ── persistence ─────────────────────────────────────────────────────────
    def _read(self) -> list[dict]:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (FileNotFoundError, ValueError, OSError):
            return []

    def _write(self, entries: list[dict]) -> None:
        os.makedirs(self.config_dir, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        os.replace(tmp, self.path)

    # ── CRUD ────────────────────────────────────────────────────────────────
    def list(self) -> list[dict]:
        """Newest first."""
        return sorted(self._read(), key=lambda e: e.get("ts", 0), reverse=True)

    def add(self, *, symbol: str, side: str, qty=None, price=None, regime: str = "",
            tags=None, notes: str = "", pnl=None, r_multiple=None,
            source: str = "manual") -> dict:
        entry = {
            "id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "symbol": str(symbol).upper(),
            "side": str(side).lower(),
            "qty": qty,
            "price": price,
            "regime": regime if regime in _REGIMES else "",
            "tags": [str(t).strip() for t in (tags or []) if str(t).strip()],
            "notes": str(notes or ""),
            "pnl": None if pnl is None else float(pnl),
            "rMultiple": None if r_multiple is None else float(r_multiple),
            "source": source,
        }
        entries = self._read()
        entries.append(entry)
        self._write(entries)
        return entry

    def update(self, entry_id: str, **fields) -> dict | None:
        entries = self._read()
        allowed = {"tags", "notes", "pnl", "rMultiple", "regime"}
        for e in entries:
            if e.get("id") == entry_id:
                for k, v in fields.items():
                    if k not in allowed:
                        continue
                    if k == "tags":
                        e[k] = [str(t).strip() for t in (v or []) if str(t).strip()]
                    elif k in ("pnl", "rMultiple"):
                        e[k] = None if v is None else float(v)
                    else:
                        e[k] = v
                self._write(entries)
                return e
        return None

    def remove(self, entry_id: str) -> bool:
        entries = self._read()
        kept = [e for e in entries if e.get("id") != entry_id]
        if len(kept) == len(entries):
            return False
        self._write(kept)
        return True

    # ── analytics ───────────────────────────────────────────────────────────
    def analytics(self) -> dict:
        """Journal performance grouped by the regime each trade was entered in.

        Uses only entries that carry a realized P&L (closed trades); open/
        unlogged trades are counted but don't skew win-rate.
        """
        entries = self._read()
        by_regime = {r: {"trades": 0, "closed": 0, "wins": 0, "pnl": 0.0,
                         "rSum": 0.0, "rCount": 0} for r in _REGIMES}
        by_regime["untagged"] = {"trades": 0, "closed": 0, "wins": 0, "pnl": 0.0,
                                 "rSum": 0.0, "rCount": 0}
        total_pnl = 0.0
        total_closed = 0
        total_wins = 0
        for e in entries:
            r = e.get("regime") or "untagged"
            if r not in by_regime:
                r = "untagged"
            b = by_regime[r]
            b["trades"] += 1
            pnl = e.get("pnl")
            if pnl is not None:
                b["closed"] += 1
                b["pnl"] += float(pnl)
                total_pnl += float(pnl)
                total_closed += 1
                if float(pnl) > 0:
                    b["wins"] += 1
                    total_wins += 1
            rm = e.get("rMultiple")
            if rm is not None:
                b["rSum"] += float(rm)
                b["rCount"] += 1

        def finalize(b):
            return {
                "trades": b["trades"],
                "closed": b["closed"],
                "winRate": (b["wins"] / b["closed"]) if b["closed"] else None,
                "pnl": round(b["pnl"], 2),
                "avgR": round(b["rSum"] / b["rCount"], 2) if b["rCount"] else None,
            }

        return {
            "byRegime": {r: finalize(b) for r, b in by_regime.items()},
            "totals": {
                "trades": len(entries),
                "closed": total_closed,
                "winRate": (total_wins / total_closed) if total_closed else None,
                "pnl": round(total_pnl, 2),
            },
        }
