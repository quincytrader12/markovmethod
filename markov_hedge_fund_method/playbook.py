"""The playbook: strategies the lab found and the terminal is actually running.

This is the bridge between research and trading, and it is the most dangerous
file in the project — everything upstream is a number on a screen, and everything
downstream is an order. So the bridge is deliberately narrow.

**Only what survived the holdout may cross.** The lab ranks by deflated Sharpe,
which corrects for how many things were tried, and then measures the top few on a
quarter of history the search never saw. A strategy that looked wonderful where
it was fitted and did nothing afterwards is the normal outcome, not the
exception, and `adopt` refuses it outright rather than warning about it. There is
no override, because the whole point of a search is that its winner is
untrustworthy by construction, and a way round that check is a way to lose money
with the terminal's blessing.

**An adopted strategy can shrink an order or refuse it. It can never enlarge
one, flip its side, or place one on its own.** It joins the same seam the
meta-labelling forest already uses. The chain still decides direction, the user
still types the ticket, the kill switch still stops everything.

**Agreement is not evidence.** Several adopted strategies agreeing means they
correlate, which is usually because they are the same idea wearing different
parameters — the lab's own search proves how easily that happens. So the vote is
a floor on conviction, not a multiplier on it.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd

from . import strategies as st

# What a strategy must have shown on data the search never touched. Not a
# suggestion: `adopt` will not store anything that misses either bar.
MIN_HOLDOUT_SHARPE = 0.3
MIN_DSR = 0.50

# The most a playbook may hold. Adopting everything that passes is how a
# portfolio of one idea ends up looking like a portfolio of twenty.
MAX_ACTIVE = 12


class NotProven(ValueError):
    """Raised when a strategy is asked to go live without having earned it."""


def _path(config_dir: str) -> str:
    return os.path.join(config_dir, "playbook.json")


def load(config_dir: str) -> list:
    try:
        with open(_path(config_dir), encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _write(config_dir: str, rows: list) -> None:
    try:
        os.makedirs(config_dir, exist_ok=True)
        with open(_path(config_dir), "w", encoding="utf-8") as fh:
            json.dump(rows[:MAX_ACTIVE], fh, indent=1)
    except OSError:
        pass


def check(entry: dict) -> None:
    """Would this be allowed to go live? Raises with the reason if not."""
    held = entry.get("heldUp")
    hold = (entry.get("holdout") or {}).get("sharpe")
    dsr = entry.get("dsr")
    if not held:
        raise NotProven(
            "This did not hold up on the quarter of history the search never "
            "saw. That is the normal outcome for a search winner, and it is "
            "exactly what the holdout is for.")
    if hold is None or float(hold) < MIN_HOLDOUT_SHARPE:
        raise NotProven(
            f"Out-of-sample Sharpe is {hold}, below the {MIN_HOLDOUT_SHARPE} "
            "floor. Doing better than nothing where it was fitted is not enough.")
    if dsr is None or float(dsr) < MIN_DSR:
        raise NotProven(
            f"Deflated Sharpe is {dsr}, below {MIN_DSR}. Given how many "
            "combinations were tried, this one is likelier than not to be luck.")
    # The bar that a cross-symbol sweep makes unmissable: three candidates
    # survived their holdouts in one run and not one of them beat simply owning
    # the symbol. A strategy that clears every statistical test and still trails
    # buy-and-hold has earned nothing but turnover, and the entry carries the
    # comparison precisely so this can be checked rather than assumed.
    if entry.get("beatBuyAndHold") is False:
        raise NotProven(
            "It survived out of sample but did not beat simply owning the "
            "symbol over the same period. Clearing the statistics is not the "
            "same as being worth trading.")


def adopt(config_dir: str, symbol: str, entry: dict) -> dict:
    """Put a lab result into the playbook, if it has earned it."""
    check(entry)
    rows = [r for r in load(config_dir)
            if not (r["symbol"] == symbol.upper() and r["strategy"] == entry["name"])]
    row = {
        "symbol": symbol.strip().upper(),
        "strategy": entry["name"],
        "kind": entry.get("kind", ""),
        "parts": entry.get("parts", []),
        "dsr": round(float(entry.get("dsr", 0.0)), 4),
        "searchedSharpe": (entry.get("searched") or {}).get("sharpe"),
        "holdoutSharpe": (entry.get("holdout") or {}).get("sharpe"),
        "maxDrawdown": (entry.get("holdout") or {}).get("maxDrawdown"),
        "turnover": (entry.get("holdout") or {}).get("turnover"),
        "nTrials": entry.get("nTrials", 0),
        "adoptedAt": time.strftime("%Y-%m-%d %H:%M"),
        "enabled": True,
    }
    rows.insert(0, row)
    _write(config_dir, rows)
    return row


def drop(config_dir: str, symbol: str, strategy: str) -> int:
    rows = load(config_dir)
    keep = [r for r in rows
            if not (r["symbol"] == symbol.strip().upper() and r["strategy"] == strategy)]
    _write(config_dir, keep)
    return len(rows) - len(keep)


def set_enabled(config_dir: str, symbol: str, strategy: str, on: bool) -> bool:
    rows = load(config_dir)
    hit = False
    for r in rows:
        if r["symbol"] == symbol.strip().upper() and r["strategy"] == strategy:
            r["enabled"] = bool(on)
            hit = True
    if hit:
        _write(config_dir, rows)
    return hit


def for_symbol(config_dir: str, symbol: str) -> list:
    sym = symbol.strip().upper()
    return [r for r in load(config_dir) if r["symbol"] == sym and r.get("enabled", True)]


def _rebuild(row: dict):
    """Reconstruct the strategy function from what was stored.

    Only names the library still knows are rebuilt. A playbook entry whose
    strategy has since been renamed or removed is dropped rather than guessed
    at — silently substituting a different rule would be worse than not running.
    """
    lib = st.library()
    parts = list(row.get("parts") or [])
    kind = row.get("kind", "single")
    if kind == "single":
        return lib.get(parts[0] if parts else row.get("strategy", ""))
    if len(parts) != 2 or parts[0] not in lib or parts[1] not in lib:
        return None
    combiner = st.PAIRINGS.get(kind)
    return combiner(lib[parts[0]], lib[parts[1]]) if combiner else None


def consult(config_dir: str, symbol: str, close: pd.Series) -> dict:
    """What the playbook thinks of a position in this symbol, right now.

    Returns a multiplier in [0, 1] and the votes behind it. Never above one:
    this layer exists to hold back, not to press.
    """
    rows = for_symbol(config_dir, symbol)
    out = {"active": len(rows), "votes": [], "multiplier": 1.0,
           "reason": "no adopted strategy for this symbol"}
    if not rows or close is None or len(close) < 60:
        return out

    votes, usable = [], 0
    for row in rows:
        fn = _rebuild(row)
        if fn is None:
            votes.append({"strategy": row["strategy"], "position": None,
                          "note": "no longer in the library"})
            continue
        try:
            pos = float(pd.Series(fn(close)).iloc[-1])
        except Exception:  # noqa: BLE001 — a broken rule is an abstention
            continue
        usable += 1
        votes.append({"strategy": row["strategy"], "position": round(pos, 3),
                      "holdoutSharpe": row.get("holdoutSharpe")})

    if not usable:
        out["votes"] = votes
        return out

    longs = sum(1 for v in votes if (v.get("position") or 0) > 0)
    shorts = sum(1 for v in votes if (v.get("position") or 0) < 0)
    flat = usable - longs - shorts

    # A floor on conviction, not a multiplier on it. Strategies found by the same
    # search over the same history agree far more often than independent
    # evidence would, so unanimity earns full size and nothing more.
    if longs and not shorts:
        mult, reason = 1.0, f"{longs} of {usable} adopted strategies are long"
    elif shorts and not longs:
        mult, reason = 1.0, f"{shorts} of {usable} adopted strategies are short"
    elif longs and shorts:
        mult, reason = 0.5, f"adopted strategies disagree ({longs} long, {shorts} short)"
    else:
        mult, reason = 0.5, f"all {flat} adopted strategies are flat"

    out.update({"votes": votes, "multiplier": mult, "reason": reason,
                "long": longs, "short": shorts, "flat": flat})
    return out


def side_conflict(consulted: dict, side: str) -> bool:
    """Is the order's direction one every adopted strategy is against?

    The only veto this layer has, and a narrow one: it fires when the playbook
    is unanimous the other way, never on a split.
    """
    if not consulted or not consulted.get("votes"):
        return False
    longs, shorts = consulted.get("long", 0), consulted.get("short", 0)
    s = str(side or "").lower()
    if s.endswith("buy") and shorts and not longs:
        return True
    return bool(s in ("sell", "short") and longs and not shorts)
