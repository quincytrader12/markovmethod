"""Alerts + risk automation for the terminal.

Three jobs, all in-memory (single local user):

  * regime-flip alerts — remember each watched symbol's last regime and emit an
    event the moment it changes (Bull→Bear etc.), the terminal's signature alert.
  * price alerts        — one-shot "SYM above/below X" rules.
  * risk automation     — a manual kill switch and an optional daily-loss limit
    that trips the kill switch automatically. When tripped, `halted` blocks new
    orders until the user resets it (the flatten action lives in the broker).

Pure Python, no deps — the web layer feeds it fresh regimes/prices/P&L and acts
on the events it returns.
"""

from __future__ import annotations

import time


class AlertEngine:
    MAX_EVENTS = 200

    def __init__(self):
        self._last_regime: dict[str, str] = {}
        self._events: list[dict] = []
        self._price_alerts: list[dict] = []
        self.loss_limit: float | None = None
        self.halted: bool = False
        self._seq = 0

    # ── event helpers ───────────────────────────────────────────────────────
    def _emit(self, ev: dict) -> dict:
        self._seq += 1
        ev["id"] = f"ev{self._seq}"
        ev["ts"] = time.time()
        ev["at"] = time.strftime("%H:%M:%S")
        self._events.append(ev)
        if len(self._events) > self.MAX_EVENTS:
            self._events = self._events[-self.MAX_EVENTS:]
        return ev

    def recent(self, limit: int = 40) -> list[dict]:
        return self._events[-limit:][::-1]

    # ── regime-flip detection ───────────────────────────────────────────────
    def check_regimes(self, regimes: dict[str, str]) -> list[dict]:
        """Compare current regimes to last-seen; emit a flip event on change.
        First sighting of a symbol only seeds state (no event)."""
        out = []
        for sym, reg in regimes.items():
            if not reg:
                continue
            prev = self._last_regime.get(sym)
            self._last_regime[sym] = reg
            if prev is not None and prev != reg:
                out.append(self._emit({
                    "type": "regime_flip", "symbol": sym, "from": prev, "to": reg,
                    "level": "bull" if reg == "bull" else "bear" if reg == "bear" else "info",
                    "message": f"{sym} regime flip: {prev.upper()} → {reg.upper()}"}))
        return out

    # ── price alerts ────────────────────────────────────────────────────────
    def add_price_alert(self, symbol: str, op: str, price: float) -> dict:
        self._seq += 1
        alert = {"id": f"pa{self._seq}", "symbol": symbol.upper(),
                 "op": op, "price": float(price), "active": True}
        self._price_alerts.append(alert)
        return alert

    def remove_price_alert(self, alert_id: str) -> bool:
        n = len(self._price_alerts)
        self._price_alerts = [a for a in self._price_alerts if a["id"] != alert_id]
        return len(self._price_alerts) != n

    def price_alerts(self) -> list[dict]:
        return list(self._price_alerts)

    def check_prices(self, prices: dict[str, float]) -> list[dict]:
        out = []
        for a in self._price_alerts:
            if not a["active"]:
                continue
            last = prices.get(a["symbol"])
            if last is None:
                continue
            hit = ((a["op"] == "above" and last >= a["price"]) or
                   (a["op"] == "below" and last <= a["price"]))
            if hit:
                a["active"] = False  # one-shot
                out.append(self._emit({
                    "type": "price", "symbol": a["symbol"], "op": a["op"],
                    "price": a["price"], "last": round(float(last), 4), "level": "info",
                    "message": f"{a['symbol']} {a['op']} {a['price']:g} — now {last:g}"}))
        return out

    # ── risk automation ─────────────────────────────────────────────────────
    def set_loss_limit(self, value) -> None:
        if value in (None, "", 0, 0.0):
            self.loss_limit = None
        else:
            self.loss_limit = abs(float(value))

    def trip_kill(self, reason: str = "") -> dict:
        self.halted = True
        msg = "KILL SWITCH — trading halted"
        if reason:
            msg += f" ({reason})"
        return self._emit({"type": "kill", "reason": reason or "manual",
                           "level": "bear", "message": msg})

    def reset_kill(self) -> None:
        self.halted = False

    def check_loss_limit(self, day_pl: float) -> dict | None:
        """Trip the kill switch if day P&L breaches the (negative) limit."""
        if self.loss_limit is None or self.halted:
            return None
        if day_pl <= -abs(self.loss_limit):
            return self.trip_kill(
                f"daily loss limit ${self.loss_limit:g} hit — day P&L ${day_pl:,.0f}")
        return None
