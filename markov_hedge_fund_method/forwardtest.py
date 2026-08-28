"""Forward testing: let an adopted strategy trade paper money and keep score.

This is the honest answer to the problem the lab cannot solve on its own. A
search picks its winner from history, and no amount of deflation or holding out
changes the fact that the winner was chosen *after* seeing how the data turned
out. The one kind of evidence immune to that is a track record made on days that
had not happened yet — so the terminal builds one, on paper, in public.

The rule that governs everything here, and the reason the gate is a single
function rather than a condition scattered through the loop:

**Auto-trading is permitted on a paper account only. Never live.**

Not "discouraged in live", not "live with a confirmation" — refused. A strategy
here is by construction one that a computer chose because it fitted the past
well; the whole point of forward testing is that we do not yet know whether it
works. Handing that real money is the exact mistake this module exists to
prevent, and a mode switch is far too easy a thing to do by accident for the
protection to live anywhere but in code.

In live mode adopted strategies keep doing what they did before: sizing and
gating orders the user places. They simply never place one.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pandas as pd

from .config import Mode

# How often the loop wakes. Positions here change on daily bars, so anything
# faster is just load; anything slower risks missing a day's turn entirely.
POLL_SECONDS = 300.0

# Fraction of buying power one forward-tested strategy may command. Small: this
# is a measurement apparatus, not an allocation.
NOTIONAL_PCT = 0.05

# Below this the order is not worth the spread it would pay.
MIN_ORDER_USD = 25.0


def may_autotrade(settings) -> tuple[bool, str]:
    """The gate. One place, one answer, no exceptions.

    Paper only. This is deliberately not a preference, a setting or a
    confirmation dialog — a strategy under forward test is one whose edge is
    still unproven by definition, and the mode selector is one click away from
    LIVE at all times.
    """
    mode = getattr(settings, "mode", None)
    if mode is Mode.PAPER:
        return True, ""
    if mode is Mode.LIVE:
        return False, ("Auto-trading is disabled in LIVE. A forward test exists "
                       "because the strategy is not yet proven, and unproven is "
                       "not something to hand real money. Adopted strategies "
                       "still size and gate the orders you place.")
    return False, f"{getattr(mode, 'value', mode)} mode places no orders at all."


class ForwardTester:
    """Keeps paper positions in line with what the adopted strategies want."""

    def __init__(self, state, config_dir: str | None = None):
        self.state = state
        self._dir = config_dir
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_run = 0.0
        self.last_error: str | None = None
        self.actions: list[dict] = []
        self.blocked_reason = ""

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="forward-test")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _loop(self) -> None:
        self._stop.wait(45)              # let the terminal finish starting
        while not self._stop.is_set():
            try:
                self.tick()
                self.last_error = None
            except Exception as exc:  # noqa: BLE001 — never let the loop die
                self.last_error = str(exc)
            self._stop.wait(POLL_SECONDS)

    # ── the work ────────────────────────────────────────────────────────────
    def config_dir(self) -> str:
        if self._dir:
            return self._dir
        from .accounts import default_config_dir
        return default_config_dir()

    def desired(self, symbol: str) -> float | None:
        """The position the adopted strategies want in this symbol, -1 to +1."""
        from . import playbook

        rows = playbook.for_symbol(self.config_dir(), symbol)
        rows = [r for r in rows if r.get("forward")]
        if not rows:
            return None
        df, _src = self.state.ohlc_for(symbol)
        if df is None or df.empty:
            return None
        wants = []
        for row in rows:
            fn = playbook._rebuild(row)
            if fn is None:
                continue
            try:
                wants.append(float(pd.Series(fn(df["Close"])).iloc[-1]))
            except Exception:  # noqa: BLE001
                continue
        if not wants:
            return None
        return max(-1.0, min(1.0, sum(wants) / len(wants)))

    def tick(self) -> dict:
        """One pass: work out what each adopted strategy wants and close the gap."""
        from . import playbook

        ok, reason = may_autotrade(self.state.settings)
        self.blocked_reason = reason
        self.last_run = time.time()
        if not ok:
            return {"traded": 0, "blocked": reason}
        if self.state.broker is None:
            return {"traded": 0, "blocked": "no account connected"}
        if getattr(self.state, "alerts", None) is not None and self.state.alerts.halted:
            return {"traded": 0, "blocked": "kill switch is on"}

        symbols = sorted({r["symbol"] for r in playbook.load(self.config_dir())
                          if r.get("enabled", True) and r.get("forward")})
        if not symbols:
            return {"traded": 0, "blocked": "nothing under forward test"}

        try:
            account = self.state.broker.get_account()
            positions = {p.symbol.upper(): p for p in self.state.broker.list_positions()}
        except Exception as exc:  # noqa: BLE001
            return {"traded": 0, "blocked": f"broker read failed: {exc}"}

        traded = 0
        for sym in symbols:
            want = self.desired(sym)
            if want is None:
                continue
            try:
                if self._reconcile(sym, want, positions.get(sym), account):
                    traded += 1
            except Exception as exc:  # noqa: BLE001 — one symbol must not stop the rest
                self._log(sym, "error", str(exc))
        return {"traded": traded, "symbols": len(symbols)}

    def _reconcile(self, symbol: str, want: float, position, account) -> bool:
        """Move the paper position toward what the strategies want."""
        from .orders import OrderTicket

        equity = float(getattr(account, "equity", 0) or 0)
        budget = equity * NOTIONAL_PCT
        target_usd = want * budget

        held_usd = 0.0
        if position is not None:
            held_usd = float(getattr(position, "market_value", 0) or 0)

        delta = target_usd - held_usd
        if abs(delta) < MIN_ORDER_USD:
            return False

        side = "buy" if delta > 0 else "sell"
        ticket = OrderTicket(symbol=symbol, side=side, notional=round(abs(delta), 2),
                             order_type="market", order_class="simple",
                             time_in_force="day")
        # Straight to the broker rather than through /api/orders: the playbook
        # layer there exists to second-guess a human's ticket, and second-
        # guessing the strategy with the strategy would be circular.
        result = self.state.broker.submit_ticket(ticket)
        self._log(symbol, side, f"${abs(delta):,.0f} toward target {want:+.2f}",
                  order_id=getattr(result, "id", ""))
        try:
            self.state.journal.add(symbol=symbol, side=side, qty=None,
                                   price=None, regime=self.state.journal_regime(symbol),
                                   source="forward-test",
                                   notes=f"paper forward test, target {want:+.2f}")
        except Exception:  # noqa: BLE001 — journalling is never fatal
            pass
        return True

    def _log(self, symbol: str, action: str, detail: str, order_id: str = "") -> None:
        self.actions.insert(0, {"at": time.strftime("%Y-%m-%d %H:%M"),
                                "symbol": symbol, "action": action,
                                "detail": detail, "orderId": order_id})
        del self.actions[50:]
        self._append_history(symbol, action, detail)

    def _append_history(self, symbol: str, action: str, detail: str) -> None:
        """A durable record, so a forward test survives a restart.

        The whole value of this is a track record built over months; keeping it
        only in memory would mean every restart quietly began the evidence again.
        """
        path = os.path.join(self.config_dir(), "forward_test.json")
        try:
            os.makedirs(self.config_dir(), exist_ok=True)
            try:
                with open(path, encoding="utf-8") as fh:
                    rows = json.load(fh)
                    rows = rows if isinstance(rows, list) else []
            except (OSError, ValueError):
                rows = []
            rows.insert(0, {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "symbol": symbol, "action": action, "detail": detail})
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(rows[:500], fh, indent=1)
        except OSError:
            pass

    def status(self) -> dict:
        ok, reason = may_autotrade(self.state.settings)
        from . import playbook

        under_test = [r for r in playbook.load(self.config_dir()) if r.get("forward")]
        return {
            "running": self.running,
            "allowed": ok,
            "reason": reason,
            "mode": getattr(self.state.settings.mode, "value", ""),
            "underTest": under_test,
            "pollSeconds": POLL_SECONDS,
            "notionalPct": NOTIONAL_PCT,
            "lastRun": self.last_run,
            "lastError": self.last_error,
            "actions": self.actions[:20],
        }
