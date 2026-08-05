"""Textual TUI — the Markov 2.0 terminal dashboard.

Renders the honest (stride-sampled) transition matrix, the Fix-1 persistence
comparison with its warning, the Fix-2 label-verification badge, and the
Fix-3 strategy/target position, plus Alpaca account/positions when connected.
It shows the *target* position; it never places orders itself (that stays
behind the gated broker seam).
"""

from __future__ import annotations

from datetime import datetime

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, RichLog, Static

from .broker import make_broker
from .config import Mode, Settings
from .engine import SnapshotV2, analyze2
from .markov2 import Strategy
from .market_data import get_history, synthetic_close
from .regime import STATES

_STATE_STYLE = {0: "red", 1: "grey70", 2: "green"}  # Bear / Sideways / Bull


class MatrixPanel(Static):
    def update_snapshot(self, snap: SnapshotV2) -> None:
        P = snap.honest_matrix
        t = Table(title="Transition matrix — stride-sampled (honest)  P(next | now)",
                  expand=True, title_style="bold #84bba1")
        t.add_column("from ╲ to", justify="right", style="bold")
        for name in STATES:
            t.add_column(name, justify="right")
        for i, from_name in enumerate(STATES):
            cells = [Text(from_name, style=f"bold {_STATE_STYLE[i]}")]
            for j in range(3):
                pct = f"{P[i, j] * 100:5.1f}%"
                style = f"bold {_STATE_STYLE[i]} reverse" if i == j else "grey58"
                cells.append(Text(pct, style=style))
            t.add_row(*cells)
        self.update(t)


class Fix1Panel(Static):
    """Fix 1 — overlapping vs stride persistence, side by side, with warning."""

    def update_snapshot(self, snap: SnapshotV2) -> None:
        c = snap.comparison
        t = Table(title="Fix 1 · persistence diagonal", expand=True, title_style="bold")
        t.add_column("state", justify="right", style="bold")
        t.add_column("overlap", justify="right")
        t.add_column("stride", justify="right")
        t.add_column("faked", justify="right")
        for k, name in enumerate(STATES):
            t.add_row(
                Text(name, style=_STATE_STYLE[k]),
                Text(f"{c.stickiness_legacy[k]*100:4.0f}%", style="#c57f86"),
                Text(f"{c.stickiness_honest[k]*100:4.0f}%", style="#84bba1"),
                Text(f"+{c.inflation[k]:4.0f}pp", style="grey58"),
            )
        warn = Text(
            f"overlap counts {c.n_legacy} transitions sharing {c.window-1}/"
            f"{c.window} days → fake persistence. Only the {c.n_honest} "
            "stride-sampled ones are honest.",
            style="italic #c9a227",
        )
        self.update(Group(t, warn))


class SignalPanel(Static):
    def update_snapshot(self, snap: SnapshotV2) -> None:
        body = Text()
        body.append(f"{snap.ticker}", style="bold")
        body.append(f"   last {snap.last_price:,.2f}   ")
        body.append(f"{snap.n_rows} bars  {snap.start}→{snap.end}\n\n")

        body.append("current regime : ")
        body.append(f"{snap.current_state_name}\n", style=f"bold {_STATE_STYLE[snap.current_state]}")
        body.append("signal (honest): ")
        body.append(f"{snap.signal:+.3f}", style="bold")
        body.append("   P(Bull)−P(Bear)\n")
        body.append("strategy       : ")
        body.append(f"{snap.strategy.value.upper()}\n", style="bold #84bba1")
        body.append("target position: ")
        tstyle = "bold green" if snap.target_position > 0 else "bold red" if snap.target_position < 0 else "bold grey70"
        body.append(f"{snap.target_label}\n\n", style=tstyle)

        # Fix 2 — verification badge
        v = snap.verification
        if v.passed:
            body.append(" Fix 2 · labels verified ✓ \n", style="black on #84bba1")
        else:
            body.append(" Fix 2 · LABEL CHECK FAILED ✗ \n", style="bold white on red")
            for c in v.checks:
                if not c.ok:
                    body.append(f"   {c.description}: {c.detail}\n", style="red")
        self.update(body)


class AccountPanel(Static):
    def show_none(self, reason: str) -> None:
        self.update(Text(reason, style="grey58"))

    def update_account(self, account, position, mode: Mode) -> None:
        body = Text()
        badge = {Mode.DASHBOARD: ("DASHBOARD (read-only)", "black on grey70"),
                 Mode.PAPER: ("PAPER", "black on yellow"),
                 Mode.LIVE: ("LIVE — REAL MONEY", "bold white on red")}[mode]
        body.append(" " + badge[0] + " \n\n", style=badge[1])
        if account is not None:
            body.append(f"equity        {account.equity:,.2f}\n")
            body.append(f"cash          {account.cash:,.2f}\n")
            body.append(f"buying power  {account.buying_power:,.2f}\n\n")
        if position is not None:
            pl_style = "green" if position.unrealized_pl >= 0 else "red"
            body.append(f"position   {position.side} {position.qty:g}\n")
            body.append("unreal P&L  ")
            body.append(f"{position.unrealized_pl:+,.2f}\n", style=pl_style)
        else:
            body.append("position   flat\n", style="grey58")
        self.update(body)


class TerminalApp(App):
    CSS = """
    Screen { layout: vertical; }
    #top { height: 1fr; }
    #left { width: 3fr; }
    #right { width: 2fr; }
    MatrixPanel, Fix1Panel, SignalPanel, AccountPanel {
        border: round #555555; padding: 0 1; margin: 0 1;
    }
    Fix1Panel { height: 9; }
    SignalPanel { height: 13; }
    RichLog { height: 7; border: round #555555; margin: 0 1; }
    """
    BINDINGS = [
        ("r", "refresh", "Refresh now"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, settings: Settings, demo: bool = False):
        super().__init__()
        self.settings = settings
        self.demo = demo
        self.broker = None if demo else make_broker(settings)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top"):
            with Vertical(id="left"):
                yield MatrixPanel(id="matrix")
                yield Fix1Panel(id="fix1")
            with Vertical(id="right"):
                yield SignalPanel(id="signal")
                yield AccountPanel(id="account")
        yield RichLog(id="log", markup=True, highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "markov 2.0 — hedge fund terminal"
        self.sub_title = (f"{self.settings.ticker} · {self.settings.strategy.value} · "
                          f"{self.settings.mode.value}" + (" · DEMO" if self.demo else ""))
        self._log(f"[bold]starting[/] — strategy={self.settings.strategy.value} "
                  f"mode={self.settings.mode.value} {'(offline demo data)' if self.demo else ''}")
        if self.settings.mode is Mode.DASHBOARD:
            self._log("[grey70]dashboard: target position shown only, no orders will be placed.[/]")
        self.action_refresh()
        self.set_interval(self.settings.poll_seconds, self.action_refresh)

    def action_refresh(self) -> None:
        self._log(f"[grey58]{datetime.now():%H:%M:%S}[/] refreshing…")
        self.refresh_data()

    @work(exclusive=True, thread=True)
    def refresh_data(self) -> None:
        try:
            close = synthetic_close() if self.demo else get_history(self.settings)
            snap = analyze2(
                close, self.settings.ticker,
                window=self.settings.window, threshold=self.settings.threshold,
                strategy=self.settings.strategy, signal_threshold=self.settings.signal_threshold,
                size_cap=self.settings.size_cap, size_scale=self.settings.size_scale,
            )
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._log, f"[red]data error:[/] {exc}")
            return

        account = position = None
        if self.broker is not None:
            try:
                account = self.broker.get_account()
                position = self.broker.get_position(self.settings.ticker)
            except Exception as exc:  # noqa: BLE001
                self.call_from_thread(self._log, f"[yellow]broker read failed:[/] {exc}")

        self.call_from_thread(self._apply, snap, account, position)

    def _apply(self, snap: SnapshotV2, account, position) -> None:
        self.query_one(MatrixPanel).update_snapshot(snap)
        self.query_one(Fix1Panel).update_snapshot(snap)
        self.query_one(SignalPanel).update_snapshot(snap)
        acct = self.query_one(AccountPanel)
        if self.demo:
            acct.show_none("demo mode — no broker connected")
        elif self.broker is None:
            acct.show_none("no Alpaca credentials — data-only mode\n(set ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY)")
        else:
            acct.update_account(account, position, self.settings.mode)
        vflag = "labels ✓" if snap.verification.passed else "[red]labels ✗[/]"
        self._log(f"[green]updated[/] {snap.ticker}: {snap.current_state_name} → "
                  f"target {snap.target_label} ({vflag})")

    def _log(self, message: str) -> None:
        self.query_one(RichLog).write(message)
