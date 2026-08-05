"""Textual TUI — the terminal dashboard.

Renders the live Markov snapshot (transition matrix, stationary mix, current
regime, signal, desired position) plus the Alpaca account/position when
credentials are present. It never places orders itself: it displays the
*desired* position and leaves execution to the (gated) broker seam. A future
paper-trading executor would call `broker.set_target_position` from here.
"""

from __future__ import annotations

from datetime import datetime

from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, RichLog, Static

from .broker import make_broker
from .config import Mode, Settings
from .engine import Snapshot, analyze
from .market_data import get_history, synthetic_close
from .regime import STATES

_STATE_STYLE = {0: "red", 1: "grey70", 2: "green"}  # Bear / Sideways / Bull


class MatrixPanel(Static):
    def update_snapshot(self, snap: Snapshot) -> None:
        t = Table(title="Transition matrix  P(tomorrow | today)", expand=True, title_style="bold")
        t.add_column("from ╲ to", justify="right", style="bold")
        for name in STATES:
            t.add_column(name, justify="right")
        for i, from_name in enumerate(STATES):
            cells = [Text(from_name, style=f"bold {_STATE_STYLE[i]}")]
            for j in range(3):
                pct = f"{snap.matrix[i, j] * 100:5.1f}%"
                if i == j:  # persistence diagonal — the load-bearing reveal
                    cells.append(Text(pct, style=f"bold {_STATE_STYLE[i]} reverse"))
                else:
                    cells.append(Text(pct, style="grey58"))
            t.add_row(*cells)
        self.update(t)


class StationaryPanel(Static):
    def update_snapshot(self, snap: Snapshot) -> None:
        t = Table(title="Long-run mix  (stationary π)", expand=True, title_style="bold")
        for name in STATES:
            t.add_column(name, justify="center")
        t.add_row(*[
            Text(f"{snap.stationary[k] * 100:4.1f}%", style=f"bold {_STATE_STYLE[k]}")
            for k in range(3)
        ])
        self.update(t)


class SignalPanel(Static):
    def update_snapshot(self, snap: Snapshot) -> None:
        pos_style = {1: "bold green", 0: "bold grey70", -1: "bold red"}[snap.desired_position]
        body = Text()
        body.append(f"{snap.ticker}", style="bold")
        body.append(f"   last {snap.last_price:,.2f}\n")
        body.append(f"{snap.n_rows} bars   {snap.start} → {snap.end}\n\n")
        body.append("current regime : ")
        body.append(f"{snap.current_state_name}\n", style=f"bold {_STATE_STYLE[snap.current_state]}")
        body.append("signal (Bull−Bear): ")
        body.append(f"{snap.signal:+.3f}\n", style="bold")
        body.append("desired position : ")
        body.append(f"{snap.desired_label}", style=pos_style)
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
            body.append(f"buying power  {account.buying_power:,.2f}\n")
            body.append(f"status        {account.status}\n\n")
        if position is not None:
            pl_style = "green" if position.unrealized_pl >= 0 else "red"
            body.append(f"position   {position.side} {position.qty:g}\n")
            body.append("unreal. P&L  ")
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
    MatrixPanel, StationaryPanel, SignalPanel, AccountPanel {
        border: round #555555; padding: 0 1; margin: 0 1;
    }
    StationaryPanel { height: 6; }
    SignalPanel { height: 11; }
    RichLog { height: 8; border: round #555555; margin: 0 1; }
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
                yield StationaryPanel(id="stationary")
            with Vertical(id="right"):
                yield SignalPanel(id="signal")
                yield AccountPanel(id="account")
        yield RichLog(id="log", markup=True, highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "markov-hedge-fund terminal"
        self.sub_title = f"{self.settings.ticker} · {self.settings.mode.value}" + (" · DEMO" if self.demo else "")
        self._log(f"[bold]starting[/] — mode={self.settings.mode.value} "
                  f"{'(offline demo data)' if self.demo else ''}")
        if self.settings.mode is Mode.DASHBOARD:
            self._log("[grey70]dashboard mode: showing desired position only, no orders will be placed.[/]")
        self.action_refresh()
        self.set_interval(self.settings.poll_seconds, self.action_refresh)

    def action_refresh(self) -> None:
        self._log(f"[grey58]{datetime.now():%H:%M:%S}[/] refreshing…")
        self.refresh_data()

    @work(exclusive=True, thread=True)
    def refresh_data(self) -> None:
        try:
            close = synthetic_close() if self.demo else get_history(self.settings)
            snap = analyze(close, self.settings.ticker, self.settings.window, self.settings.threshold)
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

    def _apply(self, snap: Snapshot, account, position) -> None:
        self.query_one(MatrixPanel).update_snapshot(snap)
        self.query_one(StationaryPanel).update_snapshot(snap)
        self.query_one(SignalPanel).update_snapshot(snap)
        acct = self.query_one(AccountPanel)
        if self.demo:
            acct.show_none("demo mode — no broker connected")
        elif self.broker is None:
            acct.show_none("no Alpaca credentials — data-only mode\n(set ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY)")
        else:
            acct.update_account(account, position, self.settings.mode)
        self._log(f"[green]updated[/] {snap.ticker}: {snap.current_state_name} → desired {snap.desired_label}")

    def _log(self, message: str) -> None:
        self.query_one(RichLog).write(message)
