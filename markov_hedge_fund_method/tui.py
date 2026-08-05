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
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    RichLog,
    Select,
    Static,
)

from .accounts import AccountStore, KeyringUnavailable
from .broker import ReadOnlyError, make_broker
from .config import Mode, Settings
from .engine import SnapshotV2, analyze2
from .markov2 import Strategy
from .market_data import get_history, synthetic_close
from .orders import (
    ORDER_CLASSES,
    ORDER_TYPES,
    SIDES,
    TIFS,
    OrderTicket,
    OrderValidationError,
)
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

    def update_account(self, account, position, mode: Mode, open_orders=None,
                       account_name=None, account_paper=None) -> None:
        body = Text()
        badge = {Mode.DASHBOARD: ("DASHBOARD (read-only)", "black on grey70"),
                 Mode.PAPER: ("PAPER", "black on yellow"),
                 Mode.LIVE: ("LIVE — REAL MONEY", "bold white on red")}[mode]
        body.append(" " + badge[0] + " \n", style=badge[1])
        if account_name:
            endpoint = "paper" if account_paper else "live"
            body.append("account  ", style="grey58")
            body.append(f"{account_name} ", style="bold #84bba1")
            body.append(f"[{endpoint}]  ", style="yellow" if account_paper else "bold red")
            body.append("press 'a' to switch\n", style="grey58")
        else:
            body.append("account  none · press 'a' to connect\n", style="grey58")
        body.append("\n")
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
        if open_orders:
            body.append(f"\nopen orders ({len(open_orders)}):\n", style="bold #c9a227")
            for o in open_orders[:5]:
                body.append(f"  {o.side} {o.qty} {o.symbol} {o.type} [{o.status}]\n",
                            style="grey70")
            if len(open_orders) > 5:
                body.append(f"  … +{len(open_orders) - 5} more\n", style="grey58")
        self.update(body)


class ExecutionPanel(Vertical):
    """Manual order entry covering every Alpaca order style.

    The widgets only *collect* a ticket; the App reads them, validates via
    `orders.build_order_request`, and submits through the gated broker seam.
    """

    def compose(self) -> ComposeResult:
        yield Static("⚡ EXECUTION — Alpaca order entry (market · limit · stop · "
                     "stop-limit · trailing · bracket/oco/oto)", classes="exec-title")
        with Horizontal(classes="exec-row"):
            yield Input(placeholder="symbol", id="exec_symbol")
            yield Select([(s.upper(), s) for s in SIDES], value="buy",
                         id="exec_side", allow_blank=False)
            yield Select([(t.replace("_", "-"), t) for t in ORDER_TYPES], value="market",
                         id="exec_type", allow_blank=False)
            yield Select([(t.upper(), t) for t in TIFS], value="day",
                         id="exec_tif", allow_blank=False)
            yield Select([(c.upper(), c) for c in ORDER_CLASSES], value="simple",
                         id="exec_class", allow_blank=False)
        with Horizontal(classes="exec-row"):
            yield Input(placeholder="qty (shares)", id="exec_qty", type="number")
            yield Input(placeholder="notional $", id="exec_notional", type="number")
            yield Input(placeholder="limit px", id="exec_limit", type="number")
            yield Input(placeholder="stop px", id="exec_stop", type="number")
            yield Input(placeholder="trail $", id="exec_trailp", type="number")
            yield Input(placeholder="trail %", id="exec_trailpct", type="number")
        with Horizontal(classes="exec-row"):
            yield Input(placeholder="take-profit", id="exec_tp", type="number")
            yield Input(placeholder="stop-loss stop", id="exec_slstop", type="number")
            yield Input(placeholder="stop-loss limit", id="exec_sllimit", type="number")
            yield Checkbox("ext hrs", id="exec_ext")
            yield Button("Submit", id="exec_submit", variant="success")
            yield Button("Cancel all", id="exec_cancel", variant="error")


class AccountsScreen(ModalScreen):
    """Manage multiple Alpaca account profiles and switch the active one.

    Reads/writes through the app's AccountStore; switching rebuilds the live
    broker so different portfolios can be traded from one terminal.
    """

    BINDINGS = [("escape", "close", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="acct-box"):
            yield Static("👤 ACCOUNTS — connect multiple portfolios", classes="acct-title")
            yield Static(id="acct-list")
            with Horizontal(classes="acct-row"):
                yield Select([], id="acct_pick", prompt="select account…")
                yield Button("Use", id="acct_use", variant="primary")
                yield Button("Remove", id="acct_remove", variant="error")
            yield Static("Add / update an account:", classes="acct-sub")
            with Horizontal(classes="acct-row"):
                yield Input(placeholder="name (e.g. swing)", id="acct_name")
                yield Input(placeholder="Alpaca API key id", id="acct_key")
                yield Input(placeholder="Alpaca API secret", id="acct_secret", password=True)
                yield Checkbox("paper", value=True, id="acct_paper")
                yield Button("Add", id="acct_add", variant="success")
            yield Button("Close", id="acct_close")

    def on_mount(self) -> None:
        self._refresh_list()

    # ── rendering ────────────────────────────────────────────────────────────
    def _refresh_list(self) -> None:
        store: AccountStore = self.app.accounts
        try:
            profiles = store.list()
        except Exception as exc:  # noqa: BLE001
            self.query_one("#acct-list", Static).update(Text(f"account store error: {exc}", style="red"))
            return
        body = Text()
        if not profiles:
            body.append("No accounts yet. Add one below.\n", style="grey58")
        for p in profiles:
            marker = "●" if p.active else " "
            tag, tstyle = ("paper", "yellow") if p.paper else ("LIVE", "bold red")
            line_style = "bold #84bba1" if p.active else "grey78"
            body.append(f" {marker} {p.name:<18s} ", style=line_style)
            body.append(f"[{tag}]\n", style=tstyle)
        self.query_one("#acct-list", Static).update(body)
        options = [(p.name, p.name) for p in profiles]
        pick = self.query_one("#acct_pick", Select)
        pick.set_options(options)

    def _selected(self) -> str | None:
        value = self.query_one("#acct_pick", Select).value
        return None if value is Select.BLANK else value

    # ── actions ──────────────────────────────────────────────────────────────
    def action_close(self) -> None:
        self.dismiss()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "acct_close":
            self.dismiss()
        elif bid == "acct_add":
            self._add()
        elif bid == "acct_use":
            self._use()
        elif bid == "acct_remove":
            self._remove()

    def _add(self) -> None:
        store: AccountStore = self.app.accounts
        name = self.query_one("#acct_name", Input).value.strip()
        key = self.query_one("#acct_key", Input).value.strip()
        secret = self.query_one("#acct_secret", Input).value.strip()
        paper = self.query_one("#acct_paper", Checkbox).value
        try:
            store.add(name, key, secret, paper=paper)
        except (ValueError, KeyringUnavailable) as exc:
            self.app._log(f"[yellow]account not saved:[/] {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            self.app._log(f"[red]account save failed:[/] {exc}")
            return
        for wid in ("acct_name", "acct_key", "acct_secret"):
            self.query_one(f"#{wid}", Input).value = ""
        self.app._log(f"[green]account saved[/] {name} ({'paper' if paper else 'LIVE'}) — now active.")
        self._refresh_list()
        self.app._apply_account_switch(name)

    def _use(self) -> None:
        name = self._selected()
        if not name:
            self.app._log("[yellow]pick an account to use first.[/]")
            return
        self.app._apply_account_switch(name)
        self._refresh_list()

    def _remove(self) -> None:
        name = self._selected()
        if not name:
            self.app._log("[yellow]pick an account to remove first.[/]")
            return
        store: AccountStore = self.app.accounts
        try:
            store.remove(name)
        except Exception as exc:  # noqa: BLE001
            self.app._log(f"[red]remove failed:[/] {exc}")
            return
        self.app._log(f"[grey70]removed account[/] {name}.")
        self._refresh_list()
        # If the active account changed, reconnect to whatever's active now.
        self.app._apply_account_switch(store.active())


class TerminalApp(App):
    CSS = """
    Screen { layout: vertical; }
    #top { height: 1fr; min-height: 10; }
    #left { width: 3fr; }
    #right { width: 2fr; }
    MatrixPanel, Fix1Panel, SignalPanel, AccountPanel {
        border: round #555555; padding: 0 1; margin: 0 1;
    }
    Fix1Panel { height: 9; }
    SignalPanel { height: 13; }
    ExecutionPanel {
        height: auto; border: round #c9a227; margin: 0 1; padding: 0 1;
    }
    .exec-title { color: #c9a227; text-style: bold; height: 1; }
    .exec-row { height: 3; }
    .exec-row Input, .exec-row Select { width: 1fr; margin: 0 1 0 0; }
    .exec-row Button { width: auto; margin: 0 0 0 1; }
    .exec-row Checkbox { width: auto; height: 3; content-align: left middle; }
    RichLog { height: 6; border: round #555555; margin: 0 1; }
    AccountsScreen { align: center middle; }
    #acct-box {
        width: 90%; max-width: 120; height: auto; padding: 1 2;
        border: thick #84bba1; background: $panel;
    }
    .acct-title { color: #84bba1; text-style: bold; height: 1; }
    .acct-sub { color: grey; height: 1; margin: 1 0 0 0; }
    #acct-list { height: auto; min-height: 3; margin: 1 0; }
    .acct-row { height: 3; }
    .acct-row Input, .acct-row Select { width: 1fr; margin: 0 1 0 0; }
    .acct-row Button { width: auto; margin: 0 0 0 1; }
    .acct-row Checkbox { width: auto; height: 3; content-align: left middle; }
    """
    BINDINGS = [
        ("r", "refresh", "Refresh now"),
        ("s", "focus_submit", "Order entry"),
        ("x", "cancel_all", "Cancel all orders"),
        ("a", "accounts", "Accounts"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, settings: Settings, demo: bool = False):
        super().__init__()
        self.settings = settings
        self.demo = demo
        self.accounts = AccountStore()
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
        yield ExecutionPanel(id="exec")
        yield RichLog(id="log", markup=True, highlight=True)
        yield Footer()

    def _subtitle(self) -> str:
        acct = f" · acct:{self.settings.account}" if self.settings.account else ""
        return (f"{self.settings.ticker} · {self.settings.strategy.value} · "
                f"{self.settings.mode.value}{acct}" + (" · DEMO" if self.demo else ""))

    def on_mount(self) -> None:
        self.title = "Mamba Terminal — by Quincy Gininda"
        self.sub_title = self._subtitle()
        self._log(f"[bold]starting[/] — strategy={self.settings.strategy.value} "
                  f"mode={self.settings.mode.value} {'(offline demo data)' if self.demo else ''}")
        # Prefill the order-entry symbol with the tracked ticker.
        self.query_one("#exec_symbol", Input).value = self.settings.ticker
        if self.settings.mode is Mode.DASHBOARD:
            self._log("[grey70]dashboard: target position shown only, no orders will be placed.[/]")
            self._log("[grey58]order entry is read-only in DASHBOARD — restart with "
                      "--mode paper to arm the execution panel.[/]")
        elif self.demo:
            self._log("[grey58]demo mode: no broker connected — order entry is disabled.[/]")
        else:
            self._log(f"[#c9a227]execution armed[/] — {self.settings.mode.value.upper()} "
                      "orders will be sent to Alpaca. Press 's' to jump to order entry.")
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
        open_orders = None
        if self.broker is not None:
            try:
                account = self.broker.get_account()
                position = self.broker.get_position(self.settings.ticker)
                open_orders = self.broker.list_open_orders()
            except Exception as exc:  # noqa: BLE001
                self.call_from_thread(self._log, f"[yellow]broker read failed:[/] {exc}")

        self.call_from_thread(self._apply, snap, account, position, open_orders)

    def _apply(self, snap: SnapshotV2, account, position, open_orders=None) -> None:
        self.query_one(MatrixPanel).update_snapshot(snap)
        self.query_one(Fix1Panel).update_snapshot(snap)
        self.query_one(SignalPanel).update_snapshot(snap)
        acct = self.query_one(AccountPanel)
        if self.demo:
            acct.show_none("demo mode — no broker connected")
        elif self.broker is None:
            acct.show_none("no Alpaca credentials — data-only mode\n(set ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY)")
        else:
            acct.update_account(account, position, self.settings.mode, open_orders,
                                account_name=self.settings.account,
                                account_paper=self.settings.account_paper)
        vflag = "labels ✓" if snap.verification.passed else "[red]labels ✗[/]"
        self._log(f"[green]updated[/] {snap.ticker}: {snap.current_state_name} → "
                  f"target {snap.target_label} ({vflag})")

    # ── execution panel ─────────────────────────────────────────────────────
    def action_focus_submit(self) -> None:
        self.query_one("#exec_symbol", Input).focus()

    def action_cancel_all(self) -> None:
        self._cancel_all_orders()

    # ── multi-account ────────────────────────────────────────────────────────
    def action_accounts(self) -> None:
        self.push_screen(AccountsScreen())

    def _apply_account_switch(self, name: str | None) -> None:
        """Switch the active Alpaca profile and rebuild the live broker."""
        if not name:
            # No accounts left — drop to data-only unless legacy keys exist.
            self.settings.account = None
            self.settings.account_paper = None
            self.settings.api_key = self.settings.api_secret = None
            self.broker = None
            self.sub_title = self._subtitle()
            self._log("[grey58]no active account — data-only mode.[/]")
            return
        try:
            self.accounts.set_active(name)
            resolved = self.accounts.resolve(name)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[red]could not switch account:[/] {exc}")
            return
        if resolved is None:
            self._log(f"[yellow]account {name} has no stored keys.[/]")
            return
        self.settings.api_key = resolved.key_id
        self.settings.api_secret = resolved.secret
        self.settings.account = resolved.name
        self.settings.account_paper = resolved.paper
        self.broker = None if self.demo else make_broker(self.settings)
        self.sub_title = self._subtitle()
        endpoint = "paper" if resolved.paper else "LIVE"
        self._log(f"[bold #84bba1]connected account[/] {resolved.name} ({endpoint} keys). "
                  "Reads use these credentials; orders still obey --mode.")
        self.action_refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "exec_submit":
            self._submit_from_form()
        elif event.button.id == "exec_cancel":
            self._cancel_all_orders()

    def _execution_blocked(self) -> bool:
        """Log a reason and return True when orders can't be placed right now."""
        if self.demo or self.broker is None:
            self._log("[yellow]order entry needs Alpaca credentials — this is data/demo mode.[/]")
            return True
        if not self.settings.can_trade:
            self._log("[yellow]DASHBOARD is read-only — restart with --mode paper to place orders.[/]")
            return True
        return False

    def _read_ticket(self) -> OrderTicket:
        def num(widget_id: str) -> float | None:
            raw = self.query_one(f"#{widget_id}", Input).value.strip()
            return float(raw) if raw else None

        symbol = self.query_one("#exec_symbol", Input).value.strip() or self.settings.ticker
        return OrderTicket(
            symbol=symbol,
            side=self.query_one("#exec_side", Select).value,
            order_type=self.query_one("#exec_type", Select).value,
            time_in_force=self.query_one("#exec_tif", Select).value,
            order_class=self.query_one("#exec_class", Select).value,
            qty=num("exec_qty"),
            notional=num("exec_notional"),
            limit_price=num("exec_limit"),
            stop_price=num("exec_stop"),
            trail_price=num("exec_trailp"),
            trail_percent=num("exec_trailpct"),
            take_profit_limit=num("exec_tp"),
            stop_loss_stop=num("exec_slstop"),
            stop_loss_limit=num("exec_sllimit"),
            extended_hours=self.query_one("#exec_ext", Checkbox).value,
        )

    def _submit_from_form(self) -> None:
        if self._execution_blocked():
            return
        try:
            ticket = self._read_ticket()
        except ValueError:
            self._log("[red]could not parse a numeric field — check the price/qty inputs.[/]")
            return
        self._log(f"[grey58]submitting[/] {ticket.symbol} {ticket.side} {ticket.order_type}…")
        self._do_submit(ticket)

    @work(exclusive=False, thread=True)
    def _do_submit(self, ticket: OrderTicket) -> None:
        try:
            result = self.broker.submit_ticket(ticket)
        except OrderValidationError as exc:
            self.call_from_thread(self._log, f"[yellow]invalid order:[/] {exc}")
            return
        except ReadOnlyError as exc:
            self.call_from_thread(self._log, f"[yellow]{exc}[/]")
            return
        except Exception as exc:  # noqa: BLE001 — Alpaca rejection (buying power, hours, …)
            self.call_from_thread(self._log, f"[red]order rejected:[/] {exc}")
            return
        self.call_from_thread(
            self._log,
            f"[bold green]order sent[/] {result.summary} → id {result.id} [{result.status}]",
        )
        self.call_from_thread(self.action_refresh)

    def _cancel_all_orders(self) -> None:
        if self._execution_blocked():
            return
        self._do_cancel()

    @work(exclusive=False, thread=True)
    def _do_cancel(self) -> None:
        try:
            n = self.broker.cancel_all_orders()
        except ReadOnlyError as exc:
            self.call_from_thread(self._log, f"[yellow]{exc}[/]")
            return
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._log, f"[red]cancel failed:[/] {exc}")
            return
        self.call_from_thread(self._log, f"[green]cancel requested[/] for {n} open order(s).")
        self.call_from_thread(self.action_refresh)

    def _log(self, message: str) -> None:
        self.query_one(RichLog).write(message)
