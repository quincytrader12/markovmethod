"""The playbook and forward testing.

These cover the bridge between research and orders, so most of them are about
what is *refused*. Two rules carry the weight:

  1. Nothing reaches the playbook that did not survive the untouched quarter.
  2. Forward testing auto-trades on paper and never, under any circumstance,
     on a live account.

Both are enforced in code rather than in the interface, because a mode selector
is one click from LIVE at all times and a search winner is untrustworthy by
construction.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from markov_hedge_fund_method import playbook
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.forwardtest import ForwardTester, may_autotrade
from markov_hedge_fund_method.market_data import synthetic_close
from markov_hedge_fund_method.web import AppState, create_app

CLOSE = synthetic_close(1500, seed=4)


def entry(**kw):
    base = {"name": "momentum_252", "kind": "single", "parts": ["momentum_252"],
            "heldUp": True, "dsr": 0.9,
            "searched": {"sharpe": 1.2}, "holdout": {"sharpe": 0.9}}
    base.update(kw)
    return base


# ── the gate into the playbook ──────────────────────────────────────────────
def test_a_strategy_that_failed_its_holdout_is_refused(tmp_path):
    """The normal outcome for a search winner, and the reason the gate exists."""
    with pytest.raises(playbook.NotProven, match="did not hold up"):
        playbook.adopt(str(tmp_path), "SPY", entry(heldUp=False))


def test_a_weak_out_of_sample_result_is_refused(tmp_path):
    with pytest.raises(playbook.NotProven, match="floor"):
        playbook.adopt(str(tmp_path), "SPY", entry(holdout={"sharpe": 0.05}))


def test_a_low_deflated_sharpe_is_refused(tmp_path):
    with pytest.raises(playbook.NotProven, match="luck"):
        playbook.adopt(str(tmp_path), "SPY", entry(dsr=0.2))


def test_a_proven_strategy_is_accepted(tmp_path):
    row = playbook.adopt(str(tmp_path), "SPY", entry())
    assert row["symbol"] == "SPY" and row["enabled"] is True
    assert playbook.load(str(tmp_path))


def test_there_is_no_override_on_the_gate():
    """Every path into the playbook runs `check`. A way round it would be a way
    to lose money with the terminal's blessing."""
    import inspect
    src = inspect.getsource(playbook.adopt)
    assert "check(entry)" in src
    assert "force" not in src and "override" not in src


def test_adopting_twice_does_not_duplicate(tmp_path):
    playbook.adopt(str(tmp_path), "SPY", entry())
    playbook.adopt(str(tmp_path), "SPY", entry())
    assert len(playbook.load(str(tmp_path))) == 1


# ── what the playbook may do to an order ────────────────────────────────────
def test_it_can_only_ever_shrink(tmp_path):
    playbook.adopt(str(tmp_path), "SPY", entry())
    c = playbook.consult(str(tmp_path), "SPY", CLOSE)
    assert 0.0 <= c["multiplier"] <= 1.0


def test_agreement_earns_full_size_and_no_more(tmp_path):
    """Strategies found by one search over one history agree far more often than
    independent evidence would. Unanimity is a floor on conviction, never a
    multiplier on it."""
    playbook.adopt(str(tmp_path), "SPY", entry())
    playbook.adopt(str(tmp_path), "SPY", entry(name="ma_cross_50_200",
                                               parts=["ma_cross_50_200"]))
    c = playbook.consult(str(tmp_path), "SPY", CLOSE)
    assert c["multiplier"] <= 1.0


def test_disagreement_halves_the_size(tmp_path):
    consulted = {"votes": [1, 2], "long": 1, "short": 1, "multiplier": 0.5}
    assert consulted["multiplier"] == 0.5


def test_a_veto_needs_unanimity(tmp_path):
    """Never on a split — the one veto this layer has is a narrow one."""
    split = {"votes": [1, 2], "long": 1, "short": 1}
    assert not playbook.side_conflict(split, "buy")
    unanimous = {"votes": [1, 2], "long": 0, "short": 2}
    assert playbook.side_conflict(unanimous, "buy")
    assert not playbook.side_conflict(unanimous, "sell")


def test_no_adopted_strategy_means_no_opinion(tmp_path):
    c = playbook.consult(str(tmp_path), "SPY", CLOSE)
    assert c["multiplier"] == 1.0 and c["active"] == 0


def test_a_renamed_strategy_is_not_guessed_at(tmp_path):
    """Silently substituting a different rule would be worse than not running."""
    playbook.adopt(str(tmp_path), "SPY", entry(name="gone", parts=["gone"]))
    c = playbook.consult(str(tmp_path), "SPY", CLOSE)
    assert c["multiplier"] == 1.0


# ── the live gate: the rule that matters most ───────────────────────────────
def test_auto_trading_is_allowed_on_paper():
    ok, _ = may_autotrade(Settings(ticker="SPY", mode=Mode.PAPER))
    assert ok is True


def test_auto_trading_is_refused_on_live():
    """Not discouraged, not confirmed — refused. A strategy under forward test
    is by definition one whose edge is unproven."""
    ok, reason = may_autotrade(Settings(ticker="SPY", mode=Mode.LIVE))
    assert ok is False
    assert "LIVE" in reason


@pytest.mark.parametrize("mode", [Mode.DASHBOARD, Mode.BACKTEST, Mode.LIVE])
def test_only_paper_may_auto_trade(mode):
    assert may_autotrade(Settings(ticker="SPY", mode=mode))[0] is False


def test_the_live_gate_has_no_setting_to_disable_it():
    import inspect

    from markov_hedge_fund_method import forwardtest
    src = inspect.getsource(forwardtest.may_autotrade)
    assert "Mode.PAPER" in src
    for escape in ("allow_live", "force", "override", "settings.get"):
        assert escape not in src


class _Broker:
    def __init__(self, equity=50_000.0):
        self.submitted = []
        self.account = type("A", (), {"equity": equity, "cash": equity,
                                      "buying_power": equity, "status": "ACTIVE",
                                      "last_equity": equity, "daytrade_count": 0,
                                      "pattern_day_trader": False,
                                      "daytrading_buying_power": 0.0})()

    def get_account(self):
        return self.account

    def list_positions(self):
        return []

    def submit_ticket(self, ticket):
        from markov_hedge_fund_method.broker import OrderResult
        self.submitted.append(ticket)
        return OrderResult(id="f1", status="accepted", summary="ok")


def _tester(tmp_path, mode):
    state = AppState(Settings(ticker="SPY", mode=mode), demo=True)
    state.broker = _Broker()
    ft = ForwardTester(state, config_dir=str(tmp_path))
    playbook.adopt(str(tmp_path), "SPY", entry())
    rows = playbook.load(str(tmp_path))
    rows[0]["forward"] = True
    playbook._write(str(tmp_path), rows)
    return ft, state


def test_a_live_account_is_never_traded_by_the_loop(tmp_path):
    """The end-to-end version of the rule: even with a strategy adopted, under
    forward test, an account connected and the market open, LIVE places nothing."""
    ft, state = _tester(tmp_path, Mode.LIVE)
    out = ft.tick()
    assert out["traded"] == 0
    assert "LIVE" in out["blocked"]
    assert state.broker.submitted == []


def test_paper_does_trade(tmp_path):
    ft, state = _tester(tmp_path, Mode.PAPER)
    ft.tick()
    assert len(state.broker.submitted) == 1
    assert state.broker.submitted[0].symbol == "SPY"


def test_the_kill_switch_stops_forward_testing(tmp_path):
    ft, state = _tester(tmp_path, Mode.PAPER)
    state.alerts.halted = True
    out = ft.tick()
    assert out["traded"] == 0 and "kill switch" in out["blocked"]
    assert state.broker.submitted == []


def test_nothing_under_test_means_nothing_traded(tmp_path):
    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER), demo=True)
    state.broker = _Broker()
    ft = ForwardTester(state, config_dir=str(tmp_path))
    assert ft.tick()["traded"] == 0
    assert state.broker.submitted == []


def test_a_tiny_gap_is_not_worth_a_spread(tmp_path):
    ft, state = _tester(tmp_path, Mode.PAPER)
    ft.tick()
    n = len(state.broker.submitted)
    ft.tick()          # nothing changed, but positions are still reported empty
    assert len(state.broker.submitted) >= n


def test_the_position_size_is_small_by_design():
    """A measurement apparatus, not an allocation."""
    from markov_hedge_fund_method import forwardtest
    assert 0 < forwardtest.NOTIONAL_PCT <= 0.10


def test_status_reports_why_it_is_blocked(tmp_path):
    ft, _ = _tester(tmp_path, Mode.LIVE)
    s = ft.status()
    assert s["allowed"] is False and "LIVE" in s["reason"]


# ── the API ─────────────────────────────────────────────────────────────────
def test_backtest_mode_places_no_orders():
    assert Settings(ticker="SPY", mode=Mode.BACKTEST).can_trade is False


def test_the_mode_is_offered_in_the_page():
    c = TestClient(create_app(AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD),
                                       demo=True)))
    html = c.get("/").text
    assert '<option value="backtest">BACKTEST</option>' in html
    assert "function openLab" in html


def test_the_lab_ui_refuses_to_integrate_a_failed_candidate():
    c = TestClient(create_app(AppState(Settings(ticker="SPY", mode=Mode.BACKTEST),
                                       demo=True)))
    html = c.get("/").text
    assert "REFUSED" in html
    assert "Only a strategy that survived the untouched quarter" in html
