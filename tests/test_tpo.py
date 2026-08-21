"""TPO / Market Profile.

The vocabulary here has fixed meanings, so most of these tests are arithmetic
against a profile small enough to check by hand. The two that matter most are
the ones catching mistakes that still *look* plausible on screen: a POC picked
arbitrarily from tied rows, and a value area taken as a percentile of the range
rather than grown outward from where time was actually spent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.tpo import (
    build_profile,
    open_type,
    pick_poc,
    profile_shape,
    range_extension,
    summarise,
    tick_size,
    value_area,
)
from markov_hedge_fund_method.web import AppState, create_app


def session(highs, lows, start="2026-08-21 09:30", freq="30min"):
    """One bar per period, so periods map one-to-one onto letters."""
    idx = pd.date_range(start, periods=len(highs), freq=freq)
    return pd.DataFrame({"Open": lows, "High": highs, "Low": lows, "Close": highs},
                        index=idx)


BELL = session([101, 102, 103, 102, 101], [99, 98, 97, 98, 99])


# ── row height ──────────────────────────────────────────────────────────────
def test_tick_size_scales_with_the_price():
    """A fixed tick makes either a $3 stock or a $600 one unreadable."""
    assert tick_size(3.0, 3.4, 40) < tick_size(100.0, 104.0, 40)
    assert tick_size(100.0, 104.0, 40) < tick_size(600.0, 640.0, 40)


def test_tick_size_snaps_to_readable_increments():
    for lo, hi in ((3.0, 3.4), (100.0, 104.0), (600.0, 640.0)):
        tick = tick_size(lo, hi, 40)
        mantissa = tick / 10 ** np.floor(np.log10(tick))
        assert mantissa in (1.0, 2.0, 2.5, 5.0), f"{tick} is not a round increment"


def test_a_flat_session_still_gets_a_tick():
    assert tick_size(50.0, 50.0) > 0


# ── the profile ─────────────────────────────────────────────────────────────
def test_each_period_becomes_a_letter():
    p = build_profile(BELL, rows=12)
    assert p["periods"] == 5
    assert "".join(p["letters"]) == "ABCDE"


def test_every_row_a_period_traded_gets_its_letter():
    p = build_profile(BELL, rows=12)
    # C is the widest period (97-103), so it must appear on every row.
    assert all("C" in r["letters"] for r in p["rows"])
    # A is the narrowest (99-101) and must not reach the extremes.
    assert "A" not in p["rows"][0]["letters"]
    assert "A" not in p["rows"][-1]["letters"]


def test_the_poc_is_the_row_with_the_most_time():
    p = build_profile(BELL, rows=12)
    counts = {r["price"]: r["count"] for r in p["rows"]}
    assert counts[p["poc"]] == max(counts.values())


def test_tied_rows_put_the_poc_in_the_middle():
    """A balanced session ties several rows. Taking the first match makes the
    POC the lowest of them, which reads as a downward bias that is not there."""
    assert pick_poc(np.array([1, 5, 5, 5, 1])) == 2
    assert pick_poc(np.array([5, 5, 1, 1, 1])) == 1
    assert pick_poc(np.array([])) == 0


def test_the_bell_puts_its_poc_at_the_centre():
    p = build_profile(BELL, rows=12)
    assert p["poc"] == pytest.approx(100.0, abs=0.6)


# ── value area ──────────────────────────────────────────────────────────────
def test_the_value_area_holds_about_the_requested_share():
    p = build_profile(BELL, rows=12)
    inside = sum(r["count"] for r in p["rows"] if p["val"] <= r["price"] <= p["vah"])
    frac = inside / p["totalTpo"]
    assert 0.70 <= frac <= 0.85, f"value area holds {frac:.0%}"


def test_the_value_area_grows_toward_the_denser_side():
    """Not a percentile of the range: it is grown from the POC, so a lopsided
    profile must produce a lopsided value area."""
    counts = np.array([1, 1, 9, 5, 1, 1])       # weight sits above the POC
    lo, hi = value_area(counts, 2, 0.70)
    assert lo == 2 and hi >= 3, "it grew downward into the thin side"


def test_the_value_area_contains_the_poc():
    for pct in (0.30, 0.60, 0.70, 0.95):
        p = build_profile(BELL, rows=12, value_pct=pct)
        assert p["val"] <= p["poc"] <= p["vah"]


def test_a_wider_setting_gives_a_wider_value_area():
    narrow = build_profile(BELL, rows=20, value_pct=0.50)
    wide = build_profile(BELL, rows=20, value_pct=0.90)
    assert (wide["vah"] - wide["val"]) >= (narrow["vah"] - narrow["val"])


def test_the_value_area_never_escapes_the_session_range():
    p = build_profile(BELL, rows=12, value_pct=0.95)
    assert p["low"] <= p["val"] and p["vah"] <= p["high"] + p["tickSize"]


# ── initial balance and range extension ─────────────────────────────────────
def test_the_initial_balance_is_the_first_two_periods():
    p = build_profile(BELL, rows=12)
    assert p["ibHigh"] == 102.0 and p["ibLow"] == 98.0


def test_range_extension_names_the_side_that_pushed():
    assert range_extension(110, 98, 102, 98) == "up"
    assert range_extension(102, 90, 102, 98) == "down"
    assert range_extension(110, 90, 102, 98) == "both"
    assert range_extension(101, 99, 102, 98) == "none"
    assert range_extension(110, 90, None, None) == "none"


def test_a_breakout_after_the_opening_hour_reads_as_extension():
    p = build_profile(session([101, 102, 108, 109, 110], [99, 98, 100, 101, 105]), rows=20)
    assert p["rangeExtension"] == "up"


# ── shape ───────────────────────────────────────────────────────────────────
def test_a_balanced_session_reads_as_normal():
    """Every period returning to the same price is the definition of balance."""
    p = build_profile(BELL, rows=12)
    assert p["shape"] == "normal"


def test_a_one_way_session_reads_as_trend():
    """Price never comes back, so no level collects many letters."""
    highs = [101, 103, 105, 107, 109, 111, 113, 115]
    lows = [100, 102, 104, 106, 108, 110, 112, 114]
    p = build_profile(session(highs, lows), rows=40)
    assert p["shape"] == "trend"


def test_shape_needs_enough_rows_to_judge():
    assert profile_shape(np.array([3, 3]), 3) == "thin"


def test_the_shape_test_would_have_caught_the_first_version():
    """The first heuristic compared the peak against the mean and called a clean
    bell a trend, because a bell has a high mean too."""
    counts = np.array([1, 1, 3, 3, 5, 5, 5, 5, 5, 3, 3, 1, 1])
    assert profile_shape(counts, periods=5) == "normal"


# ── single prints ───────────────────────────────────────────────────────────
def test_single_prints_are_levels_only_one_period_touched():
    p = build_profile(BELL, rows=12)
    singles = set(p["singlePrints"])
    for r in p["rows"]:
        assert (r["price"] in singles) == (r["count"] == 1)


# ── degenerate input ────────────────────────────────────────────────────────
def test_no_bars_gives_an_empty_profile_not_a_crash():
    p = build_profile(pd.DataFrame())
    assert p["poc"] is None and p["rows"] == [] and p["totalTpo"] == 0


def test_a_single_bar_still_profiles():
    p = build_profile(session([100], [100]))
    assert p["poc"] is not None and p["periods"] == 1


def test_a_completely_flat_session_does_not_divide_by_zero():
    p = build_profile(session([50] * 4, [50] * 4))
    assert p["poc"] == pytest.approx(50.0, abs=0.1)


def test_finer_bars_are_grouped_into_periods():
    """Five-minute bars must produce half-hour letters, not thirty of them."""
    n = 36                                    # three hours of 5-minute bars
    df = session([100 + i * 0.1 for i in range(n)],
                 [99 + i * 0.1 for i in range(n)], freq="5min")
    p = build_profile(df, period_minutes=30)
    assert p["periods"] == 6


# ── readings ────────────────────────────────────────────────────────────────
def test_the_summary_names_the_levels():
    p = build_profile(BELL, rows=12)
    text = summarise(p)
    assert "Fairest price" in text and "value" in text.lower()


def test_the_summary_survives_an_empty_profile():
    assert "No intraday data" in summarise(build_profile(pd.DataFrame()))


def test_open_type_is_blank_without_a_profile():
    assert open_type(build_profile(pd.DataFrame())) == ""


# ── the API ─────────────────────────────────────────────────────────────────
def _client():
    return TestClient(create_app(AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)))


def test_the_endpoint_returns_a_complete_profile():
    d = _client().get("/api/tpo", params={"symbol": "SPY"}).json()
    p = d["profile"]
    for key in ("rows", "letters", "poc", "vah", "val", "ibHigh", "ibLow",
                "singlePrints", "shape", "rangeExtension", "totalTpo"):
        assert key in p
    assert p["rows"] and p["poc"] is not None


def test_the_endpoint_reports_its_data_source():
    """A profile on placeholder bars must never look like one on real trade."""
    d = _client().get("/api/tpo", params={"symbol": "SPY"}).json()
    assert d["real"] is False
    assert "does not describe the market" in d["summary"]
    assert d["openType"] == "", "a reading of fabricated bars is worse than none"


def test_the_value_area_setting_reaches_the_profile():
    c = _client()
    narrow = c.get("/api/tpo", params={"symbol": "SPY", "value": 0.5}).json()["profile"]
    wide = c.get("/api/tpo", params={"symbol": "SPY", "value": 0.9}).json()["profile"]
    assert (wide["vah"] - wide["val"]) > (narrow["vah"] - narrow["val"])


def test_the_value_setting_is_bounded():
    c = _client()
    for v in (-5, 0.0, 9.9):
        p = c.get("/api/tpo", params={"symbol": "SPY", "value": v}).json()["profile"]
        assert p["val"] <= p["poc"] <= p["vah"]


def test_the_period_setting_changes_the_letter_count():
    c = _client()
    coarse = c.get("/api/tpo", params={"symbol": "SPY", "period": 60}).json()["profile"]
    fine = c.get("/api/tpo", params={"symbol": "SPY", "period": 5}).json()["profile"]
    assert fine["periods"] > coarse["periods"]


# ── the UI ──────────────────────────────────────────────────────────────────
def test_the_page_offers_a_tpo_mode():
    html = _client().get("/").text
    assert "toggleTPO" in html and 'id="tpo-toggle"' in html


def test_letters_are_coloured_by_period():
    """Warm at the open through to cool at the close, so the profile shows when
    a level traded as well as how long."""
    html = _client().get("/").text
    assert "function tpoColour" in html and "hsl(" in html


def test_the_three_levels_are_marked_not_implied():
    html = _client().get("/").text
    for tag in ("POC", "VAH", "VAL"):
        assert f"'{tag}'" in html or f'>{tag}<' in html
    assert ".tpo-tag" in html


def test_the_value_area_is_adjustable_from_the_page():
    html = _client().get("/").text
    assert "setTpoValue" in html and "setTpoPeriod" in html
    assert "value area" in html


def test_a_trend_is_not_mistaken_for_two_distributions():
    """A one-way session is thin the whole way up, so every point looks like a
    trough between two bulges. Testing bimodality before ruling out a trend
    labelled a clean trend day a double distribution — a market that never
    paused cannot have paused twice."""
    highs = [101, 103, 105, 107, 109, 111, 113, 115]
    lows = [100, 102, 104, 106, 108, 110, 112, 114]
    assert build_profile(session(highs, lows), rows=40)["shape"] == "trend"


def test_two_real_bulges_still_read_as_a_double_distribution():
    """Half the session at one price, half at another, with a fast move between."""
    highs = [101, 101, 101, 110, 110, 110]
    lows = [100, 100, 100, 109, 109, 109]
    p = build_profile(session(highs, lows), rows=40)
    assert p["shape"] == "double distribution"
