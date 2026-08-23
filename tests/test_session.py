"""Market sessions.

Checked against the published NYSE calendar rather than against the code's own
logic, because the value of this module is entirely in matching reality. The
awkward cases each get a test: a Saturday New Year that is *not* observed, Good
Friday tracking Easter, and the three half-days.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from markov_hedge_fund_method.session import (
    describe,
    early_close,
    easter,
    holidays,
    is_half_day,
    is_open,
    is_trading_day,
    minutes_into_session,
    minutes_to_close,
    next_open,
    phase,
    session_bounds,
)


def d(s: str) -> date:
    return date.fromisoformat(s)


def t(s: str) -> datetime:
    return datetime.fromisoformat(s)


# ── the published calendar ──────────────────────────────────────────────────
NYSE_2025 = {"2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
             "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25"}
NYSE_2026 = {"2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
             "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"}


@pytest.mark.parametrize("year,expected", [(2025, NYSE_2025), (2026, NYSE_2026)])
def test_the_holiday_list_matches_the_exchange(year, expected):
    assert {str(x) for x in holidays(year)} == expected


def test_good_friday_tracks_easter():
    """Not a federal holiday, but the exchange closes for it, and it moves."""
    assert easter(2025) == d("2025-04-20")
    assert easter(2026) == d("2026-04-05")
    assert d("2025-04-18") in holidays(2025)     # two days before Easter
    assert d("2026-04-03") in holidays(2026)


def test_a_saturday_new_year_is_not_observed():
    """Most holidays shift to the nearest weekday. This one does not — the
    market traded normally on Friday 31 December 2021."""
    assert not any(x.month == 1 for x in holidays(2022) if x.day == 1)
    assert is_trading_day(d("2021-12-31"))


def test_weekend_holidays_shift_to_a_weekday():
    assert d("2027-07-05") in holidays(2027)     # the 4th is a Sunday
    assert d("2027-12-24") in holidays(2027)     # Christmas is a Saturday
    assert d("2022-12-26") in holidays(2022)     # Christmas is a Sunday


def test_juneteenth_starts_in_2022():
    assert not any(x.month == 6 for x in holidays(2021))
    assert d("2022-06-20") in holidays(2022)     # the 19th was a Sunday


# ── half-days ───────────────────────────────────────────────────────────────
def test_the_three_half_days():
    assert early_close(d("2025-11-28")) is not None     # day after Thanksgiving
    assert early_close(d("2025-07-03")) is not None     # the 4th is a weekday
    assert early_close(d("2025-12-24")) is not None     # Christmas Eve, a Wednesday


def test_a_normal_day_is_not_a_half_day():
    assert early_close(d("2026-08-24")) is None


def test_july_third_is_not_a_half_day_when_it_is_the_holiday():
    """When the 4th falls on a Saturday the exchange closes the Friday instead,
    so the 3rd is a full holiday and asking for its early close is meaningless."""
    assert not is_trading_day(d("2026-07-03"))
    assert early_close(d("2026-07-03")) is None


def test_a_half_day_closes_at_one():
    bounds = session_bounds(d("2025-11-28"))
    assert bounds[1].hour == 13 and bounds[1].minute == 0


def test_a_half_day_session_is_shorter():
    half = session_bounds(d("2025-11-28"))
    full = session_bounds(d("2026-08-24"))
    assert (half[1] - half[0]) < (full[1] - full[0])


# ── phases ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("when,expected", [
    ("2026-08-24 03:00", "closed"),
    ("2026-08-24 07:30", "premarket"),
    ("2026-08-24 09:35", "opening"),
    ("2026-08-24 10:30", "morning"),
    ("2026-08-24 12:30", "midday"),
    ("2026-08-24 14:30", "afternoon"),
    ("2026-08-24 15:45", "closing"),
    ("2026-08-24 17:00", "afterhours"),
    ("2026-08-24 21:00", "closed"),
])
def test_the_day_walks_through_its_phases(when, expected):
    assert phase(t(when)) == expected


def test_the_weekend_is_closed():
    assert phase(t("2026-08-22 11:00")) == "closed"
    assert phase(t("2026-08-23 11:00")) == "closed"


def test_a_holiday_is_closed_at_midday():
    assert phase(t("2026-11-26 11:00")) == "closed"     # Thanksgiving


def test_a_half_day_has_no_lunch_lull():
    """Naming a midday lull in a session that ends at one o'clock would put the
    label on the part of the day that is actually the run into the close."""
    assert phase(t("2026-11-27 11:00")) == "morning"
    assert phase(t("2026-11-27 12:45")) == "closing"


def test_is_open_excludes_the_extended_sessions():
    assert is_open(t("2026-08-24 10:30"))
    assert not is_open(t("2026-08-24 07:30"))   # pre-market is not open
    assert not is_open(t("2026-08-24 17:00"))   # nor is after hours


# ── clock arithmetic ────────────────────────────────────────────────────────
def test_minutes_into_the_session():
    assert minutes_into_session(t("2026-08-24 09:30")) == pytest.approx(0)
    assert minutes_into_session(t("2026-08-24 10:30")) == pytest.approx(60)


def test_minutes_to_the_close_respects_a_half_day():
    assert minutes_to_close(t("2026-08-24 15:00")) == pytest.approx(60)
    assert minutes_to_close(t("2026-11-27 12:00")) == pytest.approx(60)


def test_the_clock_is_undefined_outside_regular_hours():
    assert minutes_into_session(t("2026-08-24 07:30")) is None
    assert minutes_to_close(t("2026-08-22 11:00")) is None


def test_next_open_skips_the_weekend_and_the_holiday():
    assert next_open(t("2026-08-21 17:00")).date() == d("2026-08-24")   # Fri -> Mon
    assert next_open(t("2026-11-25 17:00")).date() == d("2026-11-27")   # Thanksgiving


def test_next_open_returns_today_before_the_bell():
    assert next_open(t("2026-08-24 07:00")).date() == d("2026-08-24")


def test_is_half_day_reads_the_moment():
    assert is_half_day(t("2026-11-27 11:00"))
    assert not is_half_day(t("2026-08-24 11:00"))


# ── the readable line ───────────────────────────────────────────────────────
def test_describe_says_when_it_reopens_while_shut():
    assert "opens" in describe(t("2026-08-22 11:00"))


def test_describe_counts_down_while_open():
    assert "min to the close" in describe(t("2026-08-24 14:30"))


def test_describe_flags_a_half_day():
    assert "half-day" in describe(t("2026-11-27 11:00"))


# ── the last completed session ──────────────────────────────────────────────
@pytest.mark.parametrize("when,expected", [
    ("2026-08-21 17:00", "2026-08-21"),   # Friday, after the close
    ("2026-08-21 11:00", "2026-08-20"),   # Friday, mid-session — today is unfinished
    ("2026-08-22 11:00", "2026-08-21"),   # Saturday
    ("2026-08-23 11:00", "2026-08-21"),   # Sunday
    ("2026-08-24 11:00", "2026-08-21"),   # Monday, mid-session
    ("2026-08-24 17:00", "2026-08-24"),   # Monday, after the close
    ("2026-11-26 11:00", "2026-11-25"),   # Thanksgiving
    ("2026-11-27 14:00", "2026-11-27"),   # after a half-day's one o'clock close
    ("2026-11-27 12:00", "2026-11-25"),   # during the half-day, still unfinished
])
def test_the_last_completed_session(when, expected):
    from markov_hedge_fund_method.session import last_completed_session
    assert str(last_completed_session(t(when))) == expected


def test_the_weekend_does_not_make_fridays_data_stale():
    """The bug this exists to stop: comparing cached bars against yesterday's
    calendar date marks every symbol stale from Friday's close until Monday's,
    because Friday's bar is older than Saturday. That re-downloaded the whole
    universe every weekend — the slow full sweep, on a loop."""
    from datetime import timedelta

    from markov_hedge_fund_method.session import last_completed_session
    friday = date.fromisoformat("2026-08-21")
    for when in ("2026-08-21 17:00", "2026-08-22 09:00", "2026-08-23 20:00",
                 "2026-08-24 09:00"):
        cutoff = last_completed_session(t(when))
        assert friday >= cutoff, f"Friday's bar judged stale at {when}"
    # And the naive cutoff really was wrong — from Sunday onward it sits past
    # Friday's bar. Saturday is the one day it accidentally gets right.
    for when in ("2026-08-23 20:00", "2026-08-24 09:00"):
        assert friday < t(when).date() - timedelta(days=1)
