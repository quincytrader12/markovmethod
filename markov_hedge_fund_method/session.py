"""US equity market sessions — when the market is open, and where in the day you are.

Every intraday reading depends on this. A VWAP is anchored to the session open;
an opening range is the first thirty minutes of it; a TPO period is a slice of
it. Get the boundary wrong and every one of those is quietly measuring the
wrong window.

Two things make this more than a pair of clock times.

Holidays move. Most shift to the nearest weekday when they fall on one, but not
uniformly — a Saturday New Year's Day is simply not observed, so the market
trades the Friday before. Good Friday is not a federal holiday at all, yet the
NYSE closes for it, and it tracks Easter rather than a fixed date.

Half-days exist and they are not obvious. The market closes at 13:00 the day
after Thanksgiving, and around Independence Day and Christmas depending on which
weekday those land on. A half-day session that the terminal thinks runs to 16:00
produces an opening range and a value area computed over an hour of nothing.

No exchange-calendar dependency: the rules are stable, short, and worth being
able to read.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

OPEN = time(9, 30)
CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
PREMARKET_OPEN = time(4, 0)
AFTERHOURS_CLOSE = time(20, 0)

OPENING_MINUTES = 30        # the auction and the noise around it
CLOSING_MINUTES = 30        # the run into the close

# Phases in the order they occur, so a caller can compare positions.
PHASES = ("closed", "premarket", "opening", "morning", "midday",
          "afternoon", "closing", "afterhours")


# ── holidays ────────────────────────────────────────────────────────────────
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth given weekday of a month (weekday: Monday=0)."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last given weekday of a month."""
    d = date(year, month, 28)
    while (d + timedelta(days=7)).month == month:
        d += timedelta(days=7)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def easter(year: int) -> date:
    """Gregorian Easter (Meeus/Jones/Butcher). Good Friday hangs off this."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = ((h + ll - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date) -> date | None:
    """Weekend holidays shift to the nearest weekday.

    Saturday moves back to Friday, Sunday forward to Monday. The caller handles
    New Year's Day, which is the exception.
    """
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def holidays(year: int) -> set[date]:
    """Days the NYSE is closed in a given year."""
    out: set[date] = set()

    # A Saturday New Year's Day is not observed on the Friday — that Friday is
    # the last trading day of the previous year and the market is open.
    ny = date(year, 1, 1)
    if ny.weekday() != 5:
        out.add(_observed(ny))

    out.add(_nth_weekday(year, 1, 0, 3))            # Martin Luther King Jr Day
    out.add(_nth_weekday(year, 2, 0, 3))            # Washington's Birthday
    out.add(easter(year) - timedelta(days=2))       # Good Friday
    out.add(_last_weekday(year, 5, 0))              # Memorial Day
    if year >= 2022:                                # first observed by the NYSE in 2022
        out.add(_observed(date(year, 6, 19)))       # Juneteenth
    out.add(_observed(date(year, 7, 4)))            # Independence Day
    out.add(_nth_weekday(year, 9, 0, 1))            # Labor Day
    out.add(_nth_weekday(year, 11, 3, 4))           # Thanksgiving
    out.add(_observed(date(year, 12, 25)))          # Christmas
    return out


def is_holiday(d: date) -> bool:
    return d in holidays(d.year)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and not is_holiday(d)


def early_close(d: date) -> time | None:
    """The 13:00 close, when the day is a half-day. None otherwise."""
    if not is_trading_day(d):
        return None

    # The Friday after Thanksgiving.
    if d.month == 11 and d.weekday() == 4 and d == _nth_weekday(d.year, 11, 3, 4) + timedelta(days=1):
        return EARLY_CLOSE

    # July 3, when the 4th is itself a weekday. If the 4th falls at a weekend
    # the holiday is observed adjacent to it and there is no half-day.
    if d.month == 7 and d.day == 3 and date(d.year, 7, 4).weekday() < 5:
        return EARLY_CLOSE

    # Christmas Eve, when it lands Monday-Thursday. A Friday the 24th means
    # Christmas is a Saturday, and the 24th is the observed holiday instead.
    if d.month == 12 and d.day == 24 and d.weekday() <= 3:
        return EARLY_CLOSE

    return None


# ── session bounds ──────────────────────────────────────────────────────────
def _as_eastern(ts=None) -> datetime:
    """Anything time-like, in New York. Naive input is read as Eastern."""
    if ts is None:
        return datetime.now(EASTERN)
    if hasattr(ts, "to_pydatetime"):        # pandas Timestamp
        ts = ts.to_pydatetime()
    if not isinstance(ts, datetime):        # a plain date
        ts = datetime(ts.year, ts.month, ts.day)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=EASTERN)
    return ts.astimezone(EASTERN)


def session_bounds(d) -> tuple[datetime, datetime] | None:
    """Open and close for a date, Eastern. None when the market does not trade."""
    day = _as_eastern(d).date()
    if not is_trading_day(day):
        return None
    close = early_close(day) or CLOSE
    return (datetime.combine(day, OPEN, EASTERN),
            datetime.combine(day, close, EASTERN))


def is_open(ts=None) -> bool:
    """True during regular hours only — pre- and post-market are not open."""
    now = _as_eastern(ts)
    bounds = session_bounds(now)
    return bounds is not None and bounds[0] <= now < bounds[1]


def is_half_day(ts=None) -> bool:
    return early_close(_as_eastern(ts).date()) is not None


def phase(ts=None) -> str:
    """Where in the day you are. One of PHASES."""
    now = _as_eastern(ts)
    day = now.date()
    if not is_trading_day(day):
        return "closed"

    open_dt, close_dt = session_bounds(now)
    if now < datetime.combine(day, PREMARKET_OPEN, EASTERN):
        return "closed"
    if now < open_dt:
        return "premarket"
    if now >= datetime.combine(day, AFTERHOURS_CLOSE, EASTERN):
        return "closed"
    if now >= close_dt:
        return "afterhours"

    since = (now - open_dt).total_seconds() / 60.0
    until = (close_dt - now).total_seconds() / 60.0
    if since < OPENING_MINUTES:
        return "opening"
    if until <= CLOSING_MINUTES:
        return "closing"

    # A half-day has no lunch lull worth naming — it is over before midday
    # behaviour sets in, so the middle of it is simply "morning".
    total = (close_dt - open_dt).total_seconds() / 60.0
    if total < 300:
        return "morning"
    if since < 120:
        return "morning"
    if since < 270:
        return "midday"
    return "afternoon"


def minutes_into_session(ts=None) -> float | None:
    """Minutes since the open, or None when the market is not in regular hours."""
    now = _as_eastern(ts)
    bounds = session_bounds(now)
    if bounds is None or not (bounds[0] <= now < bounds[1]):
        return None
    return (now - bounds[0]).total_seconds() / 60.0


def minutes_to_close(ts=None) -> float | None:
    now = _as_eastern(ts)
    bounds = session_bounds(now)
    if bounds is None or not (bounds[0] <= now < bounds[1]):
        return None
    return (bounds[1] - now).total_seconds() / 60.0


def next_open(ts=None) -> datetime:
    """The next regular open at or after the given moment."""
    now = _as_eastern(ts)
    day = now.date()
    for _ in range(15):                     # no holiday run is anywhere near this long
        bounds = session_bounds(day)
        if bounds is not None and now < bounds[0]:
            return bounds[0]
        day += timedelta(days=1)
        now = datetime.combine(day, time(0, 0), EASTERN)
    raise RuntimeError("no trading day found within two weeks")


def last_completed_session(ts=None) -> date:
    """The most recent trading day whose close has already passed.

    This is the honest answer to "what is the newest daily bar that can exist",
    and it is the only correct thing to measure cached data against. Comparing
    against yesterday's calendar date instead marks every stored symbol stale
    from Friday's close until Monday's, because Friday's bar is older than
    Saturday — which quietly re-downloads the entire universe every weekend.
    """
    now = _as_eastern(ts)
    day = now.date()
    bounds = session_bounds(day)
    if bounds is None or now < bounds[1]:
        day -= timedelta(days=1)        # today has not finished, or is not a session
    for _ in range(15):
        if is_trading_day(day):
            return day
        day -= timedelta(days=1)
    raise RuntimeError("no trading day found within two weeks")


def describe(ts=None) -> str:
    """One line a person can read."""
    now = _as_eastern(ts)
    p = phase(now)
    if p == "closed":
        nxt = next_open(now)
        return f"Market closed — opens {nxt:%a %d %b %H:%M} ET"
    if p == "premarket":
        return f"Pre-market — regular session opens {session_bounds(now)[0]:%H:%M} ET"
    if p == "afterhours":
        return "After hours — regular session has closed"
    left = minutes_to_close(now)
    half = " (half-day)" if is_half_day(now) else ""
    label = {"opening": "Opening range", "morning": "Morning", "midday": "Midday lull",
             "afternoon": "Afternoon", "closing": "Into the close"}[p]
    return f"{label}{half} — {left:.0f} min to the close"
