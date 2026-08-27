"""TPO / Market Profile — Steidlmayer's distribution of time over price.

A candlestick answers "where did price go?". A profile answers a different and
often more useful question: "where did price spend its time?" Time spent at a
level is the market accepting it as fair; time refused is rejection. That single
change of axis is the whole idea, and everything below follows from it.

The session is cut into half-hour periods, each given a letter — A for the first
half hour, B for the second, and so on. Every price level a period traded
through receives that period's letter. Stack the letters horizontally and the
session's time distribution appears, usually as a rough bell.

What the standard vocabulary means, and what is computed here:

  * **POC** (point of control) — the price with the most TPOs, the fairest price
    of the session by the market's own vote.

  * **Value area** — the band holding 70% of the session's TPOs, built outward
    from the POC. Its edges are VAH and VAL. This is not a percentile of the
    range; it is grown a row at a time from the middle, which is why a lopsided
    profile produces a lopsided value area.

  * **Initial balance** — the range of the first two periods. The opening hour
    is where the day's early conviction shows, and everything after is measured
    against it.

  * **Range extension** — trade beyond the initial balance. It says one side
    took control after the opening auction rather than during it.

  * **Single prints** — levels touched by exactly one period. Inside the profile
    they mark a gap the market moved through without pausing; at the extremes
    they are tails, where an auction was rejected outright.

  * **Shape** — a bell means an auction that found agreement; an elongated
    profile means a trend that never did; two bulges mean the session held two
    separate arguments about value.

Pure functions over an intraday OHLC frame, so all of it is testable without a
chart or a network.
"""

from __future__ import annotations

import string

import numpy as np
import pandas as pd

LETTERS = string.ascii_uppercase + string.ascii_lowercase
DEFAULT_PERIOD_MIN = 30
DEFAULT_ROWS = 48
VALUE_AREA_PCT = 0.70
IB_PERIODS = 2                    # the opening hour, at 30-minute periods


def tick_size(low: float, high: float, rows: int = DEFAULT_ROWS) -> float:
    """Row height for the profile.

    Derived from the session's own range rather than fixed, because a $3 stock
    and a $600 one need different granularity and a hard-coded tick makes one of
    them unreadable. Rounded to a friendly increment so the price labels do not
    come out as arbitrary decimals.
    """
    span = float(high) - float(low)
    if not np.isfinite(span) or span <= 0:
        return 0.01
    raw = span / max(int(rows), 4)
    # Snap to 1/2/2.5/5 x a power of ten — the increments a trader reads easily.
    exp = np.floor(np.log10(raw))
    base = 10.0 ** exp
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= mult * base:
            return float(mult * base)
    return float(10.0 * base)


def _period_key(index: pd.DatetimeIndex, minutes: int) -> pd.DatetimeIndex:
    """Bucket timestamps into fixed periods — the letters of the profile."""
    return pd.DatetimeIndex(index).floor(f"{int(minutes)}min")


def build_profile(ohlc: pd.DataFrame, *, period_minutes: int = DEFAULT_PERIOD_MIN,
                  rows: int = DEFAULT_ROWS, value_pct: float = VALUE_AREA_PCT,
                  tick: float | None = None, base_low: float | None = None,
                  base_high: float | None = None) -> dict:
    """Turn intraday bars into a market profile.

    Bars finer than the period are grouped into periods; each period marks every
    row between its own high and low. A period that trades a single price still
    marks one row, so a quiet half hour is not silently dropped.
    """
    empty = {"rows": [], "letters": [], "poc": None, "vah": None, "val": None,
             "ibHigh": None, "ibLow": None, "high": None, "low": None,
             "open": None, "close": None, "tickSize": None, "totalTpo": 0,
             "singlePrints": [], "shape": "", "rangeExtension": "none",
             "periods": 0, "valuePct": value_pct}
    if ohlc is None or ohlc.empty or not {"High", "Low", "Close"} <= set(ohlc.columns):
        return empty

    df = ohlc.dropna(subset=["High", "Low", "Close"]).copy()
    if df.empty:
        return empty
    df.index = pd.to_datetime(df.index)

    lo = float(df["Low"].min())
    hi = float(df["High"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + max(abs(lo) * 0.001, 0.01)

    # A caller drawing several sessions side by side passes one grid for all of
    # them. Without that each column would size its own rows, so a given height
    # would mean a different price in every column and the layout would be
    # unreadable — which is the entire reason to lay them out together.
    if base_low is not None and base_high is not None and base_high > base_low:
        lo, hi = float(base_low), float(base_high)
    if tick is None or not np.isfinite(tick) or tick <= 0:
        tick = tick_size(lo, hi, rows)
    base = np.floor(lo / tick) * tick
    n_rows = int(np.ceil((hi - base) / tick)) + 1
    n_rows = max(1, min(n_rows, 400))          # a profile is read, not scrolled
    prices = base + np.arange(n_rows) * tick

    # Group the bars into periods and letter them in time order.
    groups = df.groupby(_period_key(df.index, period_minutes), sort=True)
    letters_used: list[str] = []
    row_letters: list[list[str]] = [[] for _ in range(n_rows)]
    ib_hi = ib_lo = None

    for i, (_, chunk) in enumerate(groups):
        # Past the alphabet the letters wrap with a prime rather than repeating
        # the last one. Fifty-two half-hours is more than any real session, so
        # this only fires on a misuse — but a run of identical letters silently
        # merges distinct periods, which is the kind of wrong that still looks
        # like a profile.
        letter = LETTERS[i] if i < len(LETTERS) else LETTERS[i % len(LETTERS)] + "'"
        letters_used.append(letter)
        p_hi, p_lo = float(chunk["High"].max()), float(chunk["Low"].min())
        top = int(np.floor((p_hi - base) / tick))
        bot = int(np.floor((p_lo - base) / tick))
        bot, top = max(0, min(bot, n_rows - 1)), max(0, min(top, n_rows - 1))
        for r in range(min(bot, top), max(bot, top) + 1):
            row_letters[r].append(letter)
        if i < IB_PERIODS:
            ib_hi = p_hi if ib_hi is None else max(ib_hi, p_hi)
            ib_lo = p_lo if ib_lo is None else min(ib_lo, p_lo)

    counts = np.array([len(x) for x in row_letters], dtype=int)
    total = int(counts.sum())
    if total == 0:
        return empty

    poc_idx = pick_poc(counts)
    val_idx, vah_idx = value_area(counts, poc_idx, value_pct)

    singles = [round(float(prices[i]), 4)
               for i in range(n_rows) if counts[i] == 1]

    out_rows = [{"price": round(float(prices[i]), 4),
                 "letters": "".join(row_letters[i]),
                 "count": int(counts[i])}
                for i in range(n_rows)]

    return {
        "rows": out_rows,
        "letters": letters_used,
        "periods": len(letters_used),
        "tickSize": round(float(tick), 6),
        "poc": round(float(prices[poc_idx]), 4),
        "vah": round(float(prices[vah_idx]), 4),
        "val": round(float(prices[val_idx]), 4),
        "ibHigh": None if ib_hi is None else round(float(ib_hi), 4),
        "ibLow": None if ib_lo is None else round(float(ib_lo), 4),
        "high": round(float(hi), 4),
        "low": round(float(lo), 4),
        "open": round(float(df["Open"].iloc[0]), 4) if "Open" in df.columns else None,
        "close": round(float(df["Close"].iloc[-1]), 4),
        "totalTpo": total,
        "singlePrints": singles,
        "shape": profile_shape(counts, len(letters_used)),
        "rangeExtension": range_extension(hi, lo, ib_hi, ib_lo),
        "valuePct": value_pct,
    }


def pick_poc(counts: np.ndarray) -> int:
    """The row with the most TPOs; ties broken toward the middle of the range.

    Ties are common — a balanced session often has several levels touched by
    every period. Taking the first match makes the POC the *lowest* of them,
    which is arbitrary and reads as a downward bias that is not in the data. The
    convention is the level nearest the centre of the profile, so that is used.
    """
    counts = np.asarray(counts, dtype=int)
    if not len(counts):
        return 0
    best = int(counts.max())
    tied = np.flatnonzero(counts == best)
    centre = (len(counts) - 1) / 2.0
    return int(tied[np.argmin(np.abs(tied - centre))])


def value_area(counts: np.ndarray, poc_idx: int, pct: float = VALUE_AREA_PCT) -> tuple[int, int]:
    """Grow the value area outward from the POC until it holds `pct` of TPOs.

    The standard construction, and worth doing properly: at each step compare
    the two rows above the current band against the two below, and take whichever
    pair holds more. Taking a flat percentile of the range instead would centre
    the value area on the midpoint rather than on where time was actually spent,
    which is the one thing a profile exists to show.
    """
    counts = np.asarray(counts, dtype=int)
    n = len(counts)
    if n == 0:
        return 0, 0
    target = counts.sum() * float(pct)
    lo = hi = int(poc_idx)
    inside = int(counts[poc_idx])

    while inside < target and (lo > 0 or hi < n - 1):
        up1 = counts[hi + 1] if hi + 1 < n else -1
        up2 = counts[hi + 2] if hi + 2 < n else 0
        dn1 = counts[lo - 1] if lo - 1 >= 0 else -1
        dn2 = counts[lo - 2] if lo - 2 >= 0 else 0
        up = (up1 + up2) if up1 >= 0 else -1
        dn = (dn1 + dn2) if dn1 >= 0 else -1
        if up < 0 and dn < 0:
            break
        # The pair is compared, but added a row at a time and stopped as soon as
        # the target is met — taking the whole pair regardless overshoots 70% by
        # more than it needs to on a profile with wide rows.
        if up >= dn:
            for _ in range(2):
                if hi + 1 < n and inside < target:
                    hi += 1
                    inside += int(counts[hi])
        else:
            for _ in range(2):
                if lo - 1 >= 0 and inside < target:
                    lo -= 1
                    inside += int(counts[lo])
    return lo, hi


def profile_shape(counts: np.ndarray, periods: int = 0) -> str:
    """A one-word read of the session's structure.

    Judged by how much of the session the busiest level actually held. In a
    balanced auction most periods return to the same price, so the peak
    approaches the period count; in a trend the market keeps leaving, and no
    level collects many letters however long the session runs. Comparing the
    peak against the mean instead — the first thing tried here — called a clean
    bell a trend, because a bell has a high mean too.

    Deliberately coarse: profile shape is a qualitative read, and a
    precise-looking number would lend it authority it has not earned.
    """
    counts = np.asarray(counts, dtype=int)
    live = counts[counts > 0]
    if len(live) < 4:
        return "thin"
    peak = int(live.max())
    periods = int(periods) or peak
    if periods <= 0:
        return "thin"
    concentration = peak / periods          # 1.0 = every period revisited the POC

    # Concentration is judged first, and that ordering matters. A one-way
    # session never settles anywhere, so its profile is thin the whole way up
    # and *every* point looks like a trough between two bulges — testing for
    # bimodality before ruling out a trend labelled a clean trend day a double
    # distribution. A market that never paused cannot have paused twice.
    if concentration <= 0.4:
        return "trend"

    # Two genuine bulges with a real trough between them: the session held two
    # separate arguments about value.
    mid = len(counts) // 2
    lower, upper = counts[:mid], counts[mid:]
    if len(lower) and len(upper):
        lo_peak, up_peak = int(lower.max()), int(upper.max())
        trough = int(counts[max(0, mid - 2): mid + 2].min()) if len(counts) > 4 else peak
        if (lo_peak >= peak * 0.7 and up_peak >= peak * 0.7
                and trough <= peak * 0.45):
            return "double distribution"

    return "normal" if concentration >= 0.6 else "developing"


def range_extension(high, low, ib_high, ib_low) -> str:
    """Which side, if any, pushed beyond the opening hour."""
    if ib_high is None or ib_low is None:
        return "none"
    up = float(high) > float(ib_high) + 1e-9
    down = float(low) < float(ib_low) - 1e-9
    if up and down:
        return "both"
    if up:
        return "up"
    if down:
        return "down"
    return "none"


def open_type(profile: dict) -> str:
    """How the session opened, relative to where it settled.

    The four classic readings, kept plain: whether the open was accepted where
    it happened or rejected and driven away from.
    """
    o, poc = profile.get("open"), profile.get("poc")
    vah, val = profile.get("vah"), profile.get("val")
    if o is None or poc is None or vah is None or val is None:
        return ""
    if val <= o <= vah:
        return "open-auction — opened inside value, no early conviction"
    if o > vah:
        return "open-drive lower — opened above value and was sold back into it"
    return "open-drive higher — opened below value and was bought back into it"


def summarise(profile: dict) -> str:
    """One line a trader can read without decoding the vocabulary."""
    poc, vah, val = profile.get("poc"), profile.get("vah"), profile.get("val")
    if poc is None:
        return "No intraday data to build a profile from."
    shape = profile.get("shape", "")
    ext = profile.get("rangeExtension", "none")
    close = profile.get("close")
    where = ("inside value" if val <= close <= vah
             else "above value" if close > vah else "below value")
    ext_txt = {"none": "held the opening hour's range",
               "up": "extended above the opening hour",
               "down": "extended below the opening hour",
               "both": "extended both sides of the opening hour"}[ext]
    return (f"Fairest price {poc:g}, value {val:g}–{vah:g}. "
            f"Closed {where}; {ext_txt}. Shape: {shape}.")


# ── one profile per session, which is what a TPO chart actually is ──────────
# A real market profile is a row of daily distributions, each restarting its
# lettering at A, laid out left to right. Merging a month into one shape is not
# a coarser version of that — it is a different and much less useful object,
# because the whole point is watching value migrate from one day to the next.
MAX_SESSIONS = 40          # beyond this the columns are too narrow to read


def session_keys(index: pd.DatetimeIndex) -> np.ndarray:
    """Which trading day each bar belongs to."""
    return pd.DatetimeIndex(index).normalize().to_numpy()


def build_sessions(ohlc: pd.DataFrame, *, period_minutes: int = DEFAULT_PERIOD_MIN,
                   rows: int = DEFAULT_ROWS, value_pct: float = VALUE_AREA_PCT,
                   max_sessions: int = MAX_SESSIONS) -> dict:
    """A profile per trading day, plus the composite over all of them.

    Each day is built independently, so its letters start at A and its point of
    control and value area describe that day alone. The composite is the merged
    shape — still worth having, but as one column beside the others rather than
    as the only thing on screen.

    The most recent sessions are kept when there are more than will fit. A
    profile is read, and forty narrow columns is already past the point where
    another one adds anything.
    """
    out = {"sessions": [], "composite": build_profile(
        ohlc, period_minutes=period_minutes, rows=rows, value_pct=value_pct),
        "truncated": 0, "requested": 0}
    if ohlc is None or ohlc.empty or not {"High", "Low", "Close"} <= set(ohlc.columns):
        return out

    df = ohlc.dropna(subset=["High", "Low", "Close"]).copy()
    if df.empty:
        return out
    df.index = pd.to_datetime(df.index)

    keys = session_keys(df.index)
    unique = list(pd.unique(keys))
    out["requested"] = len(unique)
    if len(unique) > max_sessions:
        out["truncated"] = len(unique) - max_sessions
        unique = unique[-max_sessions:]

    # Every session is measured on the same price grid, or the columns cannot be
    # read against each other — a row at one height would mean a different price
    # in each column, which defeats the entire layout.
    lo = float(df["Low"].min())
    hi = float(df["High"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + max(abs(lo) * 0.001, 0.01)
    tick = tick_size(lo, hi, rows)

    sessions = []
    for key in unique:
        part = df[keys == key]
        if part.empty:
            continue
        prof = build_profile(part, period_minutes=period_minutes, rows=rows,
                             value_pct=value_pct, tick=tick, base_low=lo,
                             base_high=hi)
        prof["date"] = str(pd.Timestamp(key).date())
        sessions.append(prof)

    out["sessions"] = sessions
    out["tickSize"] = tick
    out["gridLow"] = lo
    out["gridHigh"] = hi
    # The line a profile reader actually follows: where value sat each day.
    # Rising points of control are acceptance moving up, whatever any single
    # day's candle did.
    out["pocLine"] = [{"date": s["date"], "poc": s["poc"]} for s in sessions]
    return out
