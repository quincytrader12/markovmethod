"""Entry timing for a multi-day hold.

This module does not decide *what* to buy or *whether* to buy. The scanner finds
the name and the daily regime decides whether a position is allowed. What is
missing between "this name is worth owning" and an order ticket is the question
of when, inside today, to put it on — and that is all this answers.

The distinction matters enough to state plainly, because getting it wrong is how
a terminal ends up with two opinions that argue with each other. **Everything
here is a measure of fill quality, not of edge.** A score of 90 does not mean the
trade will work; it means that if you were going to buy this today anyway, now
is a better moment than most. A score of 20 is not a bearish signal — it means
wait a couple of hours. Nothing in this file can veto the daily regime, and
nothing in it can authorise a trade the daily regime did not.

Four things go into it, and each is something a trader would actually look at:

VWAP is the day's volume-weighted average price — what the average share
traded at. Buying below it means paying less than the crowd did; buying well
above it means paying up. It is the single most-used intraday execution
reference, and it needs volume, which is why the data path had to start
carrying it.

The value area is where the session spent 70% of its time (see `tpo.py`). In a
name the daily chain already likes, the low edge of value is the dip; above the
high edge you are chasing.

The opening range is the first thirty minutes' high and low. How far price has
travelled from it says how extended the move already is.

The session phase decides whether to be doing this at all. The first thirty
minutes carry the widest spreads and the auction's noise; the lunch lull is
thin. Neither is where a patient buyer wants to be lifting offers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import session

# What each component is worth. VWAP and value area carry the most because they
# are about price paid; the rest are context.
WEIGHTS = {"vwap": 0.35, "value": 0.30, "range": 0.20, "phase": 0.15}

# Phases scored for how good a moment they are to work an order.
PHASE_SCORE = {"opening": 0.15, "morning": 1.0, "midday": 0.70,
               "afternoon": 0.95, "closing": 0.75}
TRADEABLE = set(PHASE_SCORE)

# Score thresholds. WAIT_CEILING also caps the opening, so a "wait" verdict can
# never carry a number that argues with it.
NOW_FLOOR = 70.0
FAIR_FLOOR = 50.0
WAIT_CEILING = 45.0


def _sessions(index: pd.DatetimeIndex) -> np.ndarray:
    """Calendar day for each bar — the reset boundary for a session VWAP."""
    return pd.DatetimeIndex(index).normalize().to_numpy()


def typical_price(df: pd.DataFrame) -> pd.Series:
    """(H + L + C) / 3 — the conventional VWAP input."""
    return (df["High"] + df["Low"] + df["Close"]) / 3.0


def vwap(df: pd.DataFrame) -> pd.Series | None:
    """Session VWAP, restarting each day. None when the feed sent no volume.

    Returning None rather than falling back to an unweighted average is
    deliberate: an average price labelled VWAP that no volume went into is a
    different quantity wearing the same name, and it would be read as the real
    thing.
    """
    if df is None or df.empty or "Volume" not in df.columns:
        return None
    vol = df["Volume"].astype(float)
    if vol.sum() <= 0:
        return None
    tp = typical_price(df)
    day = _sessions(df.index)
    out = pd.Series(index=df.index, dtype=float)
    for key in pd.unique(day):
        m = day == key
        v = vol[m]
        cum_v = v.cumsum()
        cum_pv = (tp[m] * v).cumsum()
        out[m] = np.where(cum_v > 0, cum_pv / cum_v.replace(0, np.nan), tp[m])
    return out


def anchored_vwap(df: pd.DataFrame, anchor) -> pd.Series | None:
    """VWAP measured from a chosen bar onward — an event, a low, a breakout.

    Unlike the session VWAP this does not reset daily; the whole point is to
    carry the average paid since the thing you anchored to.
    """
    if df is None or df.empty or "Volume" not in df.columns:
        return None
    anchor = pd.Timestamp(anchor)
    m = pd.DatetimeIndex(df.index) >= anchor
    if not m.any():
        return None
    sub = df[m]
    vol = sub["Volume"].astype(float)
    if vol.sum() <= 0:
        return None
    cum_v = vol.cumsum()
    cum_pv = (typical_price(sub) * vol).cumsum()
    return pd.Series(cum_pv / cum_v.replace(0, np.nan), index=sub.index)


def opening_range(df: pd.DataFrame, minutes: int = 30) -> dict | None:
    """High and low of the first N minutes of the most recent session."""
    if df is None or df.empty:
        return None
    idx = pd.DatetimeIndex(df.index)
    last_day = idx.normalize()[-1]
    todays = df[idx.normalize() == last_day]
    if todays.empty:
        return None
    start = pd.DatetimeIndex(todays.index)[0]
    window = todays[pd.DatetimeIndex(todays.index) < start + pd.Timedelta(minutes=minutes)]
    if window.empty:
        return None
    hi, lo = float(window["High"].max()), float(window["Low"].min())
    return {"high": hi, "low": lo, "mid": (hi + lo) / 2.0,
            "height": max(hi - lo, 0.0), "bars": int(len(window)),
            "complete": bool(len(todays) > len(window))}


def atr(df: pd.DataFrame, n: int = 14) -> float | None:
    """Average true range — the unit a stop should be measured in.

    True range counts the overnight gap, which a plain high-minus-low misses;
    on a multi-day hold the gap is exactly the risk that matters.
    """
    if df is None or len(df) < 2:
        return None
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr = tr.dropna()
    if tr.empty:
        return None
    return float(tr.tail(n).mean())


# ── the score ───────────────────────────────────────────────────────────────
def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _vwap_component(price: float, vw: float, spread: float, side: str) -> float:
    """How much better than the day's average price you are getting.

    Scaled by the session's own dispersion, so a quiet name and a volatile one
    are judged on the same footing.
    """
    if not (np.isfinite(vw) and np.isfinite(spread)) or spread <= 0:
        return 0.5
    z = (price - vw) / spread
    if side == "sell":
        z = -z
    return _clamp01(0.5 - z / 2.0)          # one dispersion cheaper -> 1.0


def _value_component(price: float, profile: dict | None, side: str) -> float:
    """Where price sits in the session's accepted value."""
    if not profile:
        return 0.5
    val, vah, poc = profile.get("val"), profile.get("vah"), profile.get("poc")
    if val is None or vah is None or poc is None:
        return 0.5
    if side == "sell":                       # mirror the ladder for a short
        price = poc - (price - poc)
    if price <= val:
        return 1.0
    if price <= poc:
        return 0.8
    if price <= vah:
        return 0.5
    return 0.15


def _range_component(price: float, orange: dict | None, side: str) -> float:
    """How extended price is from the opening range, in range-heights."""
    if not orange or orange["height"] <= 0:
        return 0.5
    ext = (price - orange["mid"]) / orange["height"]
    if side == "sell":
        ext = -ext
    return _clamp01(0.75 - ext / 2.0)        # half a range up costs a quarter


def entry_timing(df: pd.DataFrame, *, price: float | None = None,
                 profile: dict | None = None, side: str = "buy",
                 now=None) -> dict:
    """Score how good a moment this is to work an order, 0-100.

    `df` is intraday OHLC(V) for the session, `profile` an optional TPO profile
    from `tpo.build_profile`. Returns the score, the components behind it, and a
    plain-English reading — never a direction, because direction is the daily
    regime's job.
    """
    side = "sell" if str(side).lower() in ("sell", "short") else "buy"
    empty = {"score": None, "verdict": "no data", "reason": "no intraday bars",
             "components": {}, "vwap": None, "openingRange": None,
             "phase": session.phase(now), "tradeable": False, "side": side}
    if df is None or df.empty:
        return empty

    px = float(price if price is not None else df["Close"].iloc[-1])
    vw_series = vwap(df)
    vw = float(vw_series.iloc[-1]) if vw_series is not None else float("nan")
    orange = opening_range(df)
    ph = session.phase(now)

    # Dispersion of price around VWAP over the session — the natural scale for
    # "how far from average is far".
    if vw_series is not None:
        dev = (df["Close"] - vw_series).abs()
        spread = float(dev.tail(78).mean()) or float("nan")
    else:
        spread = float("nan")
    if not np.isfinite(spread) or spread <= 0:
        spread = abs(px) * 0.004

    tradeable = ph in TRADEABLE
    comp = {
        "vwap": _vwap_component(px, vw, spread, side),
        "value": _value_component(px, profile, side),
        "range": _range_component(px, orange, side),
        # Outside trading hours the phase has nothing to say about fill quality,
        # so it scores neutral. Scoring it zero would knock fifteen points off a
        # reading whose price components are perfectly informative.
        "phase": PHASE_SCORE.get(ph, 0.5),
    }
    score = round(100.0 * sum(comp[k] * WEIGHTS[k] for k in WEIGHTS), 1)

    # The opening is a hold regardless of how good the price looks, so the score
    # is capped to match. A screen reading "87 — wait" invites the user to
    # overrule the very thing the number was supposed to tell them.
    if ph == "opening":
        score = min(score, WAIT_CEILING)

    if not tradeable:
        verdict, reason = "closed", session.describe(now) + " — reading the last session"
    elif ph == "opening":
        verdict = "wait"
        reason = ("First thirty minutes — widest spreads and the opening auction's "
                  "noise. A patient buyer does better after the range sets.")
    elif score >= NOW_FLOOR:
        verdict = "now"
        reason = _reason(px, vw, profile, orange, side)
    elif score >= FAIR_FLOOR:
        verdict = "fair"
        reason = _reason(px, vw, profile, orange, side)
    else:
        verdict = "wait"
        reason = _reason(px, vw, profile, orange, side)

    return {"score": score, "verdict": verdict, "reason": reason,
            "components": {k: round(v, 3) for k, v in comp.items()},
            "weights": dict(WEIGHTS),
            "vwap": None if not np.isfinite(vw) else round(vw, 4),
            "hasVolume": vw_series is not None,
            "openingRange": orange, "phase": ph, "tradeable": tradeable,
            "side": side,
            "note": "execution quality only — this does not say whether to own the name"}


def _reason(px: float, vw: float, profile: dict | None,
            orange: dict | None, side: str) -> str:
    bits = []
    if np.isfinite(vw) and vw > 0:
        gap = (px - vw) / vw * 100.0
        where = "below" if gap < 0 else "above"
        bits.append(f"{abs(gap):.2f}% {where} VWAP")
    if profile and profile.get("val") is not None:
        if px <= profile["val"]:
            bits.append("at or under the value-area low")
        elif px >= profile["vah"]:
            bits.append("above the value-area high — chasing")
        else:
            bits.append("inside the value area")
    if orange and orange["height"] > 0:
        if px > orange["high"]:
            bits.append("extended above the opening range")
        elif px < orange["low"]:
            bits.append("under the opening range")
        else:
            bits.append("still inside the opening range")
    if not bits:
        return "not enough intraday detail to judge the fill"
    verb = "Selling" if side == "sell" else "Buying"
    return f"{verb} here is " + ", ".join(bits) + "."
