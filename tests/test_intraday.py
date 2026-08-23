"""Entry timing.

The thing this module must never become is a second opinion that argues with the
daily regime, so the tests pin what it says as hard as what it computes: a score
is about fill quality, it never asserts a direction, and it can never disagree
with its own verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markov_hedge_fund_method import intraday, tpo
from markov_hedge_fund_method.intraday import (
    anchored_vwap,
    atr,
    entry_timing,
    opening_range,
    vwap,
)
from markov_hedge_fund_method.market_data import synthetic_intraday


def bars(prices, volumes=None, start="2026-08-24 09:30", freq="5min"):
    idx = pd.date_range(start, periods=len(prices), freq=freq)
    p = np.asarray(prices, dtype=float)
    df = pd.DataFrame({"Open": p, "High": p, "Low": p, "Close": p}, index=idx)
    if volumes is not None:
        df["Volume"] = np.asarray(volumes, dtype=float)
    return df


SESSION = synthetic_intraday("1D", seed=1)
PROFILE = tpo.build_profile(SESSION)
MORNING = pd.Timestamp("2026-08-24 10:30")


# ── VWAP ────────────────────────────────────────────────────────────────────
def test_vwap_is_weighted_by_volume_not_a_plain_average():
    """The trap: a mean of prices looks close enough to be missed on a chart."""
    df = bars([10.0, 20.0], [1.0, 9.0])
    assert float(vwap(df).iloc[-1]) == pytest.approx(19.0)   # not 15.0


def test_vwap_without_volume_is_none_rather_than_a_lookalike():
    """An unweighted average wearing the name VWAP would be read as the real
    thing. Absent has to look absent."""
    assert vwap(bars([10.0, 20.0])) is None
    assert vwap(bars([10.0, 20.0], [0.0, 0.0])) is None


def test_vwap_resets_each_session():
    """A VWAP that ran across days would carry yesterday's crowd into today."""
    day1 = bars([10.0] * 3, [1.0] * 3, start="2026-08-24 09:30")
    day2 = bars([50.0] * 3, [1.0] * 3, start="2026-08-25 09:30")
    v = vwap(pd.concat([day1, day2]))
    assert float(v.iloc[-1]) == pytest.approx(50.0)


def test_vwap_starts_at_the_first_bar_price():
    df = bars([10.0, 20.0, 30.0], [1.0, 1.0, 1.0])
    assert float(vwap(df).iloc[0]) == pytest.approx(10.0)


def test_anchored_vwap_does_not_reset():
    df = pd.concat([bars([10.0] * 3, [1.0] * 3, start="2026-08-24 09:30"),
                    bars([20.0] * 3, [1.0] * 3, start="2026-08-25 09:30")])
    v = anchored_vwap(df, "2026-08-24 09:30")
    assert float(v.iloc[-1]) == pytest.approx(15.0)     # both days averaged


def test_anchored_vwap_starts_where_you_anchor_it():
    df = bars([10.0, 10.0, 30.0, 30.0], [1.0] * 4)
    assert float(anchored_vwap(df, df.index[2]).iloc[-1]) == pytest.approx(30.0)


def test_anchored_vwap_past_the_end_is_none():
    assert anchored_vwap(bars([10.0], [1.0]), "2030-01-01") is None


# ── opening range ───────────────────────────────────────────────────────────
def test_the_opening_range_is_the_first_thirty_minutes():
    df = bars(list(range(20)), [1.0] * 20)          # 5-min bars from 09:30
    o = opening_range(df, minutes=30)
    assert o["bars"] == 6                            # 09:30..09:55 inclusive
    assert o["high"] == 5.0 and o["low"] == 0.0


def test_the_opening_range_uses_only_the_latest_session():
    old = bars([100.0] * 6, [1.0] * 6, start="2026-08-24 09:30")
    new = bars([5.0] * 6, [1.0] * 6, start="2026-08-25 09:30")
    o = opening_range(pd.concat([old, new]))
    assert o["high"] == 5.0, "yesterday's range leaked into today"


def test_an_unfinished_opening_range_says_so():
    assert opening_range(bars([1.0, 2.0], [1.0, 1.0]))["complete"] is False


def test_no_bars_gives_no_opening_range():
    assert opening_range(pd.DataFrame()) is None


# ── ATR ─────────────────────────────────────────────────────────────────────
def _daily(closes):
    """Daily bars with an identical one-point high-low range on every bar, so
    the only thing that can differ between two frames is the gap between them."""
    idx = pd.date_range("2026-08-24", periods=len(closes), freq="D")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"Open": c, "High": c + 0.5, "Low": c - 0.5, "Close": c}, index=idx)


def test_atr_counts_the_overnight_gap():
    """High-minus-low misses a gap entirely, and on a multi-day hold the gap is
    the risk that matters. Both frames below have the same one-point daily
    range; only one of them gaps."""
    steady = atr(_daily([10.0, 10.0, 10.0, 10.0]))
    gapping = atr(_daily([10.0, 20.0, 30.0, 40.0]))
    assert steady == pytest.approx(1.0)
    assert gapping > steady * 5


def test_the_first_bar_has_no_gap_to_measure():
    """No previous close exists, so its true range is simply high minus low."""
    assert atr(_daily([10.0])) is None          # one bar is not a range at all
    assert atr(_daily([10.0, 10.0])) == pytest.approx(1.0)


def test_atr_needs_two_bars():
    assert atr(bars([1.0], [1.0])) is None
    assert atr(pd.DataFrame()) is None


# ── the score ───────────────────────────────────────────────────────────────
def test_buying_below_vwap_beats_buying_above():
    lo, hi = float(SESSION["Close"].min()), float(SESSION["Close"].max())
    cheap = entry_timing(SESSION, price=lo, profile=PROFILE, now=MORNING)
    dear = entry_timing(SESSION, price=hi, profile=PROFILE, now=MORNING)
    assert cheap["score"] > dear["score"]
    assert cheap["verdict"] == "now" and dear["verdict"] == "wait"


def test_a_sell_mirrors_a_buy():
    """Selling into strength is the same judgement upside down."""
    hi = float(SESSION["Close"].max())
    buy = entry_timing(SESSION, price=hi, profile=PROFILE, side="buy", now=MORNING)
    sell = entry_timing(SESSION, price=hi, profile=PROFILE, side="sell", now=MORNING)
    assert sell["score"] > buy["score"]


def test_the_opening_is_a_wait_however_good_the_price():
    lo = float(SESSION["Close"].min())
    r = entry_timing(SESSION, price=lo, profile=PROFILE,
                     now=pd.Timestamp("2026-08-24 09:40"))
    assert r["verdict"] == "wait" and r["phase"] == "opening"


def test_the_score_never_argues_with_the_verdict():
    """A screen reading '87 - wait' invites the user to overrule the number that
    was meant to be telling them something."""
    for when in ("2026-08-24 09:40", "2026-08-24 10:30", "2026-08-24 12:30"):
        for px in (float(SESSION["Close"].min()), float(SESSION["Close"].max())):
            r = entry_timing(SESSION, price=px, profile=PROFILE, now=pd.Timestamp(when))
            if r["verdict"] == "now":
                assert r["score"] >= intraday.NOW_FLOOR
            elif r["verdict"] == "fair":
                assert intraday.FAIR_FLOOR <= r["score"] < intraday.NOW_FLOOR
            elif r["verdict"] == "wait":
                assert r["score"] < intraday.FAIR_FLOOR


def test_a_closed_market_scores_the_phase_neutral_not_zero():
    """Outside hours the phase says nothing about fill quality. Scoring it zero
    would dock fifteen points from a reading whose price components are fine."""
    lo = float(SESSION["Close"].min())
    shut = entry_timing(SESSION, price=lo, profile=PROFILE,
                        now=pd.Timestamp("2026-08-24 21:00"))
    assert shut["verdict"] == "closed"
    assert shut["components"]["phase"] == 0.5
    assert shut["score"] > 50, "a good price still reads as a good price"


def test_the_result_never_asserts_a_direction():
    """The daily regime owns direction. If this module ever starts implying it,
    the terminal has two opinions and the user has to arbitrate."""
    r = entry_timing(SESSION, price=100.0, profile=PROFILE, now=MORNING)
    assert "does not say whether to own the name" in r["note"]
    assert set(r) >= {"score", "verdict", "components", "phase", "side"}
    assert "signal" not in r and "regime" not in r


def test_missing_pieces_degrade_rather_than_crash():
    plain = SESSION.drop(columns=["Volume"])
    r = entry_timing(plain, price=100.0, profile=None, now=MORNING)
    assert r["hasVolume"] is False and r["vwap"] is None
    assert r["score"] is not None


def test_no_bars_at_all_is_reported_not_guessed():
    r = entry_timing(pd.DataFrame(), now=MORNING)
    assert r["score"] is None and r["verdict"] == "no data"


def test_the_components_and_weights_are_shown():
    """A score the user cannot take apart is a number to be believed on faith."""
    r = entry_timing(SESSION, price=100.0, profile=PROFILE, now=MORNING)
    assert set(r["components"]) == set(r["weights"]) == set(intraday.WEIGHTS)
    assert sum(r["weights"].values()) == pytest.approx(1.0)


def test_the_reason_names_the_levels():
    lo = float(SESSION["Close"].min())
    r = entry_timing(SESSION, price=lo, profile=PROFILE, now=MORNING)
    assert "VWAP" in r["reason"]


# ── the API ─────────────────────────────────────────────────────────────────
def _client():
    from fastapi.testclient import TestClient

    from markov_hedge_fund_method.config import Mode, Settings
    from markov_hedge_fund_method.web import AppState, create_app
    return TestClient(create_app(AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD),
                                          demo=True)))


def test_the_endpoint_returns_the_levels_and_the_score():
    d = _client().get("/api/timing", params={"symbol": "SPY"}).json()
    for key in ("score", "verdict", "components", "phase", "session",
                "isOpen", "atr", "valueArea", "openingRange", "vwap"):
        assert key in d
    assert set(d["valueArea"]) == {"poc", "vah", "val"}


def test_the_endpoint_refuses_a_timing_read_on_placeholder_bars():
    """Levels drawn from invented bars are a demo. A verdict drawn from them
    would be advice about a market that does not exist."""
    d = _client().get("/api/timing", params={"symbol": "SPY"}).json()
    assert d["real"] is False
    assert d["verdict"] == "no data"
    assert "Placeholder bars" in d["reason"]


def test_the_endpoint_takes_a_side():
    c = _client()
    buy = c.get("/api/timing", params={"symbol": "SPY", "side": "buy"}).json()
    sell = c.get("/api/timing", params={"symbol": "SPY", "side": "sell"}).json()
    assert buy["side"] == "buy" and sell["side"] == "sell"
    assert buy["components"] != sell["components"]


def test_candles_carry_vwap_and_the_opening_range():
    d = _client().get("/api/candles", params={"symbol": "SPY", "tf": "1D"}).json()
    assert d["vwap"] and len(d["vwap"]) == len(d["bars"])
    assert d["openingRange"]["high"] >= d["openingRange"]["low"]


# ── the UI ──────────────────────────────────────────────────────────────────
def test_the_page_shows_a_timing_chip():
    html = _client().get("/").text
    assert 'id="timing-chip"' in html and "function renderTiming" in html


def test_the_chip_states_are_distinguishable():
    html = _client().get("/").text
    for state in (".timing-chip.now", ".timing-chip.fair", ".timing-chip.wait"):
        assert state in html


def test_vwap_is_drawn_dashed():
    """A third solid line on the chart would read as another moving average.
    VWAP is a different quantity — an average of what was paid."""
    html = _client().get("/").text
    assert "chart.vwap" in html and "setLineDash" in html


def test_vwap_is_inside_the_y_scale():
    """A line excluded from the extent calculation gets drawn off-canvas."""
    html = _client().get("/").text
    body = html.split("function drawChart")[1].split("function line")[0]
    assert "chart.vwap" in body.split("const X =")[0], "vwap missing from the scale"


def test_the_chip_says_it_is_not_a_directional_call():
    html = _client().get("/").text
    assert "does not say whether to own the name" in html


# ── where the bars come from ────────────────────────────────────────────────
class _Fake:
    """Minimal stand-in for Settings — only what the selector reads."""
    def __init__(self, pref="auto", keys=True):
        self.intraday_source = pref
        self._keys = keys
        self.ticker = "SPY"
        self.api_key = "k" if keys else None
        self.api_secret = "s" if keys else None

    @property
    def has_credentials(self):
        return self._keys


def test_yahoo_is_tried_first_by_default():
    """Alpaca's free plan is IEX only — a few percent of consolidated volume —
    so trying it first would make connecting a broker account produce *worse*
    intraday bars than having no account at all."""
    from markov_hedge_fund_method.market_data import _intraday_order
    assert _intraday_order(_Fake("auto", keys=True))[0] == "yahoo"


def test_a_paid_data_plan_can_ask_for_alpaca_first():
    from markov_hedge_fund_method.market_data import _intraday_order
    assert _intraday_order(_Fake("alpaca", keys=True))[0] == "alpaca"


def test_every_preference_still_falls_back():
    """One feed being down must never blank the chart."""
    from markov_hedge_fund_method.market_data import INTRADAY_SOURCES, _intraday_order
    for pref in INTRADAY_SOURCES:
        assert len(_intraday_order(_Fake(pref, keys=True))) == 2


def test_without_credentials_only_the_free_feed_is_possible():
    from markov_hedge_fund_method.market_data import _intraday_order
    for pref in ("auto", "yahoo", "alpaca"):
        assert _intraday_order(_Fake(pref, keys=False)) == ["yahoo"]


def test_the_error_names_every_feed_it_tried():
    """"intraday unavailable" with no reason leaves nothing to act on."""
    import pytest as _pytest

    import markov_hedge_fund_method.market_data as md

    def boom(msg):
        def _f(*a, **k):
            raise RuntimeError(msg)
        return _f

    saved = (md._alpaca_intraday, md._yfinance_intraday)
    md._alpaca_intraday = boom("subscription does not permit")
    md._yfinance_intraday = boom("no data")
    try:
        with _pytest.raises(RuntimeError) as e:
            md.get_intraday_ohlc(_Fake("auto", keys=True), "1D")
        assert "Alpaca" in str(e.value) and "Yahoo" in str(e.value)
    finally:
        md._alpaca_intraday, md._yfinance_intraday = saved


# ── levels to a ticket ──────────────────────────────────────────────────────
def test_a_buy_enters_at_the_low_edge_of_value():
    lv = intraday.ticket_levels(SESSION, profile=PROFILE, side="buy")
    assert lv["limit"] == pytest.approx(min(PROFILE["val"],
                                            float(SESSION["Close"].iloc[-1])), abs=0.01)
    assert lv["target"] == pytest.approx(PROFILE["vah"], abs=0.01)


def test_a_buy_stop_sits_under_the_opening_range_not_on_it():
    """A stop exactly on the level everyone can see is the one taken out by the
    poke through it."""
    lv = intraday.ticket_levels(SESSION, profile=PROFILE, side="buy")
    assert lv["stop"] < intraday.opening_range(SESSION)["low"]


def test_a_sell_mirrors_the_whole_structure():
    lv = intraday.ticket_levels(SESSION, profile=PROFILE, side="sell")
    assert lv["stop"] > intraday.opening_range(SESSION)["high"]
    assert lv["target"] == pytest.approx(PROFILE["val"], abs=0.01)


def test_a_buy_never_bids_above_the_market():
    """Suggesting an entry above where price already is turns a patient limit
    into a market order that pays the spread."""
    lv = intraday.ticket_levels(SESSION, profile=PROFILE, side="buy")
    assert lv["limit"] <= float(SESSION["Close"].iloc[-1]) + 1e-9


def test_the_reward_to_risk_is_computed():
    lv = intraday.ticket_levels(SESSION, profile=PROFILE, side="buy")
    assert lv["rr"] == pytest.approx(lv["rewardPerShare"] / lv["riskPerShare"], abs=0.01)


def test_every_level_says_where_it_came_from():
    """A price the user cannot trace is a price they cannot argue with."""
    lv = intraday.ticket_levels(SESSION, profile=PROFILE, side="buy")
    assert set(lv["why"]) == {"limit", "stop", "target"}
    assert "value-area" in lv["why"]["limit"]


def test_levels_survive_a_missing_profile():
    lv = intraday.ticket_levels(SESSION, profile=None, side="buy")
    assert lv["limit"] is not None and lv["stop"] is not None
    assert "no profile" in lv["why"]["limit"]


def test_no_bars_gives_no_levels():
    import pandas as _pd
    lv = intraday.ticket_levels(_pd.DataFrame(), side="buy")
    assert lv["limit"] is None and "error" in lv["why"]


# ── sizing off risk, not cash ───────────────────────────────────────────────
def test_size_comes_from_the_distance_to_the_stop():
    """Sizing off buying power makes every trade the same bet regardless of how
    far the stop is, which stops a win rate meaning anything."""
    assert intraday.shares_for_risk(12_000, 0.01, 2.80) == 42
    assert intraday.shares_for_risk(12_000, 0.01, 5.60) == 21


def test_a_wider_stop_buys_fewer_shares():
    tight = intraday.shares_for_risk(50_000, 0.01, 1.0)
    wide = intraday.shares_for_risk(50_000, 0.01, 10.0)
    assert tight == wide * 10


def test_sizing_refuses_the_impossible():
    assert intraday.shares_for_risk(0, 0.01, 2.0) is None
    assert intraday.shares_for_risk(10_000, 0.01, 0) is None


# ── the levels endpoint ─────────────────────────────────────────────────────
def test_the_levels_endpoint_refuses_placeholder_bars():
    """Typing invented levels into a live order ticket is worse than none."""
    d = _client().get("/api/ticket-levels", params={"symbol": "SPY"}).json()
    assert d["real"] is False
    assert d["limit"] is None and d["stop"] is None and d["target"] is None
    assert "error" in d["why"]


def test_the_page_offers_the_levels_button():
    html = _client().get("/").text
    assert "function useLevels" in html and "USE LEVELS" in html


def test_applying_levels_builds_a_bracket():
    """An entry with no exits attached is the half that gets forgotten once the
    position is on."""
    body = _client().get("/").text.split("async function useLevels")[1][:1800]
    assert "'bracket'" in body and "o-slstop" in body and "o-tp" in body


def test_the_button_fills_the_ticket_but_never_sends_it():
    body = _client().get("/").text.split("async function useLevels")[1][:1800]
    assert "submitOrder" not in body


# ── timeframes ──────────────────────────────────────────────────────────────
def test_the_fifteen_and_thirty_minute_timeframes_exist():
    from markov_hedge_fund_method.market_data import _INTRADAY_TF
    assert "15M" in _INTRADAY_TF and "30M" in _INTRADAY_TF


def test_every_intraday_timeframe_renders_offline():
    from markov_hedge_fund_method.market_data import _INTRADAY_TF, synthetic_intraday
    for tf in _INTRADAY_TF:
        df = synthetic_intraday(tf)
        assert not df.empty and "Volume" in df.columns, tf


def test_a_finer_timeframe_covers_less_ground():
    """Same bar count, shorter window — otherwise the label means nothing."""
    from markov_hedge_fund_method.market_data import synthetic_intraday
    span = lambda tf: (lambda d: d.index[-1] - d.index[0])(synthetic_intraday(tf))
    assert span("15M") < span("30M") < span("1H") < span("4H")


def test_the_finer_timeframes_stay_inside_yahoos_history_limit():
    """Yahoo caps 15-minute and finer at 60 days and silently truncates past it,
    which would leave the chart quietly shorter than it claims."""
    from markov_hedge_fund_method.market_data import _INTRADAY_TF
    limits = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90}
    for tf in ("15M", "30M"):
        period, interval = _INTRADAY_TF[tf]["yf"]
        assert limits[period] <= 60, f"{tf} asks for {period} of {interval} bars"


def test_the_page_offers_the_new_timeframes():
    html = _client().get("/").text
    assert 'data-tf="15M"' in html and 'data-tf="30M"' in html


def test_the_new_timeframes_serve_candles():
    c = _client()
    for tf in ("15M", "30M"):
        d = c.get("/api/candles", params={"symbol": "SPY", "tf": tf}).json()
        assert d["bars"] and d["tf"] == tf
        assert d["vwap"], f"{tf} lost its VWAP"
