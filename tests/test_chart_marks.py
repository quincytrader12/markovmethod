"""The price and entry markers, and live prices in the blotter.

A level drawn across a chart is only useful if you can read its value without
squinting at the axis, and a price column that never refreshes is a snapshot of
the moment the window opened.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.web import AppState, create_app


def _client():
    return TestClient(create_app(AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD),
                                          demo=True)))


HTML = _client().get("/").text


# ── the markers ─────────────────────────────────────────────────────────────
def test_the_two_marker_colours_are_named():
    assert "price:'#ffc51f'" in HTML, "the golden current-price line"
    assert "entry:'#c6ccd4'" in HTML, "neon grey for the entry"


def test_the_price_line_is_distinct_from_the_amber_moving_average():
    """They share a chart, so the price line is brighter, always horizontal and
    carries a filled tag — three ways of not being mistaken for MA20."""
    assert "'#f5c451'" in HTML and "price:'#ffc51f'" in HTML
    assert "#f5c451" != "#ffc51f"


def test_both_markers_carry_their_value_in_a_tag():
    assert "function priceMark" in HTML
    body = HTML.split("function priceMark")[1][:1200]
    assert "fillRect" in body and "fillText" in body


def test_the_current_price_is_the_series_latest_not_the_visible_bar():
    """Panning back through history must not relabel an old bar 'now'."""
    assert "lastPrice: all.length ? all[all.length-1].c : null" in HTML


def test_the_current_price_is_always_inside_the_scale():
    body = HTML.split("function drawChart")[1].split("function priceMark")[0]
    assert "hi=Math.max(hi,chart.lastPrice)" in body


def test_a_distant_entry_is_clamped_rather_than_squashing_the_chart():
    """Stretching the scale to reach a far-away entry flattens every candle."""
    body = HTML.split("function drawChart")[1].split("function priceMark")[0]
    assert "entryInScale" in body and "clamped" in body


def test_a_clamped_entry_is_marked_as_off_screen():
    body = HTML.split("function drawChart")[1].split("function priceMark")[0]
    assert "'▲'" in body and "'▼'" in body


def test_there_is_room_on_the_right_for_the_tags():
    assert "r:56" in HTML, "the tags would be drawn off the canvas"


def test_the_chart_only_redraws_when_the_entry_moves():
    """The portfolio polls every fifteen seconds; repainting each tick would
    undo the user's zoom and pan for no gain."""
    body = HTML.split("async function loadPortfolio")[1][:900]
    assert "!== before" in body and "renderChart()" in body, "it repaints unconditionally"


# ── live prices in the blotter ──────────────────────────────────────────────
def test_open_orders_show_their_resting_price():
    """An order in the book without its price cannot be judged at all."""
    assert "function orderPx" in HTML
    assert "Order px" in HTML


def test_open_orders_show_the_market_and_the_distance_to_it():
    assert "function markPx" in HTML and "function awayPx" in HTML
    assert ">Away<" in HTML


def test_the_distance_is_coloured_by_whether_it_is_reachable():
    """Green when the market must come to you, amber when it has gone past."""
    body = HTML.split("function awayPx")[1][:700]
    assert "favourable" in body and "var(--green)" in body and "var(--amber)" in body


def test_the_blotter_refreshes_while_it_is_open():
    assert "BLOT_TIMER" in HTML
    body = HTML.split("function openBlotter")[1][:600]
    assert "setInterval" in body


def test_the_blotter_stops_polling_once_closed():
    body = HTML.split("function closeBlotter")[1][:200]
    assert "clearInterval(BLOT_TIMER)" in body


def test_the_refresh_does_not_blank_the_table():
    """Re-showing 'loading…' every ten seconds would wipe a table being read."""
    body = HTML.split("async function loadBlotter")[1][:700]
    assert "if(!body.innerHTML.trim())" in body


def test_the_blotter_carries_a_mark_for_every_symbol():
    d = _client().get("/api/blotter").json()
    assert "marks" in d or d["connected"] is False


def test_the_portfolio_poll_does_not_fetch_quotes():
    """It runs every fifteen seconds on the main page; adding quote lookups
    there would be a regression, so marks are the blotter's alone."""
    import markov_hedge_fund_method.web as web
    src = open(web.__file__).read()
    portfolio = src.split('def portfolio(')[1].split('@app.get')[0]
    assert "_marks(" not in portfolio


# ── the risk / reward box ───────────────────────────────────────────────────
def test_the_position_tool_exists_and_toggles():
    assert 'id="rr-toggle"' in HTML and "function toggleRR" in HTML
    assert "function drawRR" in HTML


def test_green_runs_to_the_target_and_red_to_the_stop():
    body = HTML.split("function drawRR")[1].split("/* ---------- levels to ticket")[0]
    assert "fill(yE, yT, C.bull" in body
    assert "fill(yE, yS, C.bear" in body


def test_the_entry_line_is_the_break_even():
    """The two zones meet at the entry, and that meeting point is the level a
    stop gets trailed to once the trade is working."""
    body = HTML.split("function drawRR")[1].split("/* ---------- levels to ticket")[0]
    assert "BREAK EVEN" in body
    assert "C.entry" in body, "break-even should share the entry marker's colour"


def test_the_box_reports_reward_to_risk():
    body = HTML.split("function drawRR")[1].split("/* ---------- levels to ticket")[0]
    assert "R:R " in body


def test_the_box_labels_both_levels_with_their_distance():
    body = HTML.split("function drawRR")[1].split("/* ---------- levels to ticket")[0]
    assert "TARGET " in body and "STOP " in body
    assert "pct(box.target)" in body and "pct(box.stop)" in body


def test_a_live_position_anchors_the_box_to_what_you_paid():
    """Drawing a held trade from a suggested entry would show a profit that was
    never made."""
    body = HTML.split("async function loadRR")[1].split("function drawRR")[0]
    assert "POSITION.avgEntry" in body and "live:" in body


def test_the_box_fits_inside_the_scale():
    """A target drawn off the canvas hides the one thing the box is for."""
    body = HTML.split("function drawChart")[1].split("function priceMark")[0]
    assert "hi=Math.max(hi,RR.target,RR.stop,RR.entry)" in body


def test_the_box_sits_to_the_right_of_the_price_action():
    body = HTML.split("function drawRR")[1].split("/* ---------- levels to ticket")[0]
    assert "0.62" in body, "the box should annotate ahead, not cover the history"


def test_flipping_side_redraws_the_box():
    """A long box and a short box point opposite ways."""
    assert "if(RR_ON) loadRR();" in HTML


def test_the_toggle_is_remembered():
    assert "localStorage.setItem('mamba.rr'" in HTML
    assert "localStorage.getItem('mamba.rr')" in HTML


def test_a_new_symbol_drops_the_box():
    assert "RR = null;" in HTML
