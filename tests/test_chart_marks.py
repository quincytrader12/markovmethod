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
    assert "if((POSITION && POSITION.avgEntry) !== before) renderChart();" in HTML


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
