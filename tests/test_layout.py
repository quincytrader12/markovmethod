"""Centre column layout.

The centre column exists to answer two questions: what is price doing, and what
is being said about it. Everything else there is reference material. These tests
pin the arrangement that follows from that — and, more importantly, the two ways
of getting it wrong: hiding numbers without replacing them, and unhiding a
canvas that was sized zero while it was folded away.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.web import AppState, create_app


def _html() -> str:
    app = create_app(AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True))
    return TestClient(app).get("/").text


HTML = _html()


def _rule(selector: str) -> str:
    """The declaration block for a selector, so a test reads the real CSS."""
    m = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", HTML)
    assert m, f"no rule for {selector}"
    return m.group(1)


# ── the chart gets the room ──────────────────────────────────────────────────
def test_the_chart_grows_and_has_a_floor():
    """flex alone lets a crowded column squeeze the chart to a sliver."""
    rule = _rule(".chartcard")
    assert "flex:1 1 auto" in rule
    assert "min-height:340px" in rule


def test_the_lower_panels_start_folded():
    assert "display:none" in _rule(".cbottom")
    assert "display:grid" in _rule(".cbottom.on")


def test_news_keeps_a_minimum_height():
    """A growing chart must not be allowed to squeeze the headlines to nothing."""
    rule = _rule(".news")
    assert "min-height:96px" in rule
    assert "max-height:22vh" in rule


# ── folding must not lose information ────────────────────────────────────────
def test_the_strip_carries_every_figure_it_hides():
    """Folding a panel away is only acceptable if its numbers move somewhere."""
    for stat in ("ms-sharpe", "ms-win", "ms-dd", "ms-psr", "ms-fc"):
        assert f'id="{stat}"' in HTML, f"{stat} vanished with the panel"


def test_the_strip_is_filled_from_the_same_payload_as_the_panels():
    assert "function renderStrip" in HTML
    assert "renderStrip(st.metrics, st.forecast)" in HTML


def test_a_near_tie_is_not_reported_as_a_call():
    """Three states summing to one are often within a point or two of each other.
    A bar chart shows that; a one-line summary naming the tallest states an edge
    the chain does not have."""
    body = HTML.split("function renderStrip")[1].split("function renderExtras")[0]
    assert "MIXED" in body
    assert "row[best] - row[rank[1]] < 0.05" in body
    assert "el.title" in body, "the full split must stay reachable"


def test_the_strip_opens_the_panels():
    assert 'onclick="toggleBottom()"' in HTML
    assert "function toggleBottom" in HTML


# ── the trap: a canvas sized zero while hidden ───────────────────────────────
def test_opening_the_panels_redraws_them():
    """`display:none` gives a canvas no width, so the equity curve drawn while
    folded is drawn into nothing. Unhiding it would show a blank box until the
    next fifteen-second poll unless opening redraws immediately."""
    body = HTML.split("function toggleBottom")[1].split("function renderStrip")[0]
    assert "renderExtras(LAST)" in body


def test_opening_the_panels_redraws_the_chart_too():
    """The chart's own height changes when the panels appear, and nothing in the
    page listens for a resize."""
    body = HTML.split("function toggleBottom")[1].split("function renderStrip")[0]
    assert "renderChart()" in body and "loadTPO()" in body


def test_hidden_panels_are_not_drawn_into():
    body = HTML.split("function renderExtras")[1].split("function drawDonut")[0]
    assert "if(!BOTTOM_OPEN) return;" in body
    # the strip is filled before the early return, or folding would freeze it
    assert body.index("renderStrip(") < body.index("if(!BOTTOM_OPEN) return;")


# ── the preference sticks ────────────────────────────────────────────────────
def test_the_choice_survives_a_reload():
    assert "localStorage.getItem('mamba.bottom')" in HTML
    assert "localStorage.setItem('mamba.bottom'" in HTML
    assert "applyBottom();" in HTML.split("async function boot")[1]


def test_storage_failure_does_not_break_the_page():
    """localStorage throws outright in some privacy modes."""
    for part in HTML.split("mamba.bottom")[1:]:
        assert "catch(e){}" in part[:120], "an unguarded localStorage call"
