"""Chart zoom and pan.

The window logic is plain arithmetic over a bar range, so it is worth pinning
down here rather than only in a browser: the failure modes are scrolling past
the end of the data, zooming to nothing, and keeping a stale window when the
series underneath changes.
"""

from __future__ import annotations

import re
from pathlib import Path

HTML = Path(__file__).resolve().parents[1] / "markov_hedge_fund_method" / "web_static" / "index.html"
SOURCE = HTML.read_text(encoding="utf-8")


# ── a Python mirror of the clamp, so the arithmetic is testable ──────────────
MIN_BARS = 20


def clamp(start, end, n):
    span = min(n, max(MIN_BARS, round(end - start)))
    s = max(0, min(round(start), n - span))
    return s, s + span


def test_the_window_never_runs_off_the_right_edge():
    assert clamp(500, 620, 520) == (400, 520)


def test_the_window_never_runs_off_the_left_edge():
    assert clamp(-80, 40, 520) == (0, 120)


def test_zooming_in_stops_at_the_minimum():
    s, e = clamp(300, 302, 520)
    assert e - s == MIN_BARS, "a chart zoomed to two bars is unreadable"


def test_zooming_out_stops_at_the_whole_series():
    assert clamp(-500, 2000, 520) == (0, 520)


def test_a_series_shorter_than_the_minimum_still_renders():
    s, e = clamp(0, 5, 5)
    assert (s, e) == (0, 5) or e - s <= 5, "must not demand more bars than exist"


def test_panning_preserves_the_zoom_level():
    for shift in (-200, -5, 5, 200):
        s, e = clamp(200 + shift, 260 + shift, 520)
        assert e - s == 60


# ── the anchored-zoom arithmetic ────────────────────────────────────────────
def zoom(start, end, n, frac, factor):
    span = end - start
    nxt = min(n, max(MIN_BARS, round(span * factor)))
    anchor = start + frac * span
    return clamp(anchor - frac * nxt, anchor - frac * nxt + nxt, n)


def test_zoom_keeps_the_bar_under_the_cursor_put():
    start, end, n, frac = 100, 300, 520, 0.5
    anchor = start + frac * (end - start)              # bar 200, mid-screen
    s, e = zoom(start, end, n, frac, 0.5)
    assert abs((s + frac * (e - s)) - anchor) <= 1, "the chart slid under the cursor"


def test_zooming_at_the_left_edge_anchors_left():
    s, e = zoom(100, 300, 520, 0.0, 0.5)
    assert s == 100, "zooming at the left edge should hold the left edge"


def test_zooming_at_the_right_edge_anchors_right():
    s, e = zoom(100, 300, 520, 1.0, 0.5)
    assert e == 300, "zooming at the right edge should hold the right edge"


# ── the wiring in the page ──────────────────────────────────────────────────
def test_the_canvas_listens_for_all_three_gestures():
    for gesture in ("'wheel'", "'pointerdown'", "'pointermove'", "'dblclick'"):
        assert gesture in SOURCE, f"the chart does not handle {gesture}"


def test_the_wheel_handler_can_actually_prevent_scrolling():
    """Without passive:false the page scrolls instead of the chart zooming."""
    m = re.search(r"addEventListener\('wheel'.*?\{passive:\s*false\}", SOURCE, re.S)
    assert m, "the wheel listener is passive, so preventDefault does nothing"


def test_the_canvas_opts_out_of_browser_touch_gestures():
    assert "touch-action:none" in SOURCE, "touch panning would fight the drag handler"


def test_changing_timeframe_clears_the_zoom():
    fn = re.search(r"function setTF\([^)]*\)\{(.*?)\n\}", SOURCE, re.S).group(1)
    assert "VIEW = null" in fn, "a stale window would show a random slice of the new range"


def test_changing_symbol_clears_the_zoom():
    fn = re.search(r"function selectSymbol\([^)]*\)\{(.*?)\n\}", SOURCE, re.S).group(1)
    assert "VIEW = null" in fn


def test_the_renderer_and_the_handlers_share_one_geometry():
    """Two copies of the padding would put the cursor anchor off by a few bars."""
    assert re.search(r"const PAD = \{l:\d+,r:\d+,t:\d+,b:\d+\}", SOURCE)
    assert "const pad = PAD;" in SOURCE, "drawChart kept its own padding"
    assert len(re.findall(r"\{l:46,r:12,t:12,b:34\}", SOURCE)) == 1


def test_the_reset_control_exists_and_is_reachable():
    assert 'id="zoom-tag"' in SOURCE and "resetZoom()" in SOURCE
    assert "function resetZoom()" in SOURCE


def test_the_zoom_chip_hides_at_the_default_view():
    """It reads as 'you have moved the view', so it must be off by default —
    otherwise it is lit on every timeframe short of the longest."""
    fn = re.search(r"function updateZoomTag\(n\)\{(.*?)\n\}", SOURCE, re.S).group(1)
    assert "defaultSpan(n)" in fn, "the chip compares against the full history, not the default"
