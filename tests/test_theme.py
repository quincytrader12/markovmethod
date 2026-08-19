"""The terminal's palette: neutral dark-grey surfaces, neon reserved for data.

There is a real trap here, and it has already been sprung once: the palette
lives in two places. The stylesheet holds one copy and the canvas chart holds
its own, so neutralising the CSS alone left the chart still drawing navy
gridlines and a violet moving average. These tests read the whole file, not the
`<style>` block, so the two copies cannot drift apart again.
"""

from __future__ import annotations

import colorsys
import re
from pathlib import Path

HTML = Path(__file__).resolve().parents[1] / "markov_hedge_fund_method" / "web_static" / "index.html"
SOURCE = HTML.read_text(encoding="utf-8")
STYLE = SOURCE.split("<style>", 1)[1].split("</style>", 1)[0]
SCRIPT = SOURCE.split("<style>", 1)[0] + SOURCE.split("</style>", 1)[1]

# The four neon accents, which are supposed to be loud.
NEON = {"#38f0e0", "#7ef9ee", "#2fe08a", "#ff4d5e", "#f5c451", "#f5834d", "#9fe06a"}


def _hsl(h: str):
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    hue, lum, sat = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return hue * 360, sat, lum


def _colours(text: str):
    return {c.lower() for c in re.findall(r"#[0-9a-fA-F]{6}\b", text)}


def test_no_saturated_blue_surfaces_anywhere():
    """The complaint that started this: blue surfaces everywhere, and too loud."""
    offenders = []
    for c in _colours(SOURCE):
        if c in NEON:
            continue
        hue, sat, lum = _hsl(c)
        if 190 <= hue <= 260 and sat > 0.20:
            offenders.append((c, round(hue), round(sat, 2)))
    assert not offenders, f"saturated blue is back: {sorted(offenders)}"


def test_surfaces_are_effectively_neutral():
    """Chrome may keep a whisper of coolness, but nothing more."""
    loud = []
    for c in _colours(SOURCE):
        if c in NEON:
            continue
        hue, sat, lum = _hsl(c)
        if lum < 0.45 and sat > 0.12:          # a dark surface with real colour
            loud.append((c, round(sat, 2)))
    assert not loud, f"these surfaces are still tinted: {sorted(loud)}"


def test_the_neon_survived():
    """Toning down the base must not have flattened the data colours."""
    for accent in ("#38f0e0", "#2fe08a", "#ff4d5e", "#f5c451"):
        assert accent in SOURCE.lower(), f"{accent} went missing"
        _, sat, _ = _hsl(accent)
        assert sat > 0.6, f"{accent} lost its punch"


def test_the_base_is_dark_grey_not_near_black():
    m = re.search(r"--bg:\s*(#[0-9a-fA-F]{6})", STYLE)
    assert m, "--bg is no longer declared"
    hue, sat, lum = _hsl(m.group(1))
    assert 0.05 <= lum <= 0.16, "the base should read as a light black, not pitch or grey-mid"
    assert sat < 0.12, "the base must not be tinted"


def test_text_stays_readable_against_the_base():
    bg = re.search(r"--bg:\s*(#[0-9a-fA-F]{6})", STYLE).group(1)
    ink = re.search(r"--ink:\s*(#[0-9a-fA-F]{6})", STYLE).group(1)

    def lin(c):
        v = c / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    def lum(h):
        r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
        return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)

    ratio = (lum(ink) + 0.05) / (lum(bg) + 0.05)
    assert ratio >= 7.0, f"body text contrast is only {ratio:.1f}:1"


def test_panels_are_distinguishable_from_the_page():
    """Neutralising everything to one grey would lose the panel edges."""
    bg = re.search(r"--bg:\s*(#[0-9a-fA-F]{6})", STYLE).group(1)
    panel = re.search(r"--panel:\s*(#[0-9a-fA-F]{6})", STYLE).group(1)
    assert _hsl(panel)[2] > _hsl(bg)[2], "panels must sit above the page, not merge into it"


def test_the_chart_palette_matches_the_stylesheet():
    """The canvas keeps its own copy — this is the drift that already happened."""
    js = re.search(r"const C = \{([^}]*)\}", SCRIPT)
    assert js, "the chart palette object is gone"
    for c in _colours(js.group(1)):
        if c in NEON:
            continue
        hue, sat, lum = _hsl(c)
        assert not (190 <= hue <= 260 and sat > 0.20), f"chart colour {c} is still navy"


def test_the_moving_average_lines_are_not_shouting():
    """MA lines are reference guides, not data — a violet one dominated the chart."""
    lines = re.findall(r"line\(ctx, chart\.ma\d+, X, Y, '(#[0-9a-fA-F]{6})'", SCRIPT)
    assert len(lines) == 2, "expected exactly two moving-average lines"
    for c in lines:
        hue, sat, lum = _hsl(c)
        assert not (240 <= hue <= 290), f"{c} is a violet reference line"


def test_there_is_only_one_accent_cyan():
    """Two different cyans read as a blue glow fighting a teal one."""
    assert "0,229,255" not in SOURCE, "the blue cyan is back alongside the teal one"


def test_the_scanline_overlay_is_neutral():
    assert "rgba(0,10,20" not in SOURCE, "the scanline is tinting the whole screen blue"
