"""The page must never be served from a stale browser cache.

This is the bug that made a whole release look like nothing had shipped. The HUD
is served from the same localhost URL every time, and FileResponse sent an ETag
and a Last-Modified but no Cache-Control — which a browser reads as permission
to reuse its copy without asking. Upgrading the exe swapped the server and left
the previous build's interface on screen: new API, old page, no sign anything
had changed.

Also covered: sectors reaching every place that names an asset, since a
classification that only appears in the scanner is half a feature.
"""

from __future__ import annotations

import hashlib
import re

from fastapi.testclient import TestClient

from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.telegram import format_flip, format_scan
from markov_hedge_fund_method.web import AppState, create_app


def _client():
    return TestClient(create_app(AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)))


# ── the page is never cached ────────────────────────────────────────────────
def test_the_page_forbids_caching():
    r = _client().get("/")
    cc = r.headers.get("cache-control", "")
    assert "no-store" in cc, f"the page may be cached across upgrades: {cc!r}"


def test_the_page_carries_its_own_build_stamp():
    """A cached copy keeps the old stamp, which is how it can notice it is old."""
    body = _client().get("/").text
    assert re.search(r'window\.__PAGE_BUILD="[0-9a-f]{8}"', body)


def test_the_stamp_matches_what_the_server_reports():
    c = _client()
    stamp = re.search(r'window\.__PAGE_BUILD="([0-9a-f]{8})"', c.get("/").text).group(1)
    assert c.get("/api/version").json()["page"] == stamp


def test_the_stamp_tracks_the_page_contents():
    from markov_hedge_fund_method.web import STATIC_DIR
    import os

    with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
        expected = hashlib.sha256(f.read()).hexdigest()[:8]
    assert _client().get("/api/version").json()["page"] == expected


def test_the_version_endpoint_reports_what_is_running():
    d = _client().get("/api/version").json()
    assert d["version"] and d["groups"] >= 13 and d["universe"] > 350
    assert isinstance(d["frozen"], bool)


def test_the_page_shows_the_build_and_warns_when_stale():
    html = _client().get("/").text
    assert 'id="chip-build"' in html and "checkBuild" in html
    assert "stale" in html.lower(), "a stale page never tells the user"


# ── sectors reach everywhere an asset is named ──────────────────────────────
def test_search_results_carry_the_sector():
    d = _client().get("/api/search", params={"q": "goog"}).json()
    hit = next(r for r in d["results"] if r["symbol"] == "GOOGL")
    assert hit["sector"] == "Communication Services"


def test_the_default_search_list_carries_sectors():
    d = _client().get("/api/search", params={"q": ""}).json()
    assert any(r.get("sector") for r in d["results"])


def test_the_search_dropdown_renders_the_sector():
    html = _client().get("/").text
    assert 'o.sector' in html and 'class="sg"' in html


def test_the_telegram_digest_names_the_sector():
    """Three names from one sector is a concentrated bet, and that is invisible
    from tickers alone on a phone."""
    picks = [{"symbol": "GOOGL", "name": "Alphabet", "score": 90, "verdict": "Buy",
              "lastPrice": 170.0, "rationale": "x", "daysInRegime": 2,
              "sector": "Communication Services"}]
    text = format_scan(picks, min_score=50)
    assert "Communication Services" in text


def test_a_digest_without_a_sector_still_formats():
    picks = [{"symbol": "ZZZZ", "name": "Unknown", "score": 90, "verdict": "Buy",
              "lastPrice": 1.0, "rationale": "x", "daysInRegime": 1, "sector": ""}]
    text = format_scan(picks, min_score=50)
    assert "ZZZZ" in text


def test_a_regime_flip_alert_names_the_sector():
    text = format_flip({"symbol": "GOOGL", "from": "sideways", "to": "bull"})
    assert "Communication Services" in text


def test_a_flip_on_an_unclassified_symbol_still_formats():
    text = format_flip({"symbol": "ZZZZ", "from": "bull", "to": "bear"})
    assert "ZZZZ" in text and "BEAR" in text


def test_the_heatmap_and_scanner_share_the_sector_menu():
    html = _client().get("/").text
    assert "fillHeatScopes" in html and "renderScanNav" in html
    assert html.count("loadScanGroups") >= 2, "the two menus have separate sources again"


def test_the_scanner_nav_says_so_when_it_cannot_load():
    """An empty menu with no explanation reads as 'this build has no sectors',
    which is exactly the wrong conclusion."""
    html = _client().get("/").text
    assert "Sectors unavailable" in html


# ── scan progress and retry ─────────────────────────────────────────────────
def test_the_scanner_shows_a_progress_bar():
    html = _client().get("/").text
    assert "scanLoadingHTML" in html and 'class="pbar' in html


def test_only_the_full_sweep_gets_a_real_percentage():
    """A fake percentage on a twenty-minute job is worse than none, so the
    determinate bar is reserved for the one scope that actually knows."""
    html = _client().get("/").text
    assert "SCAN_U === 'full'" in html
    assert "indet" in html, "there is no indeterminate state for scopes without progress"


def test_progress_polling_stops_when_the_panel_closes():
    html = _client().get("/").text
    assert "function stopScanProgress" in html
    assert "closeScanner(){ stopScanProgress()" in html, "it keeps polling behind a closed panel"


def test_a_failed_scan_offers_a_retry():
    html = _client().get("/").text
    assert "scanErrorHTML" in html
    assert "runScan(true)" in html and "Retry" in html


def test_a_failed_scan_retries_itself_before_giving_up():
    """A blip on a background sweep is common and self-correcting."""
    html = _client().get("/").text
    assert "SCAN_MAX_RETRY" in html
    assert "runScan(force, attempt + 1)" in html


def test_a_superseded_scan_cannot_overwrite_a_newer_one():
    """Clicking three sectors quickly must not let the first response win."""
    html = _client().get("/").text
    assert "SCAN_SEQ" in html and "seq !== SCAN_SEQ" in html
