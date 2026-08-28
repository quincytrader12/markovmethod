"""The sector taxonomy.

The scan groups used to be a muddle — "megacap" is a size, "energy" is a sector
and "crypto" is an asset class, all offered as if they were the same kind of
choice, with Real Estate, Materials and Utilities missing entirely and Alphabet
filed under "megacap" rather than Communication Services. These tests hold the
replacement to the standard eleven and keep the navigation honest about what it
does and does not know.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.sectors import (
    ALL_GROUPS,
    NON_EQUITY,
    SECTORS,
    STYLES,
    equity_universe,
    full_universe,
    sector_key,
    sector_name,
)
from markov_hedge_fund_method.web import SCAN_GROUPS, SCAN_UNIVERSE, AppState, create_app

ELEVEN = {"communication", "discretionary", "staples", "energy", "financials",
          "health", "industrials", "technology", "materials", "realestate",
          "utilities"}


def _client():
    return TestClient(create_app(AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)))


# ── the taxonomy is the standard one ────────────────────────────────────────
def test_all_eleven_sectors_are_present():
    assert set(SECTORS) == ELEVEN


def test_the_sectors_that_were_missing_now_exist():
    """Real Estate, Materials and Utilities had no representation at all."""
    for key in ("realestate", "materials", "utilities"):
        assert len(SECTORS[key]["symbols"]) >= 15


def test_every_sector_has_a_display_name_and_a_blurb():
    for key, grp in ALL_GROUPS.items():
        assert grp["name"] and grp["name"] != key
        assert grp["blurb"], f"{key} has no explanation"


def test_funds_and_crypto_are_not_filed_as_sectors():
    """An index fund holds every sector at once; crypto is not an equity."""
    assert set(NON_EQUITY) == {"etf", "crypto"}
    assert not (set(NON_EQUITY) & set(SECTORS))


def test_style_groups_are_kept_separate_from_sectors():
    """Size and style cut across sectors rather than sitting beside them."""
    assert set(STYLES) == {"megacap", "growth"}
    assert not (set(STYLES) & set(SECTORS))


# ── the classifications people will actually check ──────────────────────────
def test_alphabet_is_communication_services_not_technology():
    """The example that prompted this: Google belongs with Meta and Netflix."""
    assert sector_key("GOOGL") == "communication"
    assert sector_key("GOOG") == "communication"
    for peer in ("META", "NFLX", "DIS", "TMUS"):
        assert sector_key(peer) == "communication"


def test_payments_are_financials_not_technology():
    """GICS moved Visa and Mastercard out of tech; the map follows."""
    for sym in ("V", "MA", "PYPL"):
        assert sector_key(sym) == "financials"


def test_amazon_and_tesla_are_consumer_discretionary():
    for sym in ("AMZN", "TSLA", "SBUX", "HD", "MCD"):
        assert sector_key(sym) == "discretionary"


def test_staples_and_discretionary_are_split():
    assert sector_key("KO") == "staples" and sector_key("PG") == "staples"
    assert sector_key("NKE") == "discretionary"


def test_reits_land_in_real_estate():
    for sym in ("O", "AMT", "PLD", "EQIX", "SPG"):
        assert sector_key(sym) == "realestate"


def test_funds_and_crypto_resolve_to_their_own_sections():
    assert sector_key("SPY") == "etf" and sector_key("XLK") == "etf"
    assert sector_key("BTC-USD") == "crypto"


def test_a_symbol_outside_the_map_is_unclassified_not_guessed():
    """Alpaca publishes no sector field, so the sweep will turn up names this
    map has never heard of. Saying so beats inventing a classification."""
    assert sector_name("ZZZZ") == "" and sector_key("ZZZZ") == ""


def test_lookup_is_case_insensitive():
    assert sector_key("googl") == sector_key("GOOGL")


# ── the universes derived from it ───────────────────────────────────────────
def test_the_equity_universe_excludes_funds():
    """A fund's regime is the market's regime; a scan should surface names."""
    assert "SPY" not in SCAN_UNIVERSE and "QQQ" not in SCAN_UNIVERSE
    assert "BTC-USD" not in SCAN_UNIVERSE


def test_the_full_universe_includes_them():
    assert "SPY" in full_universe() and "BTC-USD" in full_universe()


def test_no_symbol_is_counted_twice():
    assert len(equity_universe()) == len(set(equity_universe()))
    assert len(full_universe()) == len(set(full_universe()))


def test_a_symbol_belongs_to_exactly_one_sector():
    seen: dict[str, str] = {}
    for key, grp in SECTORS.items():
        for sym in grp["symbols"]:
            assert sym not in seen, f"{sym} is in both {seen.get(sym)} and {key}"
            seen[sym] = key


def test_scan_groups_mirror_the_taxonomy():
    assert set(SCAN_GROUPS) == set(ALL_GROUPS)


def test_the_universe_is_substantially_wider_than_it_was():
    assert len(SCAN_UNIVERSE) > 350


# ── the navigation the UI is built from ─────────────────────────────────────
def test_the_groups_endpoint_serves_the_menu():
    d = _client().get("/api/groups").json()
    assert {g["key"] for g in d["sectors"]} == ELEVEN
    assert {g["key"] for g in d["other"]} == {"etf", "crypto"}
    assert {g["key"] for g in d["styles"]} == {"megacap", "growth"}
    assert all(g["count"] > 0 and g["name"] for g in d["sectors"])


def test_every_sector_can_actually_be_scanned():
    client = _client()
    for key in ELEVEN | {"etf", "crypto"}:
        d = client.get("/api/scan", params={"universe": key, "top": 3}).json()
        assert d["universe"] == key, f"{key} did not resolve"
        assert d["universeSize"] > 0 and d["results"], f"{key} returned nothing"


def test_scan_results_carry_their_sector():
    d = _client().get("/api/scan", params={"universe": "communication", "top": 5}).json()
    assert all(r["sector"] == "Communication Services" for r in d["results"])


def test_the_menu_is_built_from_the_server_not_hard_coded():
    """One definition of the taxonomy, so the menu cannot list a group the
    scanner has no idea how to scan."""
    html = _client().get("/").text
    assert "/api/groups" in html and "renderScanNav" in html
    assert "fillHeatScopes" in html, "the heatmap still has its own scope list"
