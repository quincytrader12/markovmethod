"""Full-market sweep: cover everything, hold almost nothing, yield to the user.

The sweep exists so nothing is left to chance — every tradable symbol gets
scored, not a list someone thought of. That ambition creates three ways to do
harm, and these tests pin all three down: it must not exhaust memory, it must
not make the terminal stutter, and it must never claim more coverage than it
has actually achieved.
"""

from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.sweep import CHUNK, KEEP, MarketSweep, is_common_stock
from markov_hedge_fund_method.web import (
    SCAN_ALL,
    SCAN_GROUPS,
    SCAN_UNIVERSE,
    AppState,
    create_app,
)


def _state():
    return AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)


# ── the universe actually got wider ─────────────────────────────────────────
def test_the_household_names_are_now_scannable():
    """The gap that started this: searchable and watchable, but never scanned."""
    for sym in ("SBUX", "MCD", "KO", "NKE", "DIS", "CAT", "XOM", "JPM", "UNH", "T"):
        assert sym in SCAN_UNIVERSE, f"{sym} is still invisible to the scanner"


def test_every_sector_has_single_name_coverage():
    """Sector ETFs standing in for whole sectors was the underlying problem."""
    for group in ("communication", "discretionary", "staples", "energy",
                  "financials", "health", "industrials", "technology",
                  "materials", "realestate", "utilities"):
        assert group in SCAN_GROUPS and len(SCAN_GROUPS[group]) >= 15


def test_the_curated_universe_grew_substantially():
    assert len(SCAN_UNIVERSE) > 300, "the curated sweep is still a shortlist"


def test_the_universe_has_no_duplicates():
    """A name in two groups must be fetched and scored once, not twice."""
    assert len(SCAN_UNIVERSE) == len(set(SCAN_UNIVERSE))
    assert len(SCAN_ALL) == len(set(SCAN_ALL))


def test_crypto_is_excluded_from_the_equity_sweep_but_present_in_all():
    assert "BTC-USD" not in SCAN_UNIVERSE
    assert "BTC-USD" in SCAN_ALL


# ── the asset filter ────────────────────────────────────────────────────────
def test_common_stock_filter_keeps_ordinary_tickers():
    assert is_common_stock("SBUX", "Starbucks Corporation")
    assert is_common_stock("F", "Ford Motor Company")


def test_common_stock_filter_drops_warrants_units_and_rights():
    assert not is_common_stock("ABCDW", "Acme Corp Warrant")
    assert not is_common_stock("ABCDU", "Acme Corp Unit")
    assert not is_common_stock("ABCDR", "Acme Corp Rights")
    assert not is_common_stock("ABCP", "Acme 6.5% Preferred Series A")


def test_common_stock_filter_drops_odd_symbols():
    assert not is_common_stock("BRK.B", "Berkshire Hathaway")
    assert not is_common_stock("TOOLONG", "Something")
    assert not is_common_stock("", "")


# ── memory discipline ───────────────────────────────────────────────────────
def test_a_scored_chunk_does_not_stay_in_the_price_cache():
    """Ten years of bars for the whole market is most of a gigabyte."""
    state = _state()
    state.sweep.run_chunk()
    assert len(state._ohlc_cache) <= 12, (
        f"the sweep is hoarding {len(state._ohlc_cache)} price frames")


def test_the_leaderboard_is_capped():
    state = _state()
    sweep = state.sweep
    for i in range(KEEP + 200):
        sweep.board[f"S{i}"] = {"symbol": f"S{i}", "score": i}
    sweep._trim()
    assert len(sweep.board) == KEEP


def test_trimming_keeps_the_best_rows_not_arbitrary_ones():
    state = _state()
    sweep = state.sweep
    for i in range(KEEP + 50):
        sweep.board[f"S{i}"] = {"symbol": f"S{i}", "score": i}
    sweep._trim()
    assert min(r["score"] for r in sweep.board.values()) == 50


def test_the_symbol_being_looked_at_is_never_evicted():
    """Evicting the chart the user is staring at would refetch it instantly."""
    state = _state()
    state.state_payload("SPY")
    assert "SPY" in state._ohlc_cache
    state.sweep._evict(["SPY", "AAPL"])
    assert "SPY" in state._ohlc_cache, "the focused symbol must survive eviction"


# ── coverage is stated, never implied ───────────────────────────────────────
def test_status_reports_real_progress():
    state = _state()
    st = state.sweep.status()
    for key in ("running", "universeSize", "cursor", "progress", "cycle",
                "scanned", "boardSize", "chunk"):
        assert key in st
    assert st["cycle"] == 0 and st["scanned"] == 0


def test_progress_advances_with_the_cursor():
    state = _state()
    before = state.sweep.status()["scanned"]
    state.sweep.run_chunk()
    after = state.sweep.status()
    assert after["scanned"] > before
    assert 0.0 <= after["progress"] <= 1.0


def test_a_full_pass_wraps_and_counts_a_cycle():
    state = _state()
    sweep = state.sweep
    sweep.universe = ["SPY", "QQQ"]
    sweep.cursor = 0
    sweep.run_chunk()
    assert sweep.cycle == 1 and sweep.cursor == 0


def test_the_scan_endpoint_admits_partial_coverage():
    client = TestClient(create_app(_state()))
    d = client.get("/api/scan", params={"universe": "full", "top": 5}).json()
    assert d["universe"] == "full"
    assert "sweep" in d, "a partial sweep must say so rather than look complete"
    assert "progress" in d["sweep"] and "cycle" in d["sweep"]


def test_the_ui_shows_how_much_of_the_market_has_been_covered():
    client = TestClient(create_app(_state()))
    html = client.get("/").text
    assert "sweepNote" in html and "full-market sweep" in html
    # The scope buttons are built from /api/groups now, not written into markup.
    assert "btn('full'" in html and "renderScanNav" in html


# ── it must not make the terminal stutter ───────────────────────────────────
def test_the_sweep_stands_aside_while_the_user_is_active():
    state = _state()
    state.note_activity()
    t0 = time.monotonic()
    state.sweep.yield_to_user(timeout=1.0)
    assert time.monotonic() - t0 >= 0.2, "it did not wait for the user at all"


def test_the_sweep_works_freely_when_the_user_is_idle():
    state = _state()
    state.last_activity = time.monotonic() - 600
    t0 = time.monotonic()
    state.sweep.yield_to_user(timeout=5.0)
    assert time.monotonic() - t0 < 0.2, "it waited despite an idle terminal"


def test_a_busy_session_cannot_starve_the_sweep_forever():
    state = _state()
    stop = threading.Event()

    def churn():
        while not stop.is_set():
            state.note_activity()
            time.sleep(0.02)

    th = threading.Thread(target=churn, daemon=True)
    th.start()
    try:
        t0 = time.monotonic()
        state.sweep.yield_to_user(timeout=0.6)
        waited = time.monotonic() - t0
        assert waited < 2.0, "the timeout did not release it"
    finally:
        stop.set(); th.join(timeout=2)


def test_status_polling_does_not_count_as_user_activity():
    """The page polls progress on a timer. If that counted, the terminal would
    always look busy and the sweep would never run."""
    state = _state()
    client = TestClient(create_app(state))
    state.last_activity = time.monotonic() - 600
    client.get("/api/sweep")
    assert state.idle_for() > 1.0, "a status poll suppressed the sweep"


def test_a_real_request_does_count_as_activity():
    state = _state()
    client = TestClient(create_app(state))
    state.last_activity = time.monotonic() - 600
    client.get("/api/state", params={"symbol": "SPY"})
    assert state.idle_for() < 1.0


def test_the_sweep_survives_a_broken_chunk():
    state = _state()
    sweep = MarketSweep(state)
    sweep.universe = ["SPY"]
    state.prefetch_ohlc = lambda syms: (_ for _ in ()).throw(RuntimeError("api down"))
    try:
        sweep.run_chunk()
    except RuntimeError:
        pass                                   # the loop catches it; see below
    sweep._stop.set()
    sweep._loop()                              # must return, not raise
    assert sweep.last_error is None or "api down" in str(sweep.last_error)


# ── the alert path reads the sweep ──────────────────────────────────────────
def test_the_watcher_defaults_to_the_whole_market():
    from markov_hedge_fund_method.watcher import DEFAULTS
    assert DEFAULTS["scanUniverse"] == "full"


def test_the_watcher_reads_the_sweep_leaderboard():
    state = _state()
    state.sweep.board = {"AAA": {"symbol": "AAA", "score": 99, "dsr": 0.99,
                                 "verdict": "buy", "daysInRegime": 2}}
    picks = state.watcher.candidates({**state.watcher.config(), "scanUniverse": "full",
                                      "scanMinDsr": 0.9, "scanMinScore": 50,
                                      "scanFreshDays": 0})
    assert [p["symbol"] for p in picks] == ["AAA"]


# ── batch fetching ──────────────────────────────────────────────────────────
def test_prefetch_is_a_no_op_in_demo_mode():
    state = _state()
    assert state.prefetch_ohlc(["SPY", "QQQ"])["fetched"] == 0


def test_prefetch_counts_what_it_already_has():
    state = _state()
    state.demo = False
    state.settings = Settings(ticker="SPY", mode=Mode.PAPER, api_key="k", api_secret="s")
    from markov_hedge_fund_method.market_data import synthetic_ohlc
    state._ohlc_cache["SPY"] = (time.monotonic(), synthetic_ohlc(300), "live")

    import markov_hedge_fund_method.web as web
    calls = {}

    def fake_batch(symbols, *a, **k):
        calls["symbols"] = list(symbols)
        return {}

    original = getattr(web, "batch_alpaca_ohlc", None)
    import markov_hedge_fund_method.market_data as md
    md_orig = md.batch_alpaca_ohlc
    md.batch_alpaca_ohlc = fake_batch
    try:
        stats = state.prefetch_ohlc(["SPY", "AAPL", "MSFT"])
    finally:
        md.batch_alpaca_ohlc = md_orig
    assert stats["cached"] == 1, "an already-cached symbol must not be refetched"
    assert set(calls["symbols"]) == {"AAPL", "MSFT"}


def test_a_batch_failure_leaves_the_per_symbol_path_working():
    state = _state()
    state.demo = False
    state.settings = Settings(ticker="SPY", mode=Mode.PAPER, api_key="k", api_secret="s")
    import markov_hedge_fund_method.market_data as md
    orig = md.batch_alpaca_ohlc
    md.batch_alpaca_ohlc = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
    try:
        stats = state.prefetch_ohlc(["AAPL"])
    finally:
        md.batch_alpaca_ohlc = orig
    assert stats["fetched"] == 0, "a failed prefetch must be slow, never wrong"


# ── measuring the network half ──────────────────────────────────────────────
def test_benchmark_says_it_cannot_measure_without_a_connection():
    """Zeros would read as an infinitely fast network rather than no network."""
    state = _state()
    b = state.sweep.benchmark(sample=10)
    assert b["ok"] is False
    assert "connect" in b["reason"].lower()
    assert b.get("secondsPerSymbol") is None


def test_benchmark_still_reports_the_knowable_arithmetic_offline():
    """Page counts follow from the universe size and history window alone."""
    state = _state()
    b = state.sweep.benchmark(sample=10)
    assert b["universeSize"] > 300
    assert b["projectedPagesFullPass"] >= 1


def test_benchmark_measures_a_real_cycle_when_connected(tmp_path):
    state = _state()
    state.demo = False
    state.settings = Settings(ticker="SPY", mode=Mode.PAPER, api_key="k", api_secret="s")
    from markov_hedge_fund_method.market_data import synthetic_ohlc
    from markov_hedge_fund_method.pricestore import PriceStore

    # Its own store: otherwise the benchmark reads whatever a previous run left
    # behind, and the test's result depends on the order it happened to run in.
    state.prices = PriceStore(config_dir=str(tmp_path))

    import markov_hedge_fund_method.market_data as md
    orig = md.batch_alpaca_ohlc
    md.batch_alpaca_ohlc = lambda syms, *a, **k: {s: synthetic_ohlc(700, seed=1) for s in syms}
    try:
        b = state.sweep.benchmark(sample=6)
    finally:
        md.batch_alpaca_ohlc = orig
    assert b["ok"] is True
    assert b["scored"] >= 1
    assert b["secondsPerSymbol"] > 0 and b["symbolsPerSecond"] > 0
    assert b["barsDownloaded"] >= 700
    assert b["projectedFullPassMin"] > 0


def test_benchmark_does_not_leave_its_sample_in_the_cache():
    """A measurement must not quietly become a memory leak."""
    state = _state()
    state.demo = False
    state.settings = Settings(ticker="SPY", mode=Mode.PAPER, api_key="k", api_secret="s")
    from markov_hedge_fund_method.market_data import synthetic_ohlc
    import markov_hedge_fund_method.market_data as md
    orig = md.batch_alpaca_ohlc
    md.batch_alpaca_ohlc = lambda syms, *a, **k: {s: synthetic_ohlc(700, seed=1) for s in syms}
    try:
        state.sweep.benchmark(sample=8)
    finally:
        md.batch_alpaca_ohlc = orig
    assert len(state._ohlc_cache) <= 4


def test_a_shorter_history_window_is_costed_out():
    """The history window is the one lever that moves network cost, so the
    benchmark has to price it rather than leave it to be guessed at."""
    state = _state()
    b = state.sweep.benchmark(sample=10)
    assert b["projectedPagesAt3y"] < b["projectedPagesFullPass"]
    assert b["projectedPagesAt5y"] < b["projectedPagesFullPass"]
    assert b["projectedPagesAt3y"] < b["projectedPagesAt5y"]


def test_the_page_size_matches_alpacas_real_limit():
    """The whole request estimate rests on this number being right."""
    from markov_hedge_fund_method.sweep import MarketSweep as MS
    assert MS.PAGE_BARS == 10_000


def test_the_module_does_not_overstate_the_batching_win():
    """An earlier version of this claimed the full market cost 'a few dozen
    requests'. It costs thousands; batching is a 4x saving, not 240x."""
    import markov_hedge_fund_method.sweep as sw
    doc = sw.__doc__ or ""
    assert "few dozen requests" not in doc
    assert "4x" in doc or "2,772" in doc


def test_the_benchmark_endpoint_bounds_its_sample():
    client = TestClient(create_app(_state()))
    assert client.post("/api/sweep/benchmark", params={"sample": 100000}).status_code == 200
    assert client.post("/api/sweep/benchmark", params={"sample": 0}).status_code == 200


# ── everything downstream of the scanner picked the upgrade up ──────────────
def test_the_heatmap_offers_every_scan_group():
    """The heatmap had its own stale scope list and would have kept showing the
    old five groups after the universe tripled."""
    client = TestClient(create_app(_state()))
    html = client.get("/").text
    # Both menus are filled from the one taxonomy, so the check is that they
    # share a source rather than that each option is written out by hand.
    assert "fillHeatScopes" in html and 'id="heat-sectors"' in html
    assert '<option value="full"' in html, "heatmap cannot show the sweep board"


def test_the_heatmap_resolves_the_new_groups():
    client = TestClient(create_app(_state()))
    for scope in ("communication", "realestate", "utilities"):
        d = client.get("/api/heatmap", params={"view": "regime", "scope": scope}).json()
        assert d["shown"] > 10, f"{scope} resolved to nothing"


def test_the_heatmap_admits_when_it_is_showing_a_subset():
    """Silently showing 80 of 354 looks like the whole set."""
    client = TestClient(create_app(_state()))
    d = client.get("/api/heatmap", params={"view": "regime", "scope": "market"}).json()
    assert d["requested"] > d["shown"]
    assert d["truncated"] is True and d["cap"] == d["shown"]


def test_a_small_scope_is_not_reported_as_truncated():
    client = TestClient(create_app(_state()))
    d = client.get("/api/heatmap", params={"view": "regime", "scope": "energy"}).json()
    assert d["truncated"] is False


def test_the_ui_surfaces_the_truncation():
    client = TestClient(create_app(_state()))
    html = client.get("/").text
    assert "heatCoverage" in html
    assert html.count("heatCoverage(LAST_HEAT)") == 3, "not every view reports coverage"


def test_the_full_scope_heatmap_uses_the_ranked_board():
    """Truncating a ranked board keeps the interesting names; truncating an
    alphabetical list keeps whatever starts with A."""
    state = _state()
    state.sweep.board = {f"S{i}": {"symbol": f"S{i}", "score": i} for i in range(200)}
    client = TestClient(create_app(state))
    d = client.get("/api/heatmap", params={"view": "regime", "scope": "full"}).json()
    assert d["requested"] <= 80


def test_telegram_can_push_the_full_market_board():
    """The manual Telegram push was capped at the curated list."""
    import inspect
    import markov_hedge_fund_method.web as web
    src = inspect.getsource(web.create_app)
    assert 'if universe == "full":' in src and "state.sweep.results(200)" in src


def test_startup_no_longer_prewarms_the_curated_universe():
    """It cost 17s of background CPU and took page loads from 4.8ms to 76ms,
    and the sweep already covers the same ground while yielding to the user."""
    import inspect
    import markov_hedge_fund_method.web as web
    src = inspect.getsource(web.main)
    assert 'state.prewarm("market"' not in src
    assert "state.sweep.start()" in src, "nothing replaced it"


# ── the liquidity screen ────────────────────────────────────────────────────
def test_illiquid_and_new_listings_are_skipped_before_scoring():
    from markov_hedge_fund_method.market_data import synthetic_ohlc
    sweep = _state().sweep
    assert sweep.tradable_enough(synthetic_ohlc(600)) is True
    assert sweep.tradable_enough(synthetic_ohlc(600) * 0.001) is False, "penny stock"
    assert sweep.tradable_enough(synthetic_ohlc(50)) is False, "too little history"


def test_missing_data_is_not_treated_as_illiquid():
    """Absence of evidence is not evidence — it must fall through to the scorer,
    or demo mode and the per-symbol path would score nothing at all."""
    assert _state().sweep.tradable_enough(None) is True


def test_the_sweep_reports_what_it_skipped():
    state = _state()
    state.sweep.run_chunk()
    assert "skipped" in state.sweep.status()


# ── OTC listings: tradable, but no bars are ever published for them ─────────
def test_otc_exchanges_are_recognised():
    from markov_hedge_fund_method.sweep import is_otc
    assert is_otc("OTC") and is_otc("otc") and is_otc("PINK")
    assert not is_otc("NASDAQ") and not is_otc("NYSE") and not is_otc("")


def test_otc_listings_are_kept_out_of_the_sweep():
    """The bug from live use: every one was a guaranteed failed fetch, retried
    on every pass forever."""
    class _B: pass
    state = _state()
    state.broker = _B()
    otc = ["AMKBY", "AMIGY", "ALZIF", "ADDYY"]
    good = ["AAPL", "MSFT"]
    state._alpaca_symbols = set(otc + good)
    state._alpaca_names = {s: "ADR" for s in otc} | {s: "Co" for s in good}
    state._alpaca_exchange = {s: "OTC" for s in otc} | {s: "NASDAQ" for s in good}

    uni = state.sweep.build_universe()
    assert sorted(uni) == ["AAPL", "MSFT"]
    assert not any(s in uni for s in otc)


def test_a_symbol_with_no_data_is_not_rediscovered():
    class _B: pass
    state = _state()
    state.broker = _B()
    state._alpaca_symbols = {"AAPL", "GHOST"}
    state._alpaca_names = {"AAPL": "Apple", "GHOST": "Ghost Co"}
    state._alpaca_exchange = {"AAPL": "NASDAQ", "GHOST": "NASDAQ"}
    assert "GHOST" in state.sweep.build_universe()

    state.sweep.no_data.add("GHOST")
    assert "GHOST" not in state.sweep.build_universe()


def test_missing_exchange_data_does_not_exclude_a_symbol():
    """Absence of an exchange field is not evidence of an OTC listing."""
    class _B: pass
    state = _state()
    state.broker = _B()
    state._alpaca_symbols = {"AAPL"}
    state._alpaca_names = {"AAPL": "Apple"}
    state._alpaca_exchange = {}
    assert state.sweep.build_universe() == ["AAPL"]


def test_the_sweep_reports_how_many_symbols_have_no_data():
    state = _state()
    state.sweep.no_data.update({"A", "B", "C"})
    assert state.sweep.status()["noData"] == 3
