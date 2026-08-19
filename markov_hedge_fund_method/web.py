"""Mamba Terminal — web HUD backend (`mamba-web`).

A small FastAPI app that serves the neon dashboard and feeds it live data:

    GET  /                     the HUD page
    GET  /api/config           mode / account / demo flags
    GET  /api/symbols          default watchlist
    GET  /api/state?symbol=SPY regime, matrix, signal, greed/fear, chart series
    GET  /api/portfolio        equity, positions, open orders (broker)
    GET  /api/accounts         list account profiles
    POST /api/accounts         add a profile {name,key,secret,paper}
    POST /api/accounts/active  switch active profile {name}
    DEL  /api/accounts/{name}  remove a profile
    POST /api/orders           submit any order style (full taxonomy)
    POST /api/orders/cancel_all cancel every open order

Reuses the existing engine/regime/orders/broker/accounts modules, so the web UI
and the terminal share one brain. `main()` launches uvicorn and opens a browser.

NOTE: this module deliberately does NOT use `from __future__ import annotations`.
FastAPI resolves endpoint parameter types from real annotation objects; stringized
annotations (from that future import) break body/Request detection here.
"""

import os
import time
from dataclasses import replace

from .accounts import AccountStore
from .alerts import AlertEngine
from .broker import ReadOnlyError, make_broker
from .config import Mode, Settings, load_settings
from .journal import JournalStore
from .market_data import (
    get_history,
    get_intraday_ohlc,
    get_ohlc,
    synthetic_close,
    synthetic_intraday,
    synthetic_ohlc,
)
from .markov2 import Strategy
from .news import fetch_news
from .orders import OrderTicket, OrderValidationError
from .telegram import TelegramError, TelegramNotifier, format_flip, format_scan
from .webstate import _rsi, market_state, quote_state

STATIC_DIR = os.path.join(os.path.dirname(__file__), "web_static")
DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT", "BTC-USD"]

# A small bundled universe so the search box has suggestions offline. Typing any
# ticker not in this list still works — it's used verbatim.
SYMBOL_UNIVERSE = sorted(set(DEFAULT_SYMBOLS + [
    "AMZN", "GOOGL", "GOOG", "META", "NFLX", "AMD", "INTC", "MU", "AVGO", "QCOM",
    "CRM", "ORCL", "ADBE", "CSCO", "IBM", "TXN", "NOW", "SHOP", "UBER", "ABNB",
    "PLTR", "SNOW", "COIN", "SQ", "PYPL", "V", "MA", "JPM", "BAC", "WFC", "GS",
    "MS", "C", "BRK.B", "BX", "SCHW", "KO", "PEP", "MCD", "SBUX", "NKE", "DIS",
    "WMT", "COST", "TGT", "HD", "LOW", "PG", "JNJ", "PFE", "MRK", "ABBV", "LLY",
    "UNH", "CVS", "XOM", "CVX", "COP", "OXY", "BA", "CAT", "GE", "F", "GM",
    "T", "VZ", "TMUS", "DAL", "AAL", "UAL", "RIVN", "LCID", "NIO", "SOFI",
    "DKNG", "ROKU", "ZM", "DOCU", "TWLO", "NET", "DDOG", "CRWD", "ZS", "PANW",
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "ARKK", "XLF", "XLK", "XLE", "GLD",
    "SLV", "USO", "TLT", "HYG", "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD",
]))

# A liquid subset the opportunity scanner sweeps by default. Kept modest so a
# scan stays responsive (each symbol runs the full regime + walk-forward brain).
# Hunting grounds for the opportunity scanner. "market" is deliberately wide and
# not just mega-caps — the obvious names are already picked over, so the mid-cap
# and thematic groups are where a regime flip is more likely to be early.
SCAN_GROUPS = {
    "megacap": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX",
        "COST", "JPM", "V", "UNH", "LLY", "XOM", "HD", "WMT", "PG", "MA", "ORCL",
    ],
    "midcap": [
        "PLTR", "COIN", "CRWD", "SNOW", "DDOG", "NET", "ZS", "PANW", "MDB", "TEAM",
        "HOOD", "SOFI", "AFRM", "TOST", "RBLX", "DKNG", "ABNB", "UBER", "LYFT", "SHOP",
        "SQ", "TTD", "ROKU", "PINS", "SNAP", "U", "PATH", "S", "OKTA", "TWLO",
        "ETSY", "CHWY", "CVNA", "W", "WBD", "F", "GM", "RIVN", "LCID", "PLUG",
        "ENPH", "FSLR", "RUN", "CHPT", "AI", "IONQ", "RKLB", "ASTS", "SMCI", "ARM",
    ],
    "sector": [
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
        "SMH", "IBB", "XBI", "ITB", "XRT", "XOP", "JETS", "KRE", "GDX", "URA",
        "ARKK", "IWM", "DIA", "SPY", "QQQ", "EEM", "EFA", "TLT", "HYG", "GLD",
    ],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
}

# The default sweep: everything except crypto — ~100 names, wide enough that the
# picks are not just the same ten mega-caps every day.
SCAN_UNIVERSE = (SCAN_GROUPS["megacap"] + SCAN_GROUPS["midcap"] + SCAN_GROUPS["sector"])


# Full names for the bundled universe (tooltips). When connected to Alpaca the
# real asset names are used for every symbol; this covers the offline case.
ASSET_NAMES = {
    "SPY": "SPDR S&P 500 ETF Trust", "QQQ": "Invesco QQQ Trust (Nasdaq-100)",
    "IWM": "iShares Russell 2000 ETF", "DIA": "SPDR Dow Jones Industrial Average ETF",
    "VOO": "Vanguard S&P 500 ETF", "VTI": "Vanguard Total Stock Market ETF",
    "ARKK": "ARK Innovation ETF", "XLF": "Financial Select Sector SPDR",
    "XLK": "Technology Select Sector SPDR", "XLE": "Energy Select Sector SPDR",
    "GLD": "SPDR Gold Shares", "SLV": "iShares Silver Trust", "USO": "United States Oil Fund",
    "TLT": "iShares 20+ Year Treasury Bond ETF", "HYG": "iShares iBoxx High Yield Corp Bond ETF",
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.", "GOOGL": "Alphabet Inc. (Class A)", "GOOG": "Alphabet Inc. (Class C)",
    "META": "Meta Platforms, Inc.", "TSLA": "Tesla, Inc.", "NFLX": "Netflix, Inc.",
    "AMD": "Advanced Micro Devices, Inc.", "INTC": "Intel Corporation", "MU": "Micron Technology, Inc.",
    "AVGO": "Broadcom Inc.", "QCOM": "QUALCOMM Incorporated", "CRM": "Salesforce, Inc.",
    "ORCL": "Oracle Corporation", "ADBE": "Adobe Inc.", "CSCO": "Cisco Systems, Inc.",
    "IBM": "International Business Machines", "TXN": "Texas Instruments Incorporated",
    "NOW": "ServiceNow, Inc.", "SHOP": "Shopify Inc.", "UBER": "Uber Technologies, Inc.",
    "ABNB": "Airbnb, Inc.", "PLTR": "Palantir Technologies Inc.", "SNOW": "Snowflake Inc.",
    "COIN": "Coinbase Global, Inc.", "SQ": "Block, Inc.", "PYPL": "PayPal Holdings, Inc.",
    "V": "Visa Inc.", "MA": "Mastercard Incorporated", "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corporation", "WFC": "Wells Fargo & Company", "GS": "Goldman Sachs Group",
    "MS": "Morgan Stanley", "C": "Citigroup Inc.", "BRK.B": "Berkshire Hathaway (Class B)",
    "BX": "Blackstone Inc.", "SCHW": "Charles Schwab Corporation", "KO": "The Coca-Cola Company",
    "PEP": "PepsiCo, Inc.", "MCD": "McDonald's Corporation", "SBUX": "Starbucks Corporation",
    "NKE": "NIKE, Inc.", "DIS": "The Walt Disney Company", "WMT": "Walmart Inc.",
    "COST": "Costco Wholesale Corporation", "TGT": "Target Corporation", "HD": "The Home Depot, Inc.",
    "LOW": "Lowe's Companies, Inc.", "PG": "The Procter & Gamble Company", "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer Inc.", "MRK": "Merck & Co., Inc.", "ABBV": "AbbVie Inc.", "LLY": "Eli Lilly and Company",
    "UNH": "UnitedHealth Group Incorporated", "CVS": "CVS Health Corporation",
    "XOM": "Exxon Mobil Corporation", "CVX": "Chevron Corporation", "COP": "ConocoPhillips",
    "OXY": "Occidental Petroleum", "BA": "The Boeing Company", "CAT": "Caterpillar Inc.",
    "GE": "General Electric Company", "F": "Ford Motor Company", "GM": "General Motors Company",
    "T": "AT&T Inc.", "VZ": "Verizon Communications Inc.", "TMUS": "T-Mobile US, Inc.",
    "DAL": "Delta Air Lines, Inc.", "AAL": "American Airlines Group", "UAL": "United Airlines Holdings",
    "RIVN": "Rivian Automotive, Inc.", "LCID": "Lucid Group, Inc.", "NIO": "NIO Inc.",
    "SOFI": "SoFi Technologies, Inc.", "DKNG": "DraftKings Inc.", "ROKU": "Roku, Inc.",
    "ZM": "Zoom Video Communications", "DOCU": "DocuSign, Inc.", "TWLO": "Twilio Inc.",
    "NET": "Cloudflare, Inc.", "DDOG": "Datadog, Inc.", "CRWD": "CrowdStrike Holdings",
    "ZS": "Zscaler, Inc.", "PANW": "Palo Alto Networks, Inc.", "A": "Agilent Technologies, Inc.",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana", "DOGE-USD": "Dogecoin",
}


def _seed(symbol: str) -> int:
    return sum(ord(ch) for ch in symbol) % 997


class AppState:
    """Mutable app state for the single local user."""

    # Recomputing the walk-forward backtest per symbol is the slow part; cache
    # the finished payload briefly so flipping between assets is instant.
    CACHE_TTL = 3600.0  # demo data is deterministic; refreshed by version bumps

    def __init__(self, settings: Settings, demo: bool = False):
        self.settings = settings
        self.demo = demo
        self.accounts = AccountStore()
        self.journal = JournalStore()
        self.alerts = AlertEngine()
        self.telegram = TelegramNotifier()
        from .watcher import ScanWatcher
        self.watcher = ScanWatcher(self)
        from .healing import Healer
        self.healer = Healer(self)
        self.broker = None if demo else make_broker(settings)
        self._state_cache: dict[str, tuple[float, dict]] = {}
        self._regperf_cache: dict[str, tuple[float, dict]] = {}
        self._scan_cache: dict[str, tuple[float, list]] = {}
        self._ohlc_cache: dict[str, tuple[float, object, str]] = {}
        self._state_vol_cache: dict[str, tuple[float, dict]] = {}
        self._state_meta_cache: dict[str, tuple[float, dict]] = {}
        self._heat_cache: dict[str, tuple[float, dict]] = {}
        self._meta_size_cache: dict[str, tuple[float, dict]] = {}
        self._news_cache: dict[str, tuple[float, list]] = {}
        self._alpaca_symbols: set[str] | None = None
        self._alpaca_names: dict[str, str] = {}
        self._alpaca_sorted: list[str] = []
        self._alpaca_index: dict[str, list[str]] = {}
        import threading as _threading
        self._universe_lock = _threading.Lock()
        self.ttl = self.CACHE_TTL if demo else 60.0  # live data goes stale sooner

    def reconnect(self) -> None:
        self.broker = None if self.demo else make_broker(self.settings)
        self._state_cache.clear()   # a new account may change the data source
        self._state_vol_cache.clear()
        self._state_meta_cache.clear()
        self._heat_cache.clear()
        self._meta_size_cache.clear()
        self._scan_cache.clear()
        self._regperf_cache.clear()
        self._ohlc_cache.clear()
        self._alpaca_symbols = None  # and a different asset universe
        self._alpaca_names = {}
        self._alpaca_sorted = []
        self._alpaca_index = {}
        self.prewarm_universe()      # reload in the background, never blocking

    def _ensure_alpaca_universe(self) -> None:
        """Fetch Alpaca's tradable assets once and build the search index.

        Alpaca returns tens of thousands of assets, so this is slow — callers
        that must not block (search-as-you-type) use the non-blocking helpers
        below and fall back to the bundled list until it is ready.
        """
        if self.broker is None or self._alpaca_symbols is not None:
            return
        with self._universe_lock:
            if self._alpaca_symbols is not None:      # won the race elsewhere
                return
            try:
                assets = self.broker.list_tradable_assets()  # [{symbol, name}]
                symbols = {a["symbol"] for a in assets}
                names = {a["symbol"]: a["name"] for a in assets if a.get("name")}
            except Exception:  # noqa: BLE001 — cache empty to avoid refetch storms
                symbols, names = set(), {}
            # Precompute once: sorted list + first-letter buckets, so a keystroke
            # is a dict hit over a few hundred names instead of a full re-scan.
            ordered = sorted(symbols)
            index: dict[str, list[str]] = {}
            for sym in ordered:
                index.setdefault(sym[:1], []).append(sym)
            self._alpaca_names = names
            self._alpaca_sorted = ordered
            self._alpaca_index = index
            self._alpaca_symbols = symbols       # set last — it is the ready flag

    def prewarm_universe(self) -> None:
        """Load the Alpaca asset universe in the background so the first
        keystroke in the search box never waits on the network."""
        import threading

        if self.broker is None or self._alpaca_symbols is not None:
            return

        def work():
            try:
                self._ensure_alpaca_universe()
            except Exception:  # noqa: BLE001 — never break startup
                pass

        threading.Thread(target=work, daemon=True).start()

    def universe_ready(self) -> bool:
        return self._alpaca_symbols is not None

    def alpaca_symbols(self) -> set[str] | None:
        """Set of tradable Alpaca symbols when connected (cached), else None."""
        if self.broker is None:
            return None
        self._ensure_alpaca_universe()
        return self._alpaca_symbols or None

    def name_for(self, symbol: str) -> str:
        """Full asset name — Alpaca's when loaded, else the bundled list.

        Never blocks: while the universe is still loading we just fall back.
        """
        symbol = symbol.upper()
        name = self._alpaca_names.get(symbol)
        if name:
            return name
        return ASSET_NAMES.get(symbol, "")

    def search_symbols(self, q: str, limit: int = 30) -> tuple[list[str], bool]:
        """Prefix-then-substring match. Non-blocking: uses the Alpaca index when
        it is ready, otherwise the bundled universe. Returns (symbols, live)."""
        if self.broker is not None:
            self.prewarm_universe()               # kick off load, don't wait
        live = self.universe_ready() and bool(self._alpaca_symbols)
        if live:
            starts = self._alpaca_index.get(q[:1], [])
            hits = [s for s in starts if s.startswith(q)][:limit]
            if len(hits) < limit:                 # top up with substring matches
                seen = set(hits)
                for s in self._alpaca_sorted:
                    if q in s and s not in seen:
                        hits.append(s)
                        if len(hits) >= limit:
                            break
            return hits, True
        universe = SYMBOL_UNIVERSE
        starts = [s for s in universe if s.startswith(q)]
        contains = [s for s in universe if q in s and s not in starts]
        return (starts + contains)[:limit], False

    def search_universe(self) -> list[str]:
        al = self.alpaca_symbols()
        return self._alpaca_sorted if al else SYMBOL_UNIVERSE

    # Raw price history, cached per symbol. Adding a symbol used to download its
    # history twice — once for the watchlist quote and again for the chart. Both
    # now share this one fetch, which is the bulk of the "adding is slow" wait.
    OHLC_TTL = 300.0

    def ohlc_for(self, symbol: str):
        """OHLC frame for a symbol. Returns (frame, source) where source is
        'live', 'synthetic (demo)' or 'synthetic (data unavailable)'.

        Callers MUST honour the source: a synthetic frame is fabricated data and
        must never be presented as though it described the real asset.
        """
        symbol = symbol.upper()
        cached = self._ohlc_cache.get(symbol)
        if cached and (time.monotonic() - cached[0]) < self.OHLC_TTL:
            # A cached *failure* is not the same as a cached success. Serving
            # fabricated data for the full TTL because one fetch failed leaves
            # the symbol broken long after the feed has come back, so a degraded
            # symbol gets its own short retry schedule and heals on its own.
            if not (cached[2].startswith("synthetic") and not self.demo
                    and self.healer.should_retry(symbol)):
                return cached[1], cached[2]
        if self.demo:
            df, source = synthetic_ohlc(seed=_seed(symbol)), "synthetic (demo)"
        else:
            try:
                df, source = get_ohlc(replace(self.settings, ticker=symbol)), "live"
                self.healer.note_fetch(symbol, True)
            except Exception as exc:  # noqa: BLE001 — never leave the HUD blank
                df, source = synthetic_ohlc(seed=_seed(symbol)), "synthetic (data unavailable)"
                self.healer.note_fetch(symbol, False, str(exc))
        self._ohlc_cache[symbol] = (time.monotonic(), df, source)
        if source == "live":
            # A recovered symbol must drop the payloads built from fake prices.
            for c in (self._state_cache, self._state_vol_cache, self._state_meta_cache):
                c.pop(symbol, None)
        return df, source

    def close_for(self, symbol: str):
        """Close series for a symbol — derived from the shared OHLC fetch."""
        df, source = self.ohlc_for(symbol)
        return df["Close"], source

    def intraday_for(self, symbol: str, tf: str):
        """Intraday OHLC (1D/1W) for a symbol, always renderable."""
        if self.demo:
            return synthetic_intraday(tf, seed=_seed(symbol)), "synthetic (demo)"
        try:
            return get_intraday_ohlc(replace(self.settings, ticker=symbol), tf), "live"
        except Exception as exc:  # noqa: BLE001 — fall back so the chart never blanks
            # Carry the reason: "failed to fetch" alone leaves nothing to act on.
            return (synthetic_intraday(tf, seed=_seed(symbol)),
                    f"synthetic (intraday unavailable — {exc})")

    # The meta-labelling forest is the one genuinely slow thing in the app —
    # seconds per symbol, versus milliseconds for everything else. It gets its
    # own cache and a long TTL, and nothing reaches it unless the user opens the
    # panel that asks for it.
    META_TTL = 900.0
    # Heatmaps read cached state, so this only saves the re-aggregation; short
    # enough that toggling views never shows yesterday's board.
    HEAT_TTL = 120.0

    def state_payload(self, symbol: str, *, with_vol: bool = False,
                      with_meta: bool = False) -> dict:
        """HUD payload for a symbol, memoised with a short TTL.

        The HAR volatility forecast costs ~6x the rest of the payload combined,
        and only the focused chart uses it — the scanner, watchlist quotes and
        alerts never touch it. So it is off by default and cached separately,
        which keeps a 100-symbol scan from paying for 100 forecasts it discards.
        The meta-labelling forest is slower still and follows the same rule, one
        tier further out.
        """
        if with_meta:
            cache, ttl = self._state_meta_cache, self.META_TTL
        elif with_vol:
            cache, ttl = self._state_vol_cache, self.ttl
        else:
            cache, ttl = self._state_cache, self.ttl
        cached = cache.get(symbol)
        if cached and (time.monotonic() - cached[0]) < ttl:
            return cached[1]
        df, source = self.ohlc_for(symbol)
        close = df["Close"]
        payload = market_state(close, symbol, window=self.settings.window,
                               threshold=self.settings.threshold, strategy=self.settings.strategy,
                               ohlc=df, with_vol=with_vol or with_meta, with_meta=with_meta)
        payload["dataSource"] = source
        payload["name"] = self.name_for(symbol)
        cache[symbol] = (time.monotonic(), payload)
        return payload

    # ── the meta-labelling forest, and its licence to touch a live order ─────
    def meta_report(self, symbol: str) -> dict:
        """All three curves for a symbol, each Sharpe deflated, plus the verdict.

        `beatsBase` is the deployment gate, computed here rather than left to
        the eye — and it is the same value live sizing consults, so what the
        panel shows you and what the terminal acts on cannot disagree.
        """
        from .sharpe_stats import deannualize, deflated_sharpe
        from .sharpe_stats import verdict as _verdict

        symbol = (symbol or "").strip().upper() or self.settings.ticker
        payload = self.state_payload(symbol, with_meta=True)
        m = payload.get("metrics", {})
        meta = m.get("meta")
        skew, kurt = m.get("skew", 0.0), m.get("kurtosis", 3.0)
        n_trials, s_var = 3, 0.25   # base, vol-targeted, meta-labelled

        def dsr(sharpe, n_obs):
            if sharpe is None or not n_obs or n_obs < 3:
                return None
            p = deflated_sharpe(deannualize(sharpe), int(n_obs), n_trials, s_var,
                                skew=skew, kurtosis=kurt)
            return {"dsr": round(p, 4), "verdict": _verdict(p)}

        base_dsr = dsr(m.get("sharpe"), m.get("nObs"))
        meta_dsr = dsr(None if not meta else meta.get("sharpe"),
                       None if not meta else meta.get("nTrades"))
        beats = bool(meta and meta.get("sharpe") is not None
                     and m.get("sharpe") is not None
                     and meta["sharpe"] > m["sharpe"]
                     and meta_dsr and base_dsr
                     and meta_dsr["dsr"] >= base_dsr["dsr"])
        return {
            "symbol": symbol,
            "dataSource": payload.get("dataSource"),
            "base": {"sharpe": m.get("sharpe"), "maxDrawdown": m.get("maxDrawdown"),
                     "winRate": m.get("winRate"), "nObs": m.get("nObs"),
                     "equity": m.get("equity", []), "equityIndex": m.get("equityIndex", []),
                     **(base_dsr or {})},
            "vt": m.get("vt"),
            "meta": None if not meta else {**meta, **(meta_dsr or {})},
            "beatsBase": beats,
        }

    def meta_sizing_enabled(self) -> bool:
        cfg = self.telegram.load()
        return bool(cfg.get("metaSizing", True))

    META_SIZE_TTL = 900.0

    def meta_sizing(self, symbol: str) -> dict:
        """What the forest is allowed to do to an order for this symbol.

        Three independent conditions have to hold before a single share moves:

          1. the setting is on;
          2. the forest's own skill gate opened for this symbol — purged
             cross-validation found out-of-sample predictive power;
          3. the meta-labelled curve beat the plain one *after deflation*.

        Any one of them failing returns a multiplier of exactly 1.0, meaning the
        order goes out at the size you typed. The default is the untouched
        order; the forest has to earn every deviation from it, per symbol, and
        the reason it gives is the reason shown in the panel.
        """
        symbol = (symbol or "").strip().upper()
        idle = {"symbol": symbol, "engaged": False, "multiplier": 1.0, "pWin": None,
                "threshold": None, "reason": "", "enabled": self.meta_sizing_enabled()}
        if not self.meta_sizing_enabled():
            return {**idle, "reason": "meta sizing is switched off"}

        cached = self._meta_size_cache.get(symbol)
        if cached and (time.monotonic() - cached[0]) < self.META_SIZE_TTL:
            return cached[1]

        try:
            report = self.meta_report(symbol)
        except Exception as exc:  # noqa: BLE001 — sizing must never block an order
            return {**idle, "reason": f"could not evaluate the forest ({exc})"}

        meta = report.get("meta")
        if str(report.get("dataSource", "")).startswith("synthetic"):
            out = {**idle, "reason": "no live market data for this symbol"}
        elif not meta:
            out = {**idle, "reason": "not enough history to train the forest"}
        elif not meta.get("active"):
            out = {**idle, "reason": "cross-validation found no predictive skill here"}
        elif not report.get("beatsBase"):
            out = {**idle, "reason": "the filtered curve does not beat the plain one after deflation"}
        else:
            from .meta_label import size_multiplier

            p, thr = meta.get("pWinNow"), meta.get("threshold")
            mult = 1.0 if p is None else size_multiplier(p, thr)
            out = {
                "symbol": symbol, "engaged": True, "multiplier": round(float(mult), 3),
                "pWin": p, "threshold": thr, "enabled": True,
                "cvAuc": meta.get("cvAuc"),
                "reason": ("the forest rates this signal below its own base rate"
                           if mult == 0.0 else
                           f"P(win) {p:.0%} vs a {thr:.0%} base rate"),
            }
        self._meta_size_cache[symbol] = (time.monotonic(), out)
        return out

    # Scored-universe cache. Regimes move on a daily scale, so a few minutes of
    # staleness is harmless — and it makes every rescan/filter change instant.
    SCAN_TTL = 300.0

    def scored_universe(self, scope: str, symbols: list, *, workers: int = 16) -> list:
        """Scored list for a scan scope, memoised so filters/rescans are free."""
        from .scanner import score_symbols
        cached = self._scan_cache.get(scope)
        if cached and (time.monotonic() - cached[0]) < self.SCAN_TTL:
            return cached[1]
        results = score_symbols(self, symbols, workers=workers)
        self._scan_cache[scope] = (time.monotonic(), results)
        return results

    def prewarm(self, scope: str, symbols: list, *, delay: float = 8.0,
                workers: int = 2) -> None:
        """Warm the scan cache in the background so the first scan is instant.

        Deliberately lazy and gentle: it waits for the dashboard to finish
        loading, then scores with only a couple of threads. Warming is a
        nice-to-have — it must never compete with the user's own requests for
        CPU or bandwidth, which is what was making startup feel slow.
        """
        import threading

        from .scanner import score_symbols

        def work():
            time.sleep(delay)                       # let the UI settle first
            if scope in self._scan_cache:
                return
            try:
                results = score_symbols(self, symbols, workers=workers)
                self._scan_cache[scope] = (time.monotonic(), results)
            except Exception:  # noqa: BLE001 — a warm-up must never break startup
                pass

        threading.Thread(target=work, daemon=True).start()

    def regime_perf_for(self, symbol: str) -> dict:
        """Model's walk-forward performance bucketed by regime (cached)."""
        from .regime import label_regimes, regime_performance
        key = symbol.upper()
        cached = self._regperf_cache.get(key)
        if cached and (time.monotonic() - cached[0]) < self.ttl:
            return cached[1]
        close, _ = self.close_for(key)
        labels = label_regimes(close, window=self.settings.window,
                               threshold=self.settings.threshold)
        perf = regime_performance(close, labels)
        self._regperf_cache[key] = (time.monotonic(), perf)
        return perf

    def journal_regime(self, symbol: str) -> str:
        """Best-effort current regime for a symbol, for auto-journaling."""
        try:
            return self.state_payload(symbol.upper()).get("regime", "")
        except Exception:  # noqa: BLE001
            return ""


_ORDER_FIELDS = {
    "symbol", "side", "order_type", "time_in_force", "order_class", "qty", "notional",
    "lots", "lot_size",
    "limit_price", "stop_price", "trail_price", "trail_percent",
    "take_profit_limit", "stop_loss_stop", "stop_loss_limit", "extended_hours",
}


def create_app(state: AppState):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="Mamba Terminal", docs_url=None, redoc_url=None)

    # ── page ─────────────────────────────────────────────────────────────────
    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/api/config")
    def config():
        return {
            "mode": state.settings.mode.value,
            "canTrade": state.settings.can_trade,
            "demo": state.demo,
            "account": state.settings.account,
            "accountPaper": state.settings.account_paper,
            "connected": state.broker is not None,
            "symbols": DEFAULT_SYMBOLS,
            "defaultSymbol": state.settings.ticker,
        }

    @app.get("/api/symbols")
    def symbols():
        return {"symbols": DEFAULT_SYMBOLS}

    @app.post("/api/mode")
    async def set_mode(request: Request):
        data = await request.json()
        m = str(data.get("mode", "")).strip().lower()
        try:
            state.settings.mode = Mode(m)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"invalid mode: {m!r}")
        return config()

    @app.get("/api/search")
    def search(q: str = ""):
        """Search-as-you-type. Never blocks on the network: while Alpaca's asset
        list is still loading it answers from the bundled universe."""
        q = (q or "").strip().upper()
        if not q:
            return {"results": [{"symbol": s, "name": state.name_for(s)} for s in DEFAULT_SYMBOLS],
                    "connected": state.universe_ready(), "loading": False}
        results, live = state.search_symbols(q)
        # Not verified against Alpaca (offline or still loading) — allow the
        # exact ticker typed so the user is never blocked from adding it.
        if not live and q not in results:
            results = [q] + results
        results = results[:30]
        return {"results": [{"symbol": s, "name": state.name_for(s)} for s in results],
                "connected": live,
                "loading": state.broker is not None and not state.universe_ready()}

    @app.get("/api/validate")
    def validate(symbol: str = ""):
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return {"valid": False, "reason": "empty"}
        al = state.alpaca_symbols()
        if al is not None:
            if symbol in al:
                asset = state.broker.get_asset(symbol)
                return {"valid": True, "source": "alpaca",
                        "tradable": asset["tradable"] if asset else True,
                        "name": asset["name"] if asset else ""}
            return {"valid": False, "source": "alpaca",
                    "reason": f"{symbol} is not an Alpaca-tradable symbol"}
        return {"valid": True, "source": "unverified"}  # not connected — can't check

    @app.get("/api/state")
    def state_endpoint(symbol: str = "SPY"):
        symbol = symbol.strip().upper() or "SPY"
        return state.state_payload(symbol, with_vol=True)

    @app.get("/api/heatmap")
    def heatmap_endpoint(view: str = "regime", scope: str = "watchlist",
                         symbols: str = "", watchlist: str = ""):
        """Board-wide heatmaps: regime map, signal map, correlation.

        Reads cached per-symbol state, so switching views is instant and no view
        triggers a download the dashboard has not already paid for.
        """
        from concurrent.futures import ThreadPoolExecutor

        from .heatmap import correlation_matrix, regime_grid, signal_map
        from .regime import label_regimes

        picked = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if not picked:
            if scope in SCAN_GROUPS:
                picked = list(SCAN_GROUPS[scope])
            elif scope == "market":
                picked = list(SCAN_UNIVERSE)
            else:
                picked = ([s.strip().upper() for s in watchlist.split(",") if s.strip()]
                          or list(DEFAULT_SYMBOLS))
        picked = picked[:60]

        view = (view or "regime").lower()
        key = f"{view}|{scope}|{','.join(picked)}"
        hit = state._heat_cache.get(key)
        if hit and (time.monotonic() - hit[0]) < state.HEAT_TTL:
            return hit[1]

        # Only the signal map needs the full per-symbol payload; the regime and
        # correlation views need price history and nothing else. Computing the
        # walk-forward backtest for 60 names to colour a grid would be paying
        # for the whole engine to render a picture of the labels.
        needs_state = view == "signal"

        def one(sym):
            try:
                close, _ = state.close_for(sym)
                payload = state.state_payload(sym) if needs_state else None
            except Exception:  # noqa: BLE001 — one bad symbol must not blank the board
                return None
            lab = None if view == "correlation" else label_regimes(
                close, window=state.settings.window, threshold=state.settings.threshold)
            return sym, payload, close, lab

        # Threads earn their keep when the fetches are real network calls; the
        # per-symbol maths is pandas-bound and gains nothing from more of them.
        closes, labels, states = {}, {}, []
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(picked)))) as pool:
            for got in pool.map(one, picked):
                if got is None:
                    continue
                sym, payload, close, lab = got
                if payload is not None:
                    states.append(payload)
                closes[sym] = close
                if lab is not None:
                    labels[sym] = lab

        out = {"view": view, "scope": scope, "symbols": list(closes)}
        if view == "signal":
            out["signal"] = signal_map(states)
        elif view in ("correlation", "corr"):
            out["correlation"] = correlation_matrix(closes)
        else:
            out["regime"] = regime_grid(closes, labels)
        state._heat_cache[key] = (time.monotonic(), out)
        return out

    @app.get("/api/health")
    def health_endpoint():
        """What is currently degraded, and what has been repaired."""
        return state.healer.status()

    @app.post("/api/health/heal")
    def heal_now():
        """Run the repair sweep on demand instead of waiting for the timer."""
        result = state.healer.run()
        return {"ok": True, "result": result, "status": state.healer.status()}

    @app.post("/api/health/retry")
    async def retry_symbol(request: Request):
        """Force an immediate re-fetch of a symbol stuck on fallback data."""
        payload = await request.json()
        symbol = str(payload.get("symbol", "")).strip().upper()
        if not symbol:
            raise HTTPException(status_code=400, detail="symbol is required")
        state._ohlc_cache.pop(symbol, None)
        state.healer.data_failures.pop(symbol, None)
        for c in (state._state_cache, state._state_vol_cache, state._state_meta_cache):
            c.pop(symbol, None)
        _, source = state.ohlc_for(symbol)
        return {"ok": source == "live", "symbol": symbol, "dataSource": source}

    @app.get("/api/meta-label")
    def meta_label_endpoint(symbol: str = "SPY"):
        """The meta-labelling forest for one symbol — on demand, never implicit.

        Returns the third equity curve next to the two it has to beat, and the
        deflated Sharpe of each. Deflation is the point: the filter is one of
        several variants tried on the same history, so its Sharpe is compared
        against what the best of that many noise draws would have produced. A
        filter that only wins before deflation has not won.
        """
        return state.meta_report(symbol)

    @app.get("/api/meta-sizing")
    def meta_sizing_endpoint(symbol: str = "SPY"):
        """What the forest would do to an order for this symbol, before you send it.

        The execution panel reads this so a resized order is never a surprise:
        you see the multiplier and the reason for it while you are still filling
        in the ticket.
        """
        return state.meta_sizing(symbol.strip().upper() or "SPY")

    @app.post("/api/meta-sizing/toggle")
    async def meta_sizing_toggle(request: Request):
        data = await request.json()
        enabled = bool(data.get("enabled"))
        state.telegram.save(metaSizing=enabled)
        state._meta_size_cache.clear()
        return {"ok": True, "enabled": state.meta_sizing_enabled()}

    @app.get("/api/candles")
    def candles(symbol: str = "SPY", tf: str = "1D"):
        import pandas as pd
        symbol = symbol.strip().upper() or "SPY"
        tf = (tf or "1D").upper()
        df, source = state.intraday_for(symbol, tf)
        if df is None or df.empty:
            return {"symbol": symbol, "tf": tf, "bars": [], "ma20": [], "ma50": [],
                    "rsi": [], "momentum": [], "source": source}
        df = df.iloc[-400:]
        close = df["Close"]
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        rsi = _rsi(close)
        mom = (close / close.shift(14) - 1.0) * 100.0

        def r4(v):
            return round(float(v), 4)

        def ser(s):
            return [None if pd.isna(v) else round(float(v), 4) for v in s]

        bars = [
            {"t": ts.strftime("%Y-%m-%d %H:%M"), "o": r4(o), "h": r4(h), "l": r4(lo),
             "c": r4(c), "up": bool(c >= o)}
            for ts, o, h, lo, c in zip(df.index, df["Open"], df["High"], df["Low"], df["Close"])
        ]
        return {"symbol": symbol, "tf": tf, "bars": bars, "ma20": ser(ma20), "ma50": ser(ma50),
                "rsi": ser(rsi), "momentum": ser(mom), "source": source}

    @app.get("/api/scan")
    def scan_endpoint(symbols: str = "", top: int = 20, universe: str = "",
                      fresh: int = 0, proven: bool = False, sort: str = "score",
                      watchlist: str = ""):
        from .scanner import rank
        scope = (universe or "").strip().lower() or "market"
        if symbols.strip():
            syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            scope, key = "custom", "custom:" + ",".join(sorted(syms))
        elif scope == "watchlist":
            syms = [s.strip().upper() for s in watchlist.split(",") if s.strip()] or DEFAULT_SYMBOLS
            key = "watchlist:" + ",".join(sorted(syms))
        elif scope in SCAN_GROUPS:
            syms, key = SCAN_GROUPS[scope], scope
        else:
            syms, scope, key = SCAN_UNIVERSE, "market", "market"
        # Scoring is cached per-universe, so filters and rescans are instant.
        result = rank(state.scored_universe(key, syms), top=top,
                      fresh_days=max(0, int(fresh)), proven_only=bool(proven), sort=sort)
        result["universe"] = scope
        result["universeSize"] = len(syms)
        return result

    @app.get("/api/regime-performance")
    def regime_perf(symbol: str = "SPY"):
        symbol = symbol.strip().upper() or "SPY"
        return {"symbol": symbol, "byRegime": state.regime_perf_for(symbol)}

    @app.get("/api/journal")
    def journal_list():
        return {"entries": state.journal.list(), "analytics": state.journal.analytics()}

    @app.post("/api/journal")
    async def journal_add(request: Request):
        data = await request.json()
        if not data.get("symbol"):
            raise HTTPException(status_code=400, detail="symbol required")
        entry = state.journal.add(
            symbol=data.get("symbol"), side=data.get("side", "buy"),
            qty=data.get("qty"), price=data.get("price"),
            regime=data.get("regime", ""), tags=data.get("tags"),
            notes=data.get("notes", ""), pnl=data.get("pnl"),
            r_multiple=data.get("rMultiple"), source="manual")
        return {"ok": True, "entry": entry}

    @app.post("/api/journal/update")
    async def journal_update(request: Request):
        data = await request.json()
        fields = {k: data[k] for k in ("tags", "notes", "pnl", "rMultiple", "regime") if k in data}
        entry = state.journal.update(str(data.get("id", "")), **fields)
        if entry is None:
            raise HTTPException(status_code=404, detail="entry not found")
        return {"ok": True, "entry": entry}

    @app.post("/api/journal/delete")
    async def journal_delete(request: Request):
        data = await request.json()
        if not state.journal.remove(str(data.get("id", ""))):
            raise HTTPException(status_code=404, detail="entry not found")
        return {"ok": True}

    # ── alerts + risk automation ─────────────────────────────────────────────
    @app.get("/api/alerts")
    def alerts(symbols: str = ""):
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()][:16]
        events = []
        if syms:
            regimes, prices = {}, {}
            for s in syms:
                cached = state._state_cache.get(s)
                if cached and (time.monotonic() - cached[0]) < state.ttl:
                    regimes[s] = cached[1].get("regime", "")
                    prices[s] = cached[1].get("lastPrice")
                    continue
                try:
                    close, _ = state.close_for(s)
                    q = quote_state(close, s, window=state.settings.window,
                                    threshold=state.settings.threshold)
                    regimes[s], prices[s] = q["regime"], q["lastPrice"]
                except Exception:  # noqa: BLE001
                    pass
            flips = state.alerts.check_regimes(regimes)
            events += flips
            events += state.alerts.check_prices(prices)
            if flips and state.telegram.status()["sendFlips"]:
                _tg_send("\n".join(format_flip(f) for f in flips))

        day_pl = None
        if state.broker is not None:
            try:
                acct = state.broker.get_account()
                day_pl = round(acct.equity - acct.last_equity, 2) if acct.last_equity else 0.0
                breach = state.alerts.check_loss_limit(day_pl)
                if breach is not None:
                    _flatten()
                    events.append(breach)
            except Exception:  # noqa: BLE001
                pass

        return {"events": events, "recent": state.alerts.recent(),
                "halted": state.alerts.halted, "lossLimit": state.alerts.loss_limit,
                "dayPl": day_pl, "priceAlerts": state.alerts.price_alerts()}

    # ── telegram ─────────────────────────────────────────────────────────────
    def _tg_send(text: str) -> None:
        """Fire-and-forget delivery — Telegram must never block or break a poll."""
        import threading

        def work():
            try:
                state.telegram.send(text)
            except Exception:  # noqa: BLE001
                pass

        if text and state.telegram.enabled:
            threading.Thread(target=work, daemon=True).start()

    @app.get("/api/telegram")
    def telegram_status():
        return state.telegram.status()

    @app.post("/api/telegram/connect")
    async def telegram_connect(request: Request):
        data = await request.json()
        try:
            status = state.telegram.connect(str(data.get("token", "")),
                                            str(data.get("chatId", "") or ""))
        except TelegramError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _tg_send("✅ <b>Mamba Terminal connected.</b> "
                 "You'll get scanner reports and regime-flip alerts here.")
        return status

    @app.post("/api/telegram/disconnect")
    def telegram_disconnect():
        return state.telegram.disconnect()

    @app.post("/api/telegram/settings")
    async def telegram_settings(request: Request):
        data = await request.json()
        state.telegram.save(
            sendScans=bool(data.get("sendScans", True)),
            sendFlips=bool(data.get("sendFlips", True)),
            minScore=int(data.get("minScore", 70)))
        return state.telegram.status()

    @app.post("/api/telegram/test")
    def telegram_test():
        try:
            state.telegram.send("🐍 <b>Test message</b> from Mamba Terminal — "
                                "Telegram is wired up correctly.")
        except TelegramError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"ok": True}

    @app.post("/api/telegram/send_scan")
    def telegram_send_scan(universe: str = "market", top: int = 5):
        """Push the current scanner picks to Telegram, on demand."""
        from .scanner import rank
        syms = SCAN_GROUPS.get(universe, SCAN_UNIVERSE)
        key = universe if universe in SCAN_GROUPS else "market"
        cfg = state.telegram.status()
        result = rank(state.scored_universe(key, syms), top=max(1, top))
        text = format_scan(result["results"], min_score=cfg["minScore"], limit=top)
        if not text:
            return {"ok": False, "reason": f"nothing scored at or above {cfg['minScore']}"}
        try:
            state.telegram.send(text)
        except TelegramError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"ok": True, "sent": len(result["results"])}

    # ── background scan watcher ──────────────────────────────────────────────
    @app.get("/api/watcher")
    def watcher_status():
        return state.watcher.status()

    @app.post("/api/watcher")
    async def watcher_config(request: Request):
        from .watcher import DEFAULTS
        data = await request.json()
        fields = {k: data[k] for k in DEFAULTS if k in data}
        if "scanIntervalMin" in fields:
            fields["scanIntervalMin"] = max(5, int(fields["scanIntervalMin"]))
        state.telegram.save(**fields)
        if state.watcher.config()["autoScan"]:
            state.watcher.start()          # idempotent
        return state.watcher.status()

    @app.post("/api/watcher/run")
    def watcher_run_now():
        """Sweep immediately, ignoring quiet hours — for testing the setup."""
        if not state.telegram.enabled:
            raise HTTPException(status_code=400, detail="Connect Telegram first.")
        try:
            return state.watcher.run_once(force=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"scan failed: {exc}")

    @app.post("/api/alerts/price")
    async def add_price_alert(request: Request):
        data = await request.json()
        sym = str(data.get("symbol", "")).strip().upper()
        op = str(data.get("op", "above")).lower()
        price = data.get("price")
        if not sym or op not in ("above", "below") or price is None:
            raise HTTPException(status_code=400,
                                detail="need symbol, op ('above'/'below') and price")
        alert = state.alerts.add_price_alert(sym, op, float(price))
        return {"ok": True, "alert": alert, "priceAlerts": state.alerts.price_alerts()}

    @app.post("/api/alerts/price/delete")
    async def delete_price_alert(request: Request):
        data = await request.json()
        state.alerts.remove_price_alert(str(data.get("id", "")))
        return {"ok": True, "priceAlerts": state.alerts.price_alerts()}

    @app.post("/api/risk/limit")
    async def set_risk_limit(request: Request):
        data = await request.json()
        state.alerts.set_loss_limit(data.get("lossLimit"))
        return {"ok": True, "lossLimit": state.alerts.loss_limit}

    @app.post("/api/kill")
    def kill_switch():
        event = state.alerts.trip_kill("manual kill switch")
        result = {"ok": True, "halted": True, "event": event, "flattened": None}
        if state.broker is not None:
            result["flattened"] = _flatten()
        return result

    @app.post("/api/kill/reset")
    def kill_reset():
        state.alerts.reset_kill()
        return {"ok": True, "halted": False}

    @app.get("/api/quote")
    def quote(symbol: str = "SPY"):
        symbol = symbol.strip().upper() or "SPY"
        cached = state._state_cache.get(symbol)
        if cached and (time.monotonic() - cached[0]) < state.ttl:
            p = cached[1]
            return {"ticker": symbol, "lastPrice": p["lastPrice"], "regime": p["regime"],
                    "name": p.get("name", "")}
        close, _ = state.close_for(symbol)
        q = quote_state(close, symbol, window=state.settings.window,
                        threshold=state.settings.threshold)
        q["name"] = state.name_for(symbol)
        return q

    @app.get("/api/quotes")
    def quotes(symbols: str = ""):
        """Batch watchlist quotes, fetched in parallel — one round-trip for the
        whole watchlist instead of one request (and one download) per symbol."""
        from concurrent.futures import ThreadPoolExecutor

        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()][:32]
        if not syms:
            return {"quotes": []}

        def one(sym):
            cached = state._state_cache.get(sym)
            if cached and (time.monotonic() - cached[0]) < state.ttl:
                p = cached[1]
                return {"ticker": sym, "lastPrice": p["lastPrice"], "regime": p["regime"],
                        "name": p.get("name", ""),
                        "dataSource": p.get("dataSource", "live"),
                        "real": not str(p.get("dataSource", "")).startswith("synthetic")
                                or state.demo}
            try:
                close, source = state.close_for(sym)
                q = quote_state(close, sym, window=state.settings.window,
                                threshold=state.settings.threshold)
                q["name"] = state.name_for(sym)
                # Be honest about fabricated data: a regime computed on the
                # synthetic fallback says nothing about the real asset.
                q["dataSource"] = source
                q["real"] = state.demo or not source.startswith("synthetic")
                if not q["real"]:
                    q["regime"] = "unknown"
                return q
            except Exception:  # noqa: BLE001 — a bad symbol must not sink the batch
                return None

        with ThreadPoolExecutor(max_workers=min(8, len(syms))) as pool:
            results = [q for q in pool.map(one, syms) if q]
        return {"quotes": results}

    @app.get("/api/news")
    def news(symbol: str = "SPY"):
        symbol = symbol.strip().upper() or "SPY"
        cached = state._news_cache.get(symbol)
        if cached and (time.monotonic() - cached[0]) < 300.0:
            items = cached[1]
        else:
            items = fetch_news(state.settings, symbol, demo=state.demo)
            state._news_cache[symbol] = (time.monotonic(), items)
        return {"symbol": symbol, "items": items}

    # ── portfolio ────────────────────────────────────────────────────────────
    @app.get("/api/portfolio")
    def portfolio(symbol: str = "SPY"):
        if state.broker is None:
            return {"connected": False, "account": None, "position": None, "openOrders": []}
        try:
            acct = state.broker.get_account()
            pos = state.broker.get_position(symbol.strip().upper())
            orders = state.broker.list_open_orders()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"broker read failed: {exc}")
        return {
            "connected": True,
            "account": {"equity": acct.equity, "cash": acct.cash,
                        "buyingPower": acct.buying_power, "status": acct.status},
            "position": None if pos is None else {
                "symbol": pos.symbol, "qty": pos.qty, "side": pos.side,
                "marketValue": pos.market_value, "unrealizedPl": pos.unrealized_pl},
            "openOrders": [
                {"id": o.id, "symbol": o.symbol, "side": o.side, "type": o.type,
                 "qty": o.qty, "status": o.status}
                for o in orders
            ],
        }

    # ── blotter (all positions + open orders, with actions) ──────────────────
    @app.get("/api/blotter")
    def blotter():
        if state.broker is None:
            return {"connected": False, "canTrade": False, "account": None,
                    "positions": [], "openOrders": []}
        try:
            acct = state.broker.get_account()
            positions = state.broker.list_positions()
            orders = state.broker.list_open_orders()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"broker read failed: {exc}")
        day_pl = acct.equity - acct.last_equity if acct.last_equity else 0.0
        day_pl_pct = (day_pl / acct.last_equity) if acct.last_equity else 0.0
        return {
            "connected": True,
            "canTrade": state.settings.can_trade,
            "account": {"equity": acct.equity, "cash": acct.cash,
                        "buyingPower": acct.buying_power, "status": acct.status,
                        "dayPl": round(day_pl, 2), "dayPlPct": round(day_pl_pct, 4)},
            "positions": [
                {"symbol": p.symbol, "qty": p.qty, "side": p.side,
                 "marketValue": p.market_value, "unrealizedPl": p.unrealized_pl,
                 "unrealizedPlpc": p.unrealized_plpc, "avgEntry": p.avg_entry,
                 "currentPrice": p.current_price}
                for p in positions
            ],
            "openOrders": [
                {"id": o.id, "symbol": o.symbol, "side": o.side, "type": o.type,
                 "qty": o.qty, "status": o.status}
                for o in orders
            ],
        }

    @app.post("/api/positions/close")
    async def close_position(request: Request):
        if state.broker is None:
            raise HTTPException(status_code=403, detail="No account connected.")
        data = await request.json()
        sym = str(data.get("symbol", "")).strip().upper()
        if not sym:
            raise HTTPException(status_code=400, detail="symbol required")
        # Read the position first: once it is closed the unrealized P&L is gone,
        # and that number is exactly what the trade realized.
        pos = None
        try:
            pos = state.broker.get_position(sym)
        except Exception:  # noqa: BLE001
            pass
        try:
            res = state.broker.close_position(sym)
        except ReadOnlyError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 — surface Alpaca's own wording
            raise HTTPException(status_code=502, detail=f"{sym} close rejected: {exc}")
        _journal_close(sym, pos)
        return _close_result(sym, res)

    @app.post("/api/orders/cancel")
    async def cancel_one_order(request: Request):
        if state.broker is None:
            raise HTTPException(status_code=403, detail="No account connected.")
        data = await request.json()
        oid = str(data.get("id", "")).strip()
        if not oid:
            raise HTTPException(status_code=400, detail="order id required")
        try:
            state.broker.cancel_order(oid)
        except ReadOnlyError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"cancel failed: {exc}")
        return {"ok": True}

    # ── orders ───────────────────────────────────────────────────────────────
    def _close_result(symbol: str, res) -> dict:
        """Say what actually happened, not what we hoped happened.

        A close is an order. It fills instantly in market hours and queues
        otherwise, so we re-read the position and report 'flat' or 'pending'
        rather than announcing success either way.
        """
        info = res if isinstance(res, dict) else {"symbol": symbol, "orderId": "",
                                                  "status": str(res), "cancelledOrders": 0}
        flat, qty_left = True, 0.0
        try:
            qty_left = state.broker.position_qty(symbol)
            flat = abs(qty_left) < 1e-9
        except Exception:  # noqa: BLE001 — verification is best effort
            flat = False

        pre = (f"cancelled {info['cancelledOrders']} working order(s) first · "
               if info.get("cancelledOrders") else "")
        if flat:
            msg = f"{pre}{symbol} closed — position is flat."
        else:
            msg = (f"{pre}{symbol} close order submitted"
                   f"{' (' + info['status'] + ')' if info.get('status') else ''}"
                   f" — still holding {qty_left:g}. It will fill when the market is open;"
                   " the blotter updates as it does.")
        return {"ok": True, "flat": flat, "qtyRemaining": qty_left,
                "orderId": info.get("orderId", ""), "status": info.get("status", ""),
                "cancelledOrders": info.get("cancelledOrders", 0), "message": msg}

    def _journal_close(symbol: str, pos) -> None:
        """Complete the journal entry for a position that just closed."""
        if pos is None:
            return
        try:
            state.journal.close_trade(
                symbol, float(pos.unrealized_pl),
                price=getattr(pos, "current_price", None) or None,
                qty=getattr(pos, "qty", None),
                regime_fn=lambda: state.journal_regime(symbol))
        except Exception:  # noqa: BLE001 — journaling must never block a close
            pass

    def _flatten():
        """Cancel every open order and close every position. Best-effort."""
        cancelled = 0
        try:
            cancelled = state.broker.cancel_all_orders()
        except Exception:  # noqa: BLE001
            pass
        closed = 0
        try:
            for p in state.broker.list_positions():
                try:
                    state.broker.close_position(p.symbol)
                    closed += 1
                    _journal_close(p.symbol, p)   # a flatten is still a set of exits
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        return {"cancelled": cancelled, "closed": closed}

    def _is_opening(state, ticket) -> bool:
        """True when this ticket opens or adds to a position rather than closing one.

        Sizing may only touch trades that put risk on. If you already hold the
        symbol and the order runs the other way, it is an exit — and an exit
        must go out whole, at the size you asked for, whatever any model thinks
        of it.
        """
        try:
            pos = state.broker.get_position(ticket.symbol)
        except Exception:  # noqa: BLE001 — no position, or no way to ask
            return True
        if pos is None:
            return True
        try:
            held = float(getattr(pos, "qty", 0) or 0)
        except (TypeError, ValueError):
            return True
        if held == 0:
            return True
        side = (getattr(ticket, "side", "") or "").lower()
        return (held > 0 and side == "buy") or (held < 0 and side == "sell")

    def _scale_ticket(ticket, mult: float):
        """Shrink a ticket's size, leaving every other field exactly as typed.

        Only the three size fields move. Lots are rounded down to whole lots and
        shares to whole shares where the original was whole, so scaling never
        turns a clean order into a fractional one the broker may reject.
        """
        from dataclasses import replace as _replace

        changes = {}
        if ticket.qty is not None:
            q = float(ticket.qty) * mult
            changes["qty"] = float(int(q)) if float(ticket.qty).is_integer() else round(q, 6)
        if ticket.lots is not None:
            lot = float(ticket.lots) * mult
            changes["lots"] = float(int(lot)) if float(ticket.lots).is_integer() else round(lot, 6)
        if ticket.notional is not None:
            changes["notional"] = round(float(ticket.notional) * mult, 2)
        # Scaling to nothing is a skip, not an order for zero shares.
        for field in ("qty", "lots", "notional"):
            if field in changes and changes[field] <= 0:
                raise HTTPException(status_code=409, detail=(
                    "Meta-labelling scaled this order below the smallest tradable "
                    "size. Send it again with 'skipMetaSizing' to override."))
        return _replace(ticket, **changes) if changes else ticket

    @app.post("/api/orders")
    async def submit_order(request: Request):
        if state.broker is None:
            raise HTTPException(status_code=403, detail="No account connected. Add one under Accounts.")
        if state.alerts.halted:
            raise HTTPException(status_code=423,
                                detail="Trading is HALTED by the kill switch. Reset it in Alerts to place orders.")
        data = await request.json()
        if not isinstance(data, dict) or not data.get("symbol"):
            raise HTTPException(status_code=400, detail="An order needs at least a symbol.")
        ticket = OrderTicket(**{k: v for k, v in data.items() if k in _ORDER_FIELDS})

        # Meta-labelling applied to the live ticket. It can shrink the order or
        # refuse it; it can never enlarge it, and it never touches the side, the
        # symbol, or any price. An opening trade only — reducing an exit because
        # a model dislikes it would leave you stuck in a position you asked to
        # leave, which is the one thing a sizing layer must never do.
        sizing = None
        try:
            consult = state.meta_sizing_enabled() and not data.get("skipMetaSizing")
            sizing = state.meta_sizing(ticket.symbol) if consult else None
        except Exception:  # noqa: BLE001
            # A sizing layer that breaks must not take the order down with it.
            # Failing open sends exactly what you typed, which is the same thing
            # that would have happened before this feature existed.
            sizing = None
        if sizing is not None:
            if sizing.get("engaged") and _is_opening(state, ticket):
                mult = float(sizing["multiplier"])
                if mult <= 0.0:
                    raise HTTPException(status_code=409, detail=(
                        f"Meta-labelling skipped this trade: {sizing['reason']}. "
                        "Send it again with 'skipMetaSizing' to override."))
                if mult < 1.0:
                    ticket = _scale_ticket(ticket, mult)

        try:
            result = state.broker.submit_ticket(ticket)
        except OrderValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except ReadOnlyError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 — Alpaca rejection
            raise HTTPException(status_code=502, detail=f"order rejected: {exc}")
        # auto-journal the entry with the regime the symbol was in (never fatal)
        try:
            note = ""
            if sizing is not None and sizing.get("engaged") and float(sizing["multiplier"]) < 1.0:
                note = (f"meta-labelled to {sizing['multiplier']:.2f}x "
                        f"— {sizing['reason']}")
            state.journal.add(
                symbol=ticket.symbol, side=getattr(ticket, "side", None) or "buy",
                qty=ticket.qty, price=ticket.limit_price or ticket.stop_price,
                regime=state.journal_regime(ticket.symbol), source="order", notes=note)
        except Exception:  # noqa: BLE001
            pass
        out = {"id": result.id, "status": result.status, "summary": result.summary}
        if sizing is not None and sizing.get("engaged") and float(sizing["multiplier"]) < 1.0:
            # Say so in the response. An order that went out smaller than the
            # one you typed must never be reported as though nothing happened.
            out["metaSizing"] = {**sizing, "appliedQty": ticket.qty,
                                 "appliedNotional": ticket.notional}
        return out

    @app.post("/api/orders/cancel_all")
    def cancel_all():
        if state.broker is None:
            raise HTTPException(status_code=403, detail="No account connected.")
        try:
            n = state.broker.cancel_all_orders()
        except ReadOnlyError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"cancel failed: {exc}")
        return {"cancelled": n}

    # ── accounts ─────────────────────────────────────────────────────────────
    @app.get("/api/accounts")
    def list_accounts():
        try:
            profiles = state.accounts.list()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc))
        return {
            "active": state.accounts.active() if profiles else None,
            "accounts": [{"name": p.name, "paper": p.paper, "active": p.active} for p in profiles],
        }

    def _connect(name: str):
        resolved = state.accounts.resolve(name)
        if resolved is not None:
            state.settings.api_key = resolved.key_id
            state.settings.api_secret = resolved.secret
            state.settings.account = resolved.name
            state.settings.account_paper = resolved.paper
            state.reconnect()

    @app.post("/api/accounts")
    async def add_account(request: Request):
        data = await request.json()
        try:
            state.accounts.add(str(data.get("name", "")), str(data.get("key_id", "")),
                               str(data.get("secret", "")), paper=bool(data.get("paper", True)))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc))
        _connect(str(data.get("name", "")))
        return list_accounts()

    @app.post("/api/accounts/active")
    async def set_active(request: Request):
        data = await request.json()
        name = str(data.get("name", ""))
        try:
            state.accounts.set_active(name)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc))
        _connect(name)
        return list_accounts()

    @app.delete("/api/accounts/{name}")
    def remove_account(name: str):
        try:
            state.accounts.remove(name)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc))
        active = state.accounts.active()
        if active:
            _connect(active)
        else:
            state.settings.account = state.settings.account_paper = None
            state.settings.api_key = state.settings.api_secret = None
            state.reconnect()
        return list_accounts()

    @app.exception_handler(OrderValidationError)
    def _ove(_request, exc):  # pragma: no cover — safety net
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


def main() -> int:
    import argparse
    import threading
    import webbrowser

    import uvicorn

    parser = argparse.ArgumentParser(prog="mamba-web",
                                     description="Mamba Terminal — neon web HUD")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.DASHBOARD.value)
    parser.add_argument("--account", default=None)
    parser.add_argument("--strategy", choices=[s.value for s in Strategy], default=Strategy.FILTER.value)
    parser.add_argument("--demo", action="store_true", help="offline synthetic data, no network")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--selftest", action="store_true",
                        help="Verify the bundle can build the server, then exit (CI smoke test)")
    args = parser.parse_args()

    settings = load_settings(
        account=args.account, ticker=args.symbol,
        mode=Mode(args.mode), strategy=Strategy(args.strategy),
    )
    state = AppState(settings, demo=args.demo)
    app = create_app(state)

    if args.selftest:
        import uvicorn  # noqa: F401 — ensure the server dependency is bundled
        index = os.path.join(STATIC_DIR, "index.html")
        if not os.path.exists(index):
            print(f"selftest FAILED: HUD asset missing at {index}")
            return 1
        print("mamba-web selftest OK — FastAPI app built, HUD asset present.")
        return 0

    # Score the scan universe in the background while the user is reading the
    # dashboard, so their first ⚡ SCAN comes back instantly.
    state.prewarm("market", SCAN_UNIVERSE)
    state.prewarm_universe()   # Alpaca asset list, so search is instant from the first keystroke
    if state.watcher.config()["autoScan"]:
        state.watcher.start()  # keep hunting even with the page closed
    state.healer.start()       # repair drift, dead threads and dropped connections

    url = f"http://{args.host}:{args.port}/"
    print(f"Mamba Terminal web HUD → {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
