"""Price-history sources for the terminal.

Three sources, one interface (`get_history` -> a daily close `pd.Series`):
  - Alpaca     : live/recent daily bars (used when credentials are present)
  - yfinance   : deep history fallback (no key needed)
  - synthetic  : offline demo data so the TUI runs with no network at all
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_close(days: int = 1500, seed: int = 0) -> pd.Series:
    """Deterministic trending series with an embedded bear and bull stretch.

    Lets the terminal render a realistic-looking dashboard fully offline.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    rets = rng.normal(0.0004, 0.01, days)
    rets[300:500] -= 0.004   # a bear stretch
    rets[900:1100] += 0.004  # a bull stretch
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=idx, name="Close")


def synthetic_ohlc(days: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Deterministic OHLC bars derived from the synthetic close series, so the
    candlestick chart renders fully offline."""
    close = synthetic_close(days=days, seed=seed)
    rng = np.random.default_rng(seed + 1)
    c = close.to_numpy()
    o = np.empty_like(c)
    o[0] = c[0]
    o[1:] = c[:-1]                                   # open = previous close
    amp = np.abs(rng.normal(0.0, 0.008, len(c))) + 0.0015
    hi = np.maximum(o, c) * (1.0 + amp)
    lo = np.minimum(o, c) * (1.0 - amp)
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": c}, index=close.index)


def _ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    cols = {c.lower(): c for c in df.columns}
    if not {"open", "high", "low", "close"} <= set(cols):
        # What a symbol with no price coverage looks like on the way back: a
        # frame missing the columns entirely. The old KeyError('open') reported
        # the symptom; the news is that the feed has no bars for this name.
        raise ValueError("no price bars published for this symbol")
    out = pd.DataFrame({
        "Open": df[cols["open"]], "High": df[cols["high"]],
        "Low": df[cols["low"]], "Close": df[cols["close"]],
    }).dropna()
    # Volume rides along when the feed publishes it. It is optional on purpose:
    # a VWAP needs it, but every price path in the terminal predates it and must
    # keep working without it. Absent is a fact to report, not a failure.
    if "volume" in cols:
        vol = pd.to_numeric(df[cols["volume"]], errors="coerce").reindex(out.index)
        if vol.notna().any():
            out["Volume"] = vol.fillna(0.0).astype(float)
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out


def from_yfinance(ticker: str, years: int = 10) -> pd.Series:
    return from_yfinance_ohlc(ticker, years)["Close"].dropna()


def from_yfinance_ohlc(ticker: str, years: int = 10) -> pd.DataFrame:
    from .run import _fetch_with_retry

    df = _fetch_with_retry(ticker, years)
    return _ohlc_columns(df)


def _alpaca_bars(ticker: str, years: int, api_key: str, api_secret: str) -> pd.DataFrame:
    """Daily OHLC bars from Alpaca (equities or dash-form crypto like BTC-USD)."""
    from alpaca.data.timeframe import TimeFrame

    start = (pd.Timestamp.now(tz="UTC").normalize() - pd.DateOffset(years=years)).to_pydatetime()

    if "/" in ticker or ticker.endswith("USD") and "-" in ticker:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest

        symbol = ticker.replace("-", "/")
        client = CryptoHistoricalDataClient(api_key, api_secret)
        req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start)
        bars = client.get_crypto_bars(req).df
        key = symbol
    else:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest

        client = StockHistoricalDataClient(api_key, api_secret)
        req = StockBarsRequest(symbol_or_symbols=ticker, timeframe=TimeFrame.Day, start=start)
        bars = client.get_stock_bars(req).df
        key = ticker

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(key, level=0)
    return _ohlc_columns(bars)


def batch_alpaca_ohlc(symbols: list[str], years: int, api_key: str, api_secret: str,
                      chunk: int = 200, since=None) -> dict[str, pd.DataFrame]:
    """Daily OHLC for many symbols at once.

    This is the difference between scanning a hundred names and scanning the
    whole market. Alpaca's bars endpoint takes a *list* of symbols and returns
    one multi-indexed frame, so a thousand symbols costs five requests instead
    of a thousand. The per-symbol path stays for single lookups; anything that
    sweeps a universe should come through here.

    Crypto (dash-form) symbols are skipped — they use a different endpoint and
    there are only a handful, so the caller fetches those individually.

    A chunk that fails is skipped rather than raising: one bad symbol in a
    batch of two hundred must not cost the other one hundred and ninety-nine.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    equities = [s for s in symbols if "/" not in s and "-" not in s]
    if not equities:
        return {}

    client = StockHistoricalDataClient(api_key, api_secret)
    # `since` turns a full history download into a top-up. Re-fetching a decade
    # to learn about yesterday is the single most wasteful thing the sweep did.
    if since is not None:
        start = pd.Timestamp(since).tz_localize(None).tz_localize("UTC").to_pydatetime()
    else:
        start = (pd.Timestamp.now(tz="UTC").normalize()
                 - pd.DateOffset(years=years)).to_pydatetime()
    out: dict[str, pd.DataFrame] = {}

    for i in range(0, len(equities), chunk):
        batch = equities[i: i + chunk]
        try:
            bars = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=batch, timeframe=TimeFrame.Day, start=start)).df
        except Exception:  # noqa: BLE001 — a bad chunk must not sink the sweep
            continue
        if bars is None or bars.empty:
            continue
        if isinstance(bars.index, pd.MultiIndex):
            for sym in batch:
                try:
                    sub = bars.xs(sym, level=0)
                except KeyError:
                    continue
                try:
                    frame = _ohlc_columns(sub)
                except Exception:  # noqa: BLE001
                    continue
                if not frame.empty:
                    out[sym] = frame
        elif len(batch) == 1:
            try:
                frame = _ohlc_columns(bars)
            except Exception:  # noqa: BLE001
                continue
            if not frame.empty:
                out[batch[0]] = frame
    return out


def from_alpaca(ticker: str, years: int, api_key: str, api_secret: str) -> pd.Series:
    return _alpaca_bars(ticker, years, api_key, api_secret)["Close"].dropna()


def from_alpaca_ohlc(ticker: str, years: int, api_key: str, api_secret: str) -> pd.DataFrame:
    return _alpaca_bars(ticker, years, api_key, api_secret)


def get_history(settings) -> pd.Series:
    """Pick a source based on available credentials. Alpaca if we have keys,
    else yfinance. (Demo/synthetic is selected explicitly by the caller.)
    """
    if settings.has_credentials:
        return from_alpaca(settings.ticker, settings.years, settings.api_key, settings.api_secret)
    return from_yfinance(settings.ticker, settings.years)


def get_ohlc(settings) -> pd.DataFrame:
    """Daily OHLC frame (Open/High/Low/Close) for candlesticks — Alpaca if keys,
    else yfinance."""
    if settings.has_credentials:
        return from_alpaca_ohlc(settings.ticker, settings.years, settings.api_key, settings.api_secret)
    return from_yfinance_ohlc(settings.ticker, settings.years)


# ── intraday (for the 1D / 1W timeframes) ────────────────────────────────────
# Intraday timeframes: bar count + pandas frequency for the synthetic series,
# yfinance (period, interval), and the Alpaca (minutes, lookback-days) pair.
_INTRADAY_TF = {
    # Yahoo caps 15-minute and finer history at 60 days, so the lookbacks below
    # stay inside that rather than asking for a window it will silently truncate.
    "15M": {"n": 130, "freq": "15min", "yf": ("5d", "15m"), "alpaca": (15, 7)},
    "30M": {"n": 130, "freq": "30min", "yf": ("1mo", "30m"), "alpaca": (30, 14)},
    "1H": {"n": 120, "freq": "1h", "yf": ("1mo", "1h"), "alpaca": (60, 30)},
    "4H": {"n": 120, "freq": "4h", "yf": ("3mo", "1h"), "alpaca": (240, 120)},
    "1D": {"n": 78, "freq": "5min", "yf": ("1d", "5m"), "alpaca": (5, 4)},
    "1W": {"n": 130, "freq": "30min", "yf": ("5d", "30m"), "alpaca": (30, 8)},
}


def _tf_spec(tf: str) -> dict:
    return _INTRADAY_TF.get((tf or "1D").upper(), _INTRADAY_TF["1D"])


def synthetic_intraday(tf: str = "1D", seed: int = 0) -> pd.DataFrame:
    """Deterministic intraday OHLC so the intraday views render fully offline."""
    tf = tf.upper()
    spec = _tf_spec(tf)
    n, freq = spec["n"], spec["freq"]
    rng = np.random.default_rng(seed + 7)
    idx = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n, freq=freq)
    rets = rng.normal(0.0001, 0.0035, n)
    price = 100.0 * np.exp(np.cumsum(rets))
    o = np.empty(n); o[0] = price[0]; o[1:] = price[:-1]
    c = price
    amp = np.abs(rng.normal(0.0, 0.0028, n)) + 0.0008
    hi = np.maximum(o, c) * (1.0 + amp)
    lo = np.minimum(o, c) * (1.0 - amp)
    # Intraday volume is U-shaped — heavy at the open and the close, thin over
    # lunch. Flat synthetic volume would make the demo VWAP sit on the mid-price
    # and hide the very behaviour a VWAP exists to show.
    frac = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)
    shape = 1.0 + 2.2 * (np.exp(-frac / 0.13) + np.exp(-(1.0 - frac) / 0.13))
    vol = shape * rng.lognormal(mean=0.0, sigma=0.25, size=n) * 20_000.0
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": c,
                         "Volume": np.round(vol)}, index=idx)


def _yfinance_intraday(ticker: str, tf: str) -> pd.DataFrame:
    import yfinance as yf

    tf = (tf or "1D").upper()
    period, interval = _tf_spec(tf)["yf"]
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    df = _ohlc_columns(df)
    if tf == "4H" and not df.empty:      # yfinance has no 4h bar — resample 1h
        df = df.resample("4h").agg({"Open": "first", "High": "max",
                                    "Low": "min", "Close": "last"}).dropna()
    return df


def _alpaca_intraday(ticker: str, tf: str, api_key: str, api_secret: str) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf = (tf or "1D").upper()
    minutes, days_back = _tf_spec(tf)["alpaca"]
    frame = TimeFrame(minutes, TimeFrameUnit.Minute)
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)).to_pydatetime()

    if "/" in ticker or (ticker.endswith("USD") and "-" in ticker):
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest

        symbol = ticker.replace("-", "/")
        client = CryptoHistoricalDataClient(api_key, api_secret)
        bars = client.get_crypto_bars(
            CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=frame, start=start)).df
        key = symbol
    else:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest

        client = StockHistoricalDataClient(api_key, api_secret)
        # Free and paper plans are not entitled to SIP, and cannot query the most
        # recent 15 minutes of it either — both surface as an opaque failure. Ask
        # for the default feed, then fall back to IEX, which every plan can read.
        end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)).to_pydatetime()
        try:
            bars = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=ticker, timeframe=frame, start=start)).df
            if bars is None or bars.empty:
                raise ValueError("no bars returned")
        except Exception:  # noqa: BLE001 — retry on the feed every account has
            from alpaca.data.enums import DataFeed

            bars = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=ticker, timeframe=frame, start=start, end=end,
                feed=DataFeed.IEX)).df
        key = ticker

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(key, level=0)
    df = _ohlc_columns(bars)
    if tf == "1D" and not df.empty:                 # 1D = the latest session only
        df = df[df.index >= df.index[-1].normalize()]
    return df


INTRADAY_SOURCES = ("auto", "yahoo", "alpaca")

# What each feed actually is, in one line, so the terminal can say it out loud
# rather than leaving the user to guess what "live" means.
SOURCE_NAME = {"yahoo": "Yahoo", "alpaca": "Alpaca"}
SOURCE_LABEL = {
    "yahoo": "Yahoo — consolidated tape, free, delayed ~15 min",
    "alpaca": "Alpaca — real-time on your data plan (IEX only on the free tier)",
}


def _intraday_order(settings) -> list[str]:
    """Which feeds to try, in order.

    Default is Yahoo first, and that is not an arbitrary preference. Alpaca's
    free data plan serves IEX only — around 2-3% of consolidated volume — so
    trying it first means that connecting a broker account makes the intraday
    bars *worse* than having no account at all. Yahoo carries the whole tape.
    Anyone paying for SIP wants the opposite, hence the setting.
    """
    pref = str(getattr(settings, "intraday_source", "auto") or "auto").lower()
    has_keys = bool(settings.has_credentials)
    if pref == "alpaca":
        return ["alpaca", "yahoo"] if has_keys else ["yahoo"]
    if pref == "yahoo":
        return ["yahoo", "alpaca"] if has_keys else ["yahoo"]
    return ["yahoo", "alpaca"] if has_keys else ["yahoo"]


def get_intraday_ohlc(settings, tf: str) -> pd.DataFrame:
    """Intraday OHLC (1H/4H/5-min for 1D/30-min for 1W).

    The feed you analyse and the venue you trade on are unrelated choices. This
    picks the feed; execution goes to Alpaca regardless, and Alpaca's order API
    neither knows nor cares which data produced the ticket.

    Intraday feeds fail for plenty of mundane reasons — plan entitlements, a
    symbol with no intraday history, a quiet session — so one source going down
    must not blank the chart. Raises with every reason only when none deliver.
    The frame carries the feed that served it in `.attrs["source"]`, because a
    chart that will not say where its bars came from is a chart you cannot check.
    """
    errors = []
    for name in _intraday_order(settings):
        try:
            if name == "alpaca":
                df = _alpaca_intraday(settings.ticker, tf,
                                      settings.api_key, settings.api_secret)
            else:
                df = _yfinance_intraday(settings.ticker, tf)
            if df is not None and not df.empty:
                df.attrs["source"] = name
                df.attrs["sourceLabel"] = SOURCE_LABEL[name]
                return df
            errors.append(f"{SOURCE_NAME[name]} returned no bars")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{SOURCE_NAME[name]}: {exc}")
    raise RuntimeError("; ".join(errors) or "no intraday data available")
