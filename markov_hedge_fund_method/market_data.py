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
    out = pd.DataFrame({
        "Open": df[cols["open"]], "High": df[cols["high"]],
        "Low": df[cols["low"]], "Close": df[cols["close"]],
    }).dropna()
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
def synthetic_intraday(tf: str = "1D", seed: int = 0) -> pd.DataFrame:
    """Deterministic intraday OHLC so the 1D / 1W views render fully offline."""
    tf = tf.upper()
    n, freq = (78, "5min") if tf == "1D" else (130, "30min")
    rng = np.random.default_rng(seed + 7)
    idx = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n, freq=freq)
    rets = rng.normal(0.0001, 0.0035, n)
    price = 100.0 * np.exp(np.cumsum(rets))
    o = np.empty(n); o[0] = price[0]; o[1:] = price[:-1]
    c = price
    amp = np.abs(rng.normal(0.0, 0.0028, n)) + 0.0008
    hi = np.maximum(o, c) * (1.0 + amp)
    lo = np.minimum(o, c) * (1.0 - amp)
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": c}, index=idx)


def _yfinance_intraday(ticker: str, tf: str) -> pd.DataFrame:
    import yfinance as yf

    period, interval = ("1d", "5m") if tf.upper() == "1D" else ("5d", "30m")
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    return _ohlc_columns(df)


def _alpaca_intraday(ticker: str, tf: str, api_key: str, api_secret: str) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf = tf.upper()
    if tf == "1D":
        frame, back = TimeFrame(5, TimeFrameUnit.Minute), pd.Timedelta(days=4)
    else:
        frame, back = TimeFrame(30, TimeFrameUnit.Minute), pd.Timedelta(days=8)
    start = (pd.Timestamp.now(tz="UTC") - back).to_pydatetime()

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
        bars = client.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=ticker, timeframe=frame, start=start)).df
        key = ticker

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(key, level=0)
    df = _ohlc_columns(bars)
    if tf == "1D" and not df.empty:                 # 1D = the latest session only
        df = df[df.index >= df.index[-1].normalize()]
    return df


def get_intraday_ohlc(settings, tf: str) -> pd.DataFrame:
    """Intraday OHLC (5-min for 1D, 30-min for 1W) — Alpaca if keys, else yfinance."""
    if settings.has_credentials:
        return _alpaca_intraday(settings.ticker, tf, settings.api_key, settings.api_secret)
    return _yfinance_intraday(settings.ticker, tf)
