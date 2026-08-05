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


def from_yfinance(ticker: str, years: int = 10) -> pd.Series:
    from .run import _fetch_with_retry

    df = _fetch_with_retry(ticker, years)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna()


def from_alpaca(ticker: str, years: int, api_key: str, api_secret: str) -> pd.Series:
    """Daily bars from Alpaca's market-data API.

    Handles both equities and (dash-form) crypto symbols like BTC/USD.
    """
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
    close = bars["close"].dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = "Close"
    return close


def get_history(settings) -> pd.Series:
    """Pick a source based on available credentials. Alpaca if we have keys,
    else yfinance. (Demo/synthetic is selected explicitly by the caller.)
    """
    if settings.has_credentials:
        return from_alpaca(settings.ticker, settings.years, settings.api_key, settings.api_secret)
    return from_yfinance(settings.ticker, settings.years)
