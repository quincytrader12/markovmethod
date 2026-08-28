"""Heatmaps — the whole board at once, instead of one symbol at a time.

The rest of the terminal answers "what is this asset doing?". These answer the
questions you can only see across a group:

  * **Regime map** — every symbol as a row, recent weeks as columns, coloured by
    regime. Rotations show up as a diagonal wash across the grid long before any
    single chart looks different, and a market that has gone risk-off everywhere
    at once looks nothing like one where two names rolled over.

  * **Signal map** — today's conviction across the universe, ranked. The scanner
    already finds the best names; this shows the shape of the whole distribution,
    which is what tells you whether a signal of 0.30 is exceptional or ordinary
    on this particular day.

  * **Correlation** — pairwise correlation of daily returns. The most expensive
    mistake a watchlist can hide is being one bet wearing eight tickers, and
    that is invisible until you look at it as a matrix.

Everything here reads from cached per-symbol state, so a heatmap costs a few
milliseconds of arithmetic rather than a round of downloads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_KEY = {0: "bear", 1: "sideways", 2: "bull"}


def regime_grid(closes: dict, labels: dict, *, buckets: int = 26,
                bucket_days: int = 5) -> dict:
    """Symbols x time-buckets, each cell the regime that dominated the bucket.

    Weekly buckets by default: daily columns would be unreadable at 100 symbols,
    and a regime that only survives a day or two is noise anyway. Each cell also
    carries how lopsided the bucket was, so a week that was 5/5 bull reads as
    solid and one that was 3/2 reads as contested.
    """
    rows = []
    dates: list[str] = []
    for symbol in sorted(closes):
        lab = labels.get(symbol)
        if lab is None or lab.empty:
            continue
        tail = lab.iloc[-(buckets * bucket_days):]
        cells = []
        stamps = []
        for i in range(0, len(tail), bucket_days):
            chunk = tail.iloc[i: i + bucket_days]
            if chunk.empty:
                continue
            vals = chunk.to_numpy().astype(int)
            counts = np.bincount(vals, minlength=3)
            top = int(np.argmax(counts))
            cells.append({"regime": REGIME_KEY[top],
                          "strength": round(float(counts[top] / len(vals)), 3)})
            stamps.append(chunk.index[-1].strftime("%Y-%m-%d"))
        if not cells:
            continue
        rows.append({"symbol": symbol, "cells": cells,
                     "current": cells[-1]["regime"]})
        if len(stamps) > len(dates):
            dates = stamps
    return {"dates": dates, "rows": rows}


def signal_map(states: list) -> dict:
    """Today's signal for every symbol, ranked, with the distribution's shape.

    The percentile is the part worth reading: a signal is only strong relative
    to what the rest of the board is doing today.
    """
    cells = []
    for s in states:
        sig = s.get("signal")
        if sig is None:
            continue
        cells.append({
            "symbol": s.get("ticker", ""),
            "name": s.get("name", ""),
            "signal": round(float(sig), 4),
            "regime": s.get("regime", "unknown"),
            "confidence": round(float((s.get("signalStats") or {}).get("confidence", 0.0)), 4),
            "n": int((s.get("signalStats") or {}).get("n", 0)),
            "real": not str(s.get("dataSource", "")).startswith("synthetic"),
        })
    if not cells:
        return {"cells": [], "median": None, "spread": None, "bullish": 0, "bearish": 0}
    cells.sort(key=lambda c: -c["signal"])
    vals = np.array([c["signal"] for c in cells], dtype=float)
    ranks = np.argsort(np.argsort(-vals))
    for c, r in zip(cells, ranks):
        c["percentile"] = round(float(1.0 - r / max(len(vals) - 1, 1)), 3)
    return {
        "cells": cells,
        "median": round(float(np.median(vals)), 4),
        "spread": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4),
        "bullish": int((vals > 0.05).sum()),
        "bearish": int((vals < -0.05).sum()),
    }


def correlation_matrix(closes: dict, *, window: int = 126, max_symbols: int = 40) -> dict:
    """Pairwise return correlation over a trailing window.

    Only overlapping dates are used, so a symbol with a shorter history narrows
    the comparison rather than poisoning it with filled-in values. The average
    off-diagonal correlation is reported too — that single number is the honest
    answer to "how many bets is this watchlist really?".
    """
    frames = {}
    for symbol, close in closes.items():
        if close is None or len(close) < 30:
            continue
        frames[symbol] = close.pct_change().iloc[-window:]
        if len(frames) >= max_symbols:
            break
    if len(frames) < 2:
        return {"symbols": list(frames), "matrix": [], "avgCorrelation": None,
                "clusters": []}

    df = pd.DataFrame(frames).dropna(how="all")
    df = df.dropna(axis=1, thresh=max(20, int(len(df) * 0.5)))
    if df.shape[1] < 2:
        return {"symbols": list(df.columns), "matrix": [], "avgCorrelation": None,
                "clusters": []}
    corr = df.corr(min_periods=20)
    syms = [str(c) for c in corr.columns]
    m = corr.to_numpy(dtype=float)
    m = np.where(np.isfinite(m), m, 0.0)

    off = m[~np.eye(len(m), dtype=bool)]
    avg = float(off.mean()) if off.size else 0.0

    # The tightest pairs, which is where concentration actually bites.
    pairs = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            pairs.append({"a": syms[i], "b": syms[j], "corr": round(float(m[i, j]), 3)})
    pairs.sort(key=lambda p: -abs(p["corr"]))

    return {
        "symbols": syms,
        "matrix": [[round(float(v), 3) for v in row] for row in m],
        "avgCorrelation": round(avg, 3),
        "clusters": pairs[:12],
        "window": int(min(window, len(df))),
    }
