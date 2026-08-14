"""Opportunity scanner — rank a universe by investment potential.

Reuses the per-symbol HUD payload (`AppState.state_payload`) so the scan rides
the same cache and the same Markov-regime brain the rest of the terminal uses.
For every symbol it computes a transparent 0–100 composite from five factors —
regime, forecast, trend, momentum/RSI and the model's own walk-forward edge —
assigns a Buy/Watch/Avoid verdict, and writes a short plain-English rationale
explaining *why*. Pure functions over the payload dict, so it is easy to test.

Not investment advice — a quantitative screen to surface candidates for review.
"""

from __future__ import annotations

import time

DISCLAIMER = ("Quantitative screen for research only — not investment advice. "
              "Scores blend the Markov regime model, forecast, trend, momentum "
              "and the strategy's own walk-forward record. Always do your own work.")


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _last(seq) -> float | None:
    """Last non-null value of a series list (chart series carry leading Nones)."""
    for v in reversed(seq or []):
        if v is not None:
            return float(v)
    return None


def score_payload(p: dict) -> dict:
    """Score one symbol's HUD payload → {score, verdict, factors, rationale, …}."""
    ticker = p.get("ticker", "?")
    name = p.get("name", "") or ""
    price = float(p.get("lastPrice") or 0.0)
    regime = p.get("regime", "sideways")
    stat = p.get("stationary") or [0.33, 0.34, 0.33]

    fc = {row["h"]: row for row in p.get("forecast", [])}
    bull5 = float(fc.get(5, {}).get("bull", stat[2]))
    bull20 = float(fc.get(20, {}).get("bull", stat[2]))
    bear5 = float(fc.get(5, {}).get("bear", stat[0]))

    chart = p.get("chart", {})
    ma20 = _last(chart.get("ma20"))
    ma50 = _last(chart.get("ma50"))
    rsi = _last(chart.get("rsi")) or 50.0
    mom = _last(chart.get("momentum")) or 0.0

    metrics = p.get("metrics", {})
    sharpe = metrics.get("sharpe")
    winrate = metrics.get("winRate")
    mdd = metrics.get("maxDrawdown")
    n_obs = int(metrics.get("nObs", 0) or 0)
    skew = float(metrics.get("skew", 0.0) or 0.0)
    kurt = float(metrics.get("kurtosis", 3.0) or 3.0)
    days = int(p.get("regimeTimeline", {}).get("daysInRegime", 0) or 0)
    greed = int(p.get("greedFear", {}).get("score", 50) or 50)

    factors: dict[str, float] = {}

    # 1) Regime (0–30): being in Bull is the biggest tailwind, weighted by how
    #    much of the long-run mix sits in Bull.
    if regime == "bull":
        factors["regime"] = 20.0 + 10.0 * _clip01(stat[2])
    elif regime == "sideways":
        factors["regime"] = 10.0 + 6.0 * _clip01(stat[2])
    else:
        factors["regime"] = 3.0 * _clip01(stat[2])

    # 2) Forecast (0–22): near-term odds of staying bullish, a bonus when the
    #    far horizon keeps leaning bull, a penalty for elevated near bear risk.
    fwd = 16.0 * _clip01(bull5) + 6.0 * _clip01((bull20 - bull5) + 0.5)
    fwd -= 6.0 * _clip01(bear5 - 0.34)
    factors["forecast"] = max(0.0, fwd)

    # 3) Trend (0–20): price stacked above its 20/50-day averages.
    if ma20 and ma50 and price:
        if price > ma20 > ma50:
            factors["trend"] = 20.0
        elif price > ma20 and price > ma50:
            factors["trend"] = 16.0
        elif price > ma20:
            factors["trend"] = 12.0
        elif price > ma50:
            factors["trend"] = 8.0
        else:
            factors["trend"] = 3.0
    else:
        factors["trend"] = 10.0

    # 4) Momentum / RSI (0–16): positive drift, with RSI in a healthy band
    #    (reward the sweet spot, punish overbought chasing).
    mom_pts = 8.0 * _clip01(0.5 + mom / 20.0)
    if 45.0 <= rsi <= 68.0:
        rsi_pts = 8.0
    elif 68.0 < rsi <= 78.0:
        rsi_pts = 5.0
    elif rsi > 78.0:
        rsi_pts = 1.0
    elif rsi >= 35.0:
        rsi_pts = 5.0
    else:
        rsi_pts = 2.0
    factors["momentum"] = mom_pts + rsi_pts

    # 5) Model edge (0–12): does the regime strategy actually work on this name?
    edge = 0.0
    if sharpe is not None:
        edge += 6.0 * _clip01((float(sharpe) + 0.5) / 2.5)   # ~ -0.5..2.0 → 0..1
    if winrate is not None:
        edge += 6.0 * _clip01((float(winrate) - 0.40) / 0.30)  # 0.40..0.70 → 0..1
    factors["modelEdge"] = edge

    score = sum(factors.values())
    if mdd is not None and float(mdd) < -0.40:   # deep historical drawdown
        score -= 5.0
    if greed >= 85:                               # extreme greed = late to chase
        score -= 3.0
    score = int(round(max(0.0, min(100.0, score))))

    if score >= 75:
        verdict = "Strong Buy"
    elif score >= 60:
        verdict = "Buy"
    elif score >= 45:
        verdict = "Watch"
    else:
        verdict = "Avoid"

    rationale = _rationale(name or ticker, regime, days, bull5, bull20,
                           price, ma20, ma50, mom, rsi, sharpe, winrate, mdd)

    return {
        "symbol": ticker,
        "name": name,
        "score": score,
        "verdict": verdict,
        "regime": regime,
        "lastPrice": round(price, 4),
        "forecastBull": round(bull5 * 100, 1),
        "rsi": round(rsi, 1),
        "momentum": round(mom, 2),
        "daysInRegime": days,
        "sharpe": None if sharpe is None else round(float(sharpe), 2),
        "winRate": None if winrate is None else round(float(winrate), 4),
        "maxDrawdown": None if mdd is None else round(float(mdd), 4),
        "psr": metrics.get("psr"),
        "nObs": n_obs,
        "skew": skew,
        "kurtosis": kurt,
        # filled in by deflate() once the whole scan is known
        "dsr": None,
        "edgeVerdict": None,
        "factors": {k: round(v, 1) for k, v in factors.items()},
        "rationale": rationale,
    }


def _rationale(label, regime, days, bull5, bull20, price, ma20, ma50,
               mom, rsi, sharpe, winrate, mdd) -> str:
    regime_txt = {
        "bull": "in a Bull regime",
        "bear": "in a Bear regime",
        "sideways": "range-bound (Sideways regime)",
    }.get(regime, "in a Sideways regime")

    parts = []
    held = f" and has held it for {days} day{'s' if days != 1 else ''}" if days else ""
    parts.append(f"{label} is {regime_txt}{held}.")
    tail = (", and the far-horizon mix keeps leaning bull"
            if bull20 >= bull5 else ", though the longer horizon cools off")
    parts.append(f"The model puts {round(bull5 * 100)}% odds on staying bullish "
                 f"over the next week{tail}.")

    if ma20 and ma50 and price:
        if price > ma20 > ma50:
            parts.append("Price is stacked above its 20- and 50-day averages — a clean uptrend.")
        elif price > ma20:
            parts.append("Price is holding above its 20-day average.")
        else:
            parts.append("Price is below its short-term average, so momentum is soft.")

    rsi_note = (" (overbought — watch for a pullback)" if rsi > 78
                else " (room before overbought)" if rsi < 68 else "")
    parts.append(f"20-day momentum is {mom:+.1f}% with RSI {rsi:.0f}{rsi_note}.")

    if sharpe is not None and winrate is not None:
        quality = ("strong" if (sharpe >= 1.0 and winrate >= 0.5)
                   else "mixed" if sharpe >= 0.0 else "weak")
        parts.append(f"The regime strategy's walk-forward record here is {quality} "
                     f"(Sharpe {float(sharpe):.2f}, win rate {round(float(winrate) * 100)}%).")

    if mdd is not None and float(mdd) < -0.40:
        parts.append(f"Risk note: it has seen a deep {abs(float(mdd)) * 100:.0f}% "
                     "peak-to-trough drawdown historically.")

    return " ".join(parts)


def score_symbols(state, symbols, *, workers: int = 16) -> list[dict]:
    """Score every symbol in parallel and return the full unfiltered list.

    This is the only expensive step, so callers cache its result and apply
    filters/sorting on top — changing a filter then costs nothing. Background
    warm-ups pass a small `workers` count so they cannot starve the UI.
    """
    from concurrent.futures import ThreadPoolExecutor

    seen, ordered = set(), []
    for raw in symbols:
        sym = str(raw).strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            ordered.append(sym)
    if not ordered:
        return []

    def one(sym):
        try:
            payload = state.state_payload(sym)
            # Never rank a name whose price history failed to download — its
            # score would be computed from the synthetic placeholder series.
            source = str(payload.get("dataSource", ""))
            if source.startswith("synthetic") and not getattr(state, "demo", False):
                return None
            return score_payload(payload)
        except Exception:  # noqa: BLE001 — one bad symbol must not sink the scan
            return None

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(ordered)))) as pool:
        results = [r for r in pool.map(one, ordered) if r]
    # Deflation needs the whole scan: the correction depends on how many names
    # were tried and how spread out their Sharpes are.
    return deflate(results)


def deflate(results: list[dict]) -> list[dict]:
    """Correct every Sharpe in a scan for the fact that we tried many names.

    Picking the best of N candidates inflates the winner's track record: the
    maximum of N noise draws grows roughly like sqrt(2*ln N). We measure how
    spread out the Sharpes actually are across this scan, work out the best
    score luck alone would produce over N trials, and ask each name to clear
    *that* bar instead of zero. `dsr` is the resulting P(the edge is real).
    """
    from .sharpe_stats import deannualize, deflated_sharpe, verdict

    scored = [r for r in results if r.get("sharpe") is not None and r.get("nObs", 0) > 2]
    n_trials = max(len(scored), 1)
    if len(scored) >= 2:
        vals = [deannualize(r["sharpe"]) for r in scored]
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    else:
        variance = 0.0

    for r in results:
        if r.get("sharpe") is None or r.get("nObs", 0) <= 2:
            r["dsr"], r["edgeVerdict"] = None, None
            continue
        p = deflated_sharpe(deannualize(r["sharpe"]), r["nObs"], n_trials, variance,
                            skew=r.get("skew", 0.0), kurtosis=r.get("kurtosis", 3.0))
        r["dsr"] = round(p, 4)
        r["edgeVerdict"] = verdict(p)
        r["nTrials"] = n_trials
    return results


def rank(results: list[dict], *, top: int = 20, fresh_days: int = 0,
         proven_only: bool = False, sort: str = "score") -> dict:
    """Filter + sort already-scored results. Pure and instant.

    Filters that make "early" findable:
      fresh_days  > 0     keep only names this many days (or fewer) into their
                          current regime — i.e. moves that just turned.
      proven_only         keep only names where the strategy has a real track
                          record (positive Sharpe and a win rate above 50%).
      sort                "score" (default) or "fresh" (newest regime first,
                          score breaking ties).
    """
    scanned = len(results)
    out = list(results)
    if fresh_days and fresh_days > 0:
        out = [r for r in out if 0 < r.get("daysInRegime", 0) <= fresh_days]
    if proven_only:
        # "Proven" now means the edge survives the multiple-testing correction —
        # a 95% chance it is real — not merely a positive-looking backtest.
        out = [r for r in out if (r.get("dsr") or 0.0) >= 0.95]

    if sort == "fresh":
        out.sort(key=lambda r: (r.get("daysInRegime", 9999), -r["score"]))
    else:
        out.sort(key=lambda r: r["score"], reverse=True)

    return {
        "scannedAt": time.strftime("%Y-%m-%d %H:%M"),
        "scanned": scanned,
        "matched": len(out),
        "sort": sort,
        "freshDays": fresh_days,
        "provenOnly": bool(proven_only),
        "results": out[: max(1, int(top))],
        "disclaimer": DISCLAIMER,
    }


def scan(state, symbols, *, top: int = 12, fresh_days: int = 0,
         proven_only: bool = False, sort: str = "score") -> dict:
    """Score a universe then rank it (score + filter in one call)."""
    return rank(score_symbols(state, symbols), top=top, fresh_days=fresh_days,
                proven_only=proven_only, sort=sort)
