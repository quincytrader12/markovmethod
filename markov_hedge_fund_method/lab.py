"""The strategy lab: search every pairing, then refuse to believe the winner.

The search itself is the easy half. Enumerate the library, enumerate every way
two of its members can be paired, backtest all of them, sort by return. Any of
that can be written in an afternoon and it produces a number that is almost
always wrong.

Here is why, in one line: try three hundred and seventy-eight strategies with no
edge whatsoever on the same history, and the luckiest will show an annualised
Sharpe near two. Not because it found something — because three hundred and
seventy-eight draws from a distribution centred on zero have a maximum, and the
maximum is what a search reports. The equity curve will look wonderful. It will
be a picture of the noise in one particular slice of the past.

So this module spends most of its effort on not being fooled by its own output.

**Every trial is kept, including the failures.** The number of things tried is
the input the correction needs, and quietly dropping the ones that did badly is
how that number gets understated — which is the same as overstating the winner.

**Ranking is by deflated Sharpe, never by yield.** The Deflated Sharpe Ratio
(Bailey and López de Prado) asks: given this many attempts, and given how spread
out their results were, what is the probability this one is real? A strategy with
a lower raw return and a higher DSR is the better recommendation, and the lab
will say so.

**The last quarter of history is never searched.** Deflation fixes the
arithmetic of multiple testing; it cannot tell you whether the winner does
anything on data nobody optimised against. Only untouched data answers that, and
it can only answer once — every additional peek spends a little more of it, so
the number of looks is reported alongside the result.

The headline is the holdout. The yield is a footnote. That ordering is the whole
design, and it is the opposite of what a backtest engine usually shows you.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest as bt
from . import sharpe_stats as ss
from . import strategies as st

# How many of the ranked candidates get measured on the holdout. Every one of
# them spends a little of it: the more you look, the more the holdout becomes
# just another thing you have searched. Small on purpose.
HOLDOUT_LOOKS = 5

# A pairing is only worth testing if both halves cleared a floor on their own.
# Pairing two things that do not work is a very efficient way to manufacture
# trials, and trials are what the winner has to be deflated by.
PAIR_FLOOR = 0.0


@dataclass
class Trial:
    name: str
    kind: str                       # "single" or the pairing used
    parts: tuple
    track: bt.Track
    dsr: float = 0.0
    psr: float = 0.0

    def row(self) -> dict:
        d = {"name": self.name, "kind": self.kind, "parts": list(self.parts)}
        d.update(self.track.summary())
        d["dsr"] = round(self.dsr, 4)
        d["psr"] = round(self.psr, 4)
        d["verdict"] = ss.verdict(self.dsr)
        return d


@dataclass
class LabResult:
    symbol: str
    trials: list = field(default_factory=list)
    n_trials: int = 0
    sharpe_variance: float = 0.0
    holdout: list = field(default_factory=list)
    searched_bars: int = 0
    holdout_bars: int = 0
    cost_bps: float = bt.DEFAULT_COST_BPS
    ran_at: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "nTrials": self.n_trials,
            "sharpeVariance": round(self.sharpe_variance, 6),
            "searchedBars": self.searched_bars,
            "holdoutBars": self.holdout_bars,
            "costBps": self.cost_bps,
            "ranAt": self.ran_at,
            "seconds": round(self.seconds, 1),
            "holdoutLooks": len(self.holdout),
            "ranked": [t.row() for t in self.trials],
            "holdout": self.holdout,
        }


def _score(fn, close, cost_bps) -> bt.Track | None:
    try:
        return bt.run(fn(close), close, cost_bps=cost_bps)
    except Exception:  # noqa: BLE001 — a broken candidate is a non-result
        return None


def _candidates(lib: dict, close: pd.Series, cost_bps: float, pair: bool):
    """Every single, then every pairing of the singles worth pairing.

    Yields (name, kind, parts, fn) so the caller can score them uniformly.
    """
    singles = {}
    for name, fn in lib.items():
        t = _score(fn, close, cost_bps)
        if t is not None:
            singles[name] = (fn, t)
            yield name, "single", (name,), fn

    if not pair:
        return

    # Only pair things that stand up alone. Otherwise the combination count
    # explodes with candidates that exist only to inflate the trial count.
    worth = [n for n, (_, t) in singles.items() if t.sharpe > PAIR_FLOOR]
    for a, b in itertools.combinations(sorted(worth), 2):
        fa, fb = lib[a], lib[b]
        for kind, combiner in st.PAIRINGS.items():
            if kind == "scale":
                # Scaling is directional: A sized by B is not B sized by A.
                yield f"{a} scaled by {b}", kind, (a, b), combiner(fa, fb)
                yield f"{b} scaled by {a}", kind, (b, a), combiner(fb, fa)
            else:
                yield f"{a} + {b} ({kind})", kind, (a, b), combiner(fa, fb)


def search(close: pd.Series, *, symbol: str = "", cost_bps: float = bt.DEFAULT_COST_BPS,
           holdout_frac: float = 0.25, pair: bool = True, long_only: bool = False,
           progress=None) -> LabResult:
    """Run the whole lab over one symbol's history.

    `long_only` clips every candidate at zero. Worth having as a switch rather
    than an assumption: shorting a single equity carries a borrow cost this
    engine does not model and a day-trade profile the account may not want, so a
    short-side edge found here is less tradeable than it looks.
    """
    started = time.time()
    close = pd.Series(close).astype(float).dropna()
    searched, holdout = bt.split(close, holdout_frac)
    lib = st.library()
    if long_only:
        lib = {k: st.long_only(v) for k, v in lib.items()}

    trials: list[Trial] = []
    built: dict[str, object] = {}
    for name, kind, parts, fn in _candidates(lib, searched, cost_bps, pair):
        track = _score(fn, searched, cost_bps)
        if track is None or track.n_obs < 60:
            continue
        trials.append(Trial(name=name, kind=kind, parts=parts, track=track))
        built[name] = fn
        if progress is not None and len(trials) % 25 == 0:
            progress(len(trials))

    if not trials:
        return LabResult(symbol=symbol, ran_at=time.strftime("%Y-%m-%d %H:%M"),
                         cost_bps=cost_bps)

    # The two inputs the deflation needs: how many were tried, and how spread
    # out they were. Both come from the whole set — the losers included, which
    # is exactly why they were kept.
    sharpes = np.array([t.track.sharpe for t in trials], dtype=float)
    n_trials = len(sharpes)
    per_obs_all = sharpes / math.sqrt(bt.TRADING_DAYS)   # deannualise the whole set
    variance = float(np.var(per_obs_all, ddof=1)) if n_trials > 1 else 0.0

    for t in trials:
        per_obs = ss.deannualize(t.track.sharpe)
        t.psr = ss.probabilistic_sharpe(per_obs, t.track.n_obs,
                                        skew=t.track.skew, kurtosis=t.track.kurtosis)
        t.dsr = ss.deflated_sharpe(per_obs, t.track.n_obs, n_trials, variance,
                                   skew=t.track.skew, kurtosis=t.track.kurtosis)

    # Ranked by the probability the edge is real, not by what it paid. A lower
    # return with a higher DSR is the better recommendation and the lab says so.
    trials.sort(key=lambda t: (t.dsr, t.track.sharpe), reverse=True)

    # The holdout, spent sparingly on the top of the ranking.
    holdout_rows = []
    for t in trials[:HOLDOUT_LOOKS]:
        h = _score(built[t.name], holdout, cost_bps)
        if h is None:
            continue
        row = {"name": t.name, "kind": t.kind,
               "searched": t.track.summary(), "holdout": h.summary(),
               "dsr": round(t.dsr, 4),
               "heldUp": bool(h.sharpe > 0 and h.sharpe >= 0.5 * t.track.sharpe),
               "decay": round(t.track.sharpe - h.sharpe, 3)}
        holdout_rows.append(row)

    return LabResult(
        symbol=symbol, trials=trials, n_trials=n_trials,
        sharpe_variance=variance, holdout=holdout_rows,
        searched_bars=len(searched), holdout_bars=len(holdout),
        cost_bps=cost_bps, ran_at=time.strftime("%Y-%m-%d %H:%M"),
        seconds=time.time() - started,
    )


def summarise(result: LabResult) -> str:
    """What the lab actually concluded, in plain English."""
    if not result.trials:
        return "No strategy produced a usable track record on this history."

    best = result.trials[0]
    lines = [
        f"{result.n_trials} strategies and pairings tried on {result.searched_bars} "
        f"bars, {result.holdout_bars} held back.",
        f"Best by deflated Sharpe: {best.name} — {best.track.sharpe:.2f} Sharpe "
        f"after costs, DSR {best.dsr:.0%} ({ss.verdict(best.dsr)}).",
    ]
    if result.holdout:
        h = result.holdout[0]
        lines.append(
            f"On the untouched quarter it did {h['holdout']['sharpe']:.2f} "
            f"against {h['searched']['sharpe']:.2f} in the search — "
            + ("it held up." if h["heldUp"] else
               "it did not hold up, which is the more common outcome and the "
               "reason the holdout exists."))
    # The sentence that stops the whole thing being a machine for self-deception.
    lines.append(
        f"With {result.n_trials} attempts, a strategy with no edge at all would "
        f"be expected to show about "
        f"{ss.expected_max_sharpe(result.n_trials, result.sharpe_variance) * math.sqrt(252):.2f} "
        "annualised by luck alone. That is the bar, not zero.")
    return " ".join(lines)


# ── persistence ─────────────────────────────────────────────────────────────
def _path(config_dir: str) -> str:
    return os.path.join(config_dir, "lab_results.json")


def save(result: LabResult, config_dir: str, keep: int = 40) -> None:
    """Log the run. Every trial is kept, not just the winner.

    A record that stores only what worked cannot be audited later, and re-running
    a search whose losers were discarded understates the trial count next time —
    which quietly inflates every future winner.
    """
    path = _path(config_dir)
    try:
        os.makedirs(config_dir, exist_ok=True)
        blob = load_all(config_dir)
        blob = [b for b in blob if b.get("symbol") != result.symbol]
        blob.insert(0, result.to_dict())
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob[:keep], fh, indent=1)
    except OSError:
        pass


def load_all(config_dir: str) -> list:
    try:
        with open(_path(config_dir), encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def load(config_dir: str, symbol: str) -> dict | None:
    for blob in load_all(config_dir):
        if blob.get("symbol", "").upper() == symbol.strip().upper():
            return blob
    return None


def recommendations(config_dir: str, min_dsr: float = 0.5) -> list:
    """What the lab is prepared to stand behind, across every symbol it has run.

    Filtered on the holdout rather than on the search, because a strategy that
    only worked where it was fitted is exactly the thing this whole module
    exists to keep out of the terminal.
    """
    out = []
    for blob in load_all(config_dir):
        for h in blob.get("holdout", []):
            if h.get("dsr", 0) >= min_dsr and h.get("heldUp"):
                out.append({
                    "symbol": blob.get("symbol", ""),
                    "strategy": h.get("name", ""),
                    "kind": h.get("kind", ""),
                    "dsr": h.get("dsr", 0),
                    "searchedSharpe": h.get("searched", {}).get("sharpe"),
                    "holdoutSharpe": h.get("holdout", {}).get("sharpe"),
                    "cagr": h.get("holdout", {}).get("cagr"),
                    "maxDrawdown": h.get("holdout", {}).get("maxDrawdown"),
                    "turnover": h.get("holdout", {}).get("turnover"),
                    "nTrials": blob.get("nTrials", 0),
                    "ranAt": blob.get("ranAt", ""),
                })
    out.sort(key=lambda r: r["dsr"], reverse=True)
    return out


# ── going deeper on a candidate ─────────────────────────────────────────────
# Cost levels a strategy is re-run at. The question is not what it made at one
# assumed spread but at what spread it stops working, because that number says
# whether the edge has any room in it or is an artefact of an optimistic fill.
STRESS_BPS = (0.0, 2.5, 5.0, 10.0, 20.0, 40.0)


def cost_stress(fn, close: pd.Series, levels=STRESS_BPS) -> dict:
    """Where does this strategy stop making money as costs rise?

    A Sharpe of 1.2 that survives forty basis points is a different animal from
    one that dies at seven, and the single-cost number cannot tell them apart.
    """
    curve = []
    breakeven = None
    for bps in levels:
        t = _score(fn, close, bps)
        if t is None:
            continue
        curve.append({"bps": bps, "sharpe": round(t.sharpe, 3),
                      "cagr": round(t.cagr, 4)})
        if breakeven is None and t.sharpe <= 0:
            breakeven = bps
    return {"curve": curve,
            "breakevenBps": breakeven,
            "survivesRealCosts": bool(breakeven is None or breakeven > 10.0)}


def bootstrap_sharpe(returns: pd.Series, n: int = 1000, seed: int = 0) -> dict:
    """A confidence interval on the Sharpe, by resampling the returns.

    A point estimate from a few hundred observations is a very noisy thing to
    make a decision on. Resampling says how noisy: an interval spanning zero
    means the track record is consistent with having no edge, whatever the
    headline number was.
    """
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(r) < 30:
        return {"low": None, "high": None, "spansZero": True, "n": len(r)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n, len(r)))
    draws = r[idx]
    mu = draws.mean(axis=1)
    sd = draws.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = np.where(sd > 0, mu / sd * math.sqrt(bt.TRADING_DAYS), 0.0)
    lo, hi = np.percentile(sh, [5, 95])
    return {"low": round(float(lo), 3), "high": round(float(hi), 3),
            "spansZero": bool(lo <= 0 <= hi), "n": int(len(r))}


def by_regime(fn, close: pd.Series, cost_bps: float = bt.DEFAULT_COST_BPS,
              window: int = 20, threshold: float = 0.02) -> dict:
    """How the strategy did in each of the terminal's own three regimes.

    A strategy that makes everything in bull markets and gives it back in bear
    ones is a leveraged long wearing a costume, and the blended Sharpe hides
    that completely. This is the question the rest of the terminal is built
    around, so the lab should answer it in the same vocabulary.
    """
    close = pd.Series(close).astype(float).dropna()
    track = _score(fn, close, cost_bps)
    if track is None:
        return {}
    roll = close.pct_change(window)
    state = pd.Series("sideways", index=close.index)
    state[roll > threshold] = "bull"
    state[roll < -threshold] = "bear"

    out = {}
    for name in ("bull", "sideways", "bear"):
        r = track.returns[state == name]
        if len(r) < 20:
            out[name] = {"days": int(len(r)), "sharpe": None, "total": None}
            continue
        sd = float(r.std(ddof=1))
        out[name] = {
            "days": int(len(r)),
            "sharpe": round(float(r.mean() / sd * math.sqrt(bt.TRADING_DAYS)), 3) if sd > 0 else 0.0,
            "total": round(float((1 + r).prod() - 1.0), 4),
        }
    return out


def deep_dive(symbol: str, name: str, close: pd.Series, *,
              cost_bps: float = bt.DEFAULT_COST_BPS,
              holdout_frac: float = 0.25) -> dict:
    """Everything worth knowing about one candidate, on the holdout.

    Rebuilt from the library rather than stored, so a dive always reflects the
    strategy as it is defined now rather than as it was when the search ran.
    """
    searched, holdout = bt.split(pd.Series(close).astype(float).dropna(), holdout_frac)
    fn = rebuild(name)
    if fn is None:
        return {"error": f"{name!r} is no longer in the library"}
    track = _score(fn, holdout, cost_bps)
    if track is None:
        return {"error": "no track record on the holdout"}
    return {
        "symbol": symbol, "strategy": name,
        "holdout": track.summary(),
        "confidence": bootstrap_sharpe(track.returns),
        "costStress": cost_stress(fn, holdout),
        "byRegime": by_regime(fn, holdout, cost_bps),
    }


def rebuild(name: str):
    """Turn a stored strategy name back into a function.

    Names are generated by `_candidates`, so parsing them back is the inverse of
    one function and is kept beside it deliberately.
    """
    lib = st.library()
    if name in lib:
        return lib[name]
    if " scaled by " in name:
        a, b = name.split(" scaled by ", 1)
        if a in lib and b in lib:
            return st.scale(lib[a], lib[b])
        return None
    if name.endswith(")") and " + " in name:
        body, kind = name.rsplit(" (", 1)
        kind = kind[:-1]
        a, b = body.split(" + ", 1)
        combiner = st.PAIRINGS.get(kind)
        if combiner and a in lib and b in lib:
            return combiner(lib[a], lib[b])
    return None
