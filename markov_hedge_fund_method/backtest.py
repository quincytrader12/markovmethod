"""A general backtest engine: positions in, an honest track record out.

The existing walk-forward in `regime.py` answers one question about one strategy.
This answers the same question about any strategy, so a library of them can be
compared on identical terms — which is the only way a comparison means anything.

Two decisions here do most of the work.

**Positions are lagged.** A signal computed from today's close cannot be traded
until tomorrow. Skipping that lag is the most common way a backtest invents
money, and it is invisible in the output: the equity curve simply looks good.
Every position series is shifted by one bar before it meets a return, once, here,
so no strategy in the library can forget to do it.

**Costs are charged on turnover, always.** A search that ignores costs will
reliably crown the highest-turnover strategy in the library, because noise
trading looks like edge until it has to pay a spread. There is no cost-free mode
— the default is deliberately pessimistic rather than optional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# What a round trip costs, in basis points of notional. Retail equities through
# a commission-free broker still pay the spread and some slippage; a penny on a
# fifty-dollar share is two basis points each way. Five is not pessimistic.
DEFAULT_COST_BPS = 5.0


@dataclass
class Track:
    """One strategy's record over one period."""

    returns: pd.Series
    equity: pd.Series
    sharpe: float
    cagr: float
    max_drawdown: float
    win_rate: float
    turnover: float
    cost_drag: float          # annualised return given up to costs
    n_obs: int
    skew: float
    kurtosis: float
    exposure: float           # fraction of days holding anything
    gross_sharpe: float       # before costs, to show what the costs took
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "sharpe": round(self.sharpe, 3),
            "grossSharpe": round(self.gross_sharpe, 3),
            "cagr": round(self.cagr, 4),
            "maxDrawdown": round(self.max_drawdown, 4),
            "winRate": round(self.win_rate, 4),
            "turnover": round(self.turnover, 2),
            "costDrag": round(self.cost_drag, 4),
            "exposure": round(self.exposure, 3),
            "nObs": self.n_obs,
            "skew": round(self.skew, 3),
            "kurtosis": round(self.kurtosis, 3),
        }


def _moments(r: np.ndarray) -> tuple[float, float]:
    """Skew and raw kurtosis — PSR needs both, and a strategy's returns are
    never normal enough to assume them away."""
    if len(r) < 4:
        return 0.0, 3.0
    sd = r.std(ddof=1)
    if sd <= 0:
        return 0.0, 3.0
    z = (r - r.mean()) / sd
    return float((z ** 3).mean()), float((z ** 4).mean())


def run(positions: pd.Series, close: pd.Series, *,
        cost_bps: float = DEFAULT_COST_BPS,
        lag: int = 1, meta: dict | None = None) -> Track:
    """Score a position series against a price series.

    `positions` is the desired exposure per bar, -1 to +1, computed from data
    available *at* that bar. The lag is applied here, so a strategy author never
    has to remember it and cannot get it wrong.
    """
    close = pd.Series(close).astype(float).dropna()
    pos = pd.Series(positions).astype(float).reindex(close.index).fillna(0.0)

    # The one line that separates a backtest from a fantasy.
    held = pos.shift(lag).fillna(0.0)

    ret = close.pct_change().fillna(0.0)
    gross = held * ret

    # Turnover is how much the position moved, and it is what a spread is paid
    # on. A strategy that flips daily pays a hundred times more than one that
    # holds for a hundred days, which is the whole reason costs decide searches.
    traded = held.diff().abs().fillna(held.abs())
    costs = traded * (cost_bps / 10_000.0)
    net = gross - costs

    n = int(len(net))
    r = net.to_numpy()
    g = gross.to_numpy()
    sd = float(r.std(ddof=1)) if n > 1 else 0.0
    gsd = float(g.std(ddof=1)) if n > 1 else 0.0

    equity = (1.0 + net).cumprod()
    peak = equity.cummax()
    dd = float(((equity / peak) - 1.0).min()) if n else 0.0

    years = n / TRADING_DAYS if n else 0.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else 0.0

    traded_days = net[held.abs() > 1e-9]
    skew, kurt = _moments(r)

    return Track(
        returns=net,
        equity=equity,
        sharpe=float(r.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
        gross_sharpe=float(g.mean() / gsd * math.sqrt(TRADING_DAYS)) if gsd > 0 else 0.0,
        cagr=cagr,
        max_drawdown=dd,
        win_rate=float((traded_days > 0).mean()) if len(traded_days) else 0.0,
        turnover=float(traded.sum() / years) if years > 0 else 0.0,
        cost_drag=float(costs.sum() / years) if years > 0 else 0.0,
        n_obs=n,
        skew=skew,
        kurtosis=kurt,
        exposure=float((held.abs() > 1e-9).mean()) if n else 0.0,
        meta=dict(meta or {}),
    )


def split(close: pd.Series, holdout_frac: float = 0.25) -> tuple[pd.Series, pd.Series]:
    """Cut history into a search half and a holdout the search never sees.

    This is the only defence against a search that tries hundreds of
    combinations and reports the luckiest. Deflating the Sharpe by the trial
    count corrects the arithmetic; a holdout answers the different and blunter
    question of whether the winner does anything on data nobody optimised
    against. Both are reported, because they can disagree and the disagreement
    is the interesting part.
    """
    close = pd.Series(close).dropna()
    n = len(close)
    cut = int(n * (1.0 - max(0.05, min(holdout_frac, 0.5))))
    return close.iloc[:cut], close.iloc[cut:]


def is_causal(strategy_fn, close: pd.Series, *, checks: int = 5) -> bool:
    """Does this strategy's position at time t change when the future arrives?

    Everything in the library is required to be a pure function of the past, so
    that `strategy(whole_history)` can be evaluated in one pass without leaking.
    That property is easy to state, easy to violate by accident — a centred
    rolling window, a normalisation over the full series, a `bfill` — and
    invisible in the equity curve, which just looks better. So it is checked
    rather than assumed.
    """
    close = pd.Series(close).astype(float).dropna()
    if len(close) < 60:
        return True
    full = pd.Series(strategy_fn(close)).astype(float)
    n = len(close)
    for k in range(1, checks + 1):
        cut = n - k * max(1, n // (checks * 4))
        if cut < 30:
            break
        partial = pd.Series(strategy_fn(close.iloc[:cut])).astype(float)
        a = full.iloc[:cut].to_numpy()
        b = partial.reindex(close.index[:cut]).to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        if both.any() and not np.allclose(a[both], b[both], atol=1e-9):
            return False
    return True


def walk_forward_choice(candidates: dict, close: pd.Series, *, train: int = 504,
                        step: int = 63, cost_bps: float = DEFAULT_COST_BPS):
    """Pick among candidate strategies using only the past, again and again.

    This is where a search has to be honest. Scoring every candidate on all of
    history and reporting the best is not a backtest of a strategy, it is a
    backtest of hindsight: the choice itself used the future. Here the choice is
    remade every `step` bars from the preceding data alone, and whatever it
    picked is what gets traded through the block that follows — including the
    blocks where it picks badly, which is the point.

    The candidates must be causal (see `is_causal`), so applying the winner
    across the block uses only data available at each bar within it.
    """
    close = pd.Series(close).astype(float).dropna()
    pos = pd.Series(0.0, index=close.index)
    picks: list[dict] = []
    if not candidates or len(close) < train + 2:
        return run(pos, close, cost_bps=cost_bps), picks

    for start in range(train, len(close), step):
        block = close.index[start:start + step]
        if len(block) == 0:
            break
        past = close.iloc[:start]

        best_name, best_fn, best_sharpe = None, None, -math.inf
        for name, fn in candidates.items():
            try:
                t = run(fn(past), past, cost_bps=cost_bps)
            except Exception:  # noqa: BLE001 — a broken candidate is not a pick
                continue
            if t.sharpe > best_sharpe:
                best_name, best_fn, best_sharpe = name, fn, t.sharpe
        if best_fn is None:
            continue

        applied = pd.Series(best_fn(close.iloc[:start + len(block)])).astype(float)
        pos.loc[block] = applied.reindex(block).fillna(0.0)
        picks.append({"from": str(block[0].date()), "to": str(block[-1].date()),
                      "chose": best_name, "trainSharpe": round(best_sharpe, 3)})

    track = run(pos, close, cost_bps=cost_bps,
                meta={"walkForward": True, "train": train, "step": step,
                      "switches": len({p["chose"] for p in picks})})
    return track, picks
