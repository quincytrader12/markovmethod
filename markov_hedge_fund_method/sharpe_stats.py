"""Is a Sharpe ratio real, or did it just look good?

Two corrections from Bailey & López de Prado, both cheap and both aimed at the
same failure mode: a backtest that flatters itself.

  Probabilistic Sharpe Ratio (PSR)
      P(true Sharpe > benchmark), given how long the sample is and how
      non-normal the returns are. A Sharpe of 2 over 60 days is not the same
      claim as a Sharpe of 2 over 6 years, and fat tails make it weaker still.

  Deflated Sharpe Ratio (DSR)
      PSR with the benchmark raised to the best score you would *expect* from
      pure luck after trying N strategies. Scanning 100 tickers and taking the
      winner is 100 trials: the maximum of N noise draws grows like
      sqrt(2*ln(N)), so the top name is upward-biased by construction.

Uses statistics.NormalDist from the stdlib — no SciPy, so nothing extra has to
be bundled into the frozen executable.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_N = NormalDist()
EULER = 0.5772156649015329          # Euler-Mascheroni constant
TRADING_DAYS = 252


def deannualize(sharpe_annual: float, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe -> per-observation Sharpe (what the maths expects)."""
    return float(sharpe_annual) / math.sqrt(periods)


def probabilistic_sharpe(sharpe: float, n_obs: int, *, benchmark: float = 0.0,
                         skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """P(true Sharpe > benchmark). `sharpe` is per-observation, not annualised.

    kurtosis is the raw fourth moment (3.0 for a normal distribution).
    """
    if n_obs is None or n_obs < 2:
        return 0.0
    denom = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe
    if denom <= 0:
        return 0.0
    z = (sharpe - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(_N.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """The Sharpe you would expect from the luckiest of `n_trials` strategies
    that have no edge at all. This is the bar a real edge has to clear."""
    n = max(2, int(n_trials))
    sd = math.sqrt(max(sharpe_variance, 0.0))
    if sd <= 0:
        return 0.0
    a = _N.inv_cdf(1.0 - 1.0 / n)
    b = _N.inv_cdf(1.0 - 1.0 / (n * math.e))
    return float(sd * ((1.0 - EULER) * a + EULER * b))


def deflated_sharpe(sharpe: float, n_obs: int, n_trials: int,
                    sharpe_variance: float, *, skew: float = 0.0,
                    kurtosis: float = 3.0) -> float:
    """P(the edge is real) after accounting for how many candidates were tried."""
    benchmark = expected_max_sharpe(n_trials, sharpe_variance)
    return probabilistic_sharpe(sharpe, n_obs, benchmark=benchmark,
                                skew=skew, kurtosis=kurtosis)


def verdict(p: float) -> str:
    """Plain-English reading of a PSR/DSR probability."""
    if p >= 0.95:
        return "strong evidence"
    if p >= 0.80:
        return "some evidence"
    if p >= 0.50:
        return "inconclusive"
    return "likely luck"
