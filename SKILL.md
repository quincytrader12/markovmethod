---
name: markov-hedge-fund-method
description: Observable Markov regime model for any ticker. Builds the transition matrix from a 20-day rolling-return regime label (Bull / Bear / Sideways), forecasts n-step ahead via matrix power, solves the stationary distribution, and runs a walk-forward backtest reporting Sharpe and max drawdown. Optional Hidden Markov Model upgrade via hmmlearn.
---

# markov-hedge-fund-method

Install location: `~/.claude/skills/markov-hedge-fund-method/`.
Author of the underlying framework: Roan (@RohOnChain). Installed as a Claude Code skill by Lewis Jackson.

## Invocation

Natural language. Examples the user may say in Claude Code:

- "run the markov-hedge-fund-method skill on SPY"
- "run the markov-hedge-fund-method skill on AAPL with a 60-day lookback"
- "fit the HMM on BTC-USD"

To run the skill, execute the module from within the skill directory using its pinned environment:

```
cd ~/.claude/skills/markov-hedge-fund-method
uv run python -m markov_hedge_fund_method.run --ticker <TICKER> [--years 10] [--window 20] [--no-hmm]
```

Default ticker is `SPY`. Default lookback is `10` years of daily data. Default rolling window for regime labels is `20` trading days.

## Outputs printed on every run

1. Header showing the ticker, date range, and row count.
2. The 3×3 transition matrix (Bull / Bear / Sideways) with the persistence diagonal labelled.
3. The stationary distribution (long-run baseline regime mix).
4. Walk-forward Sharpe and max drawdown from a re-estimated-at-every-step backtest.
5. Optional HMM regime mean returns if `hmmlearn` is available.

## Dependencies

`uv`-managed virtual environment under `.venv/` with Python 3.12 and:

- `yfinance>=0.2`
- `numpy>=1.26`
- `pandas>=2.0`
- `scikit-learn>=1.4`
- `hmmlearn>=0.3` (optional — graceful degrade if not installed)

The skill writes no credentials, reads no environment variables, makes no network calls beyond `yfinance` → Yahoo Finance.
