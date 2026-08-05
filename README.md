# markov-hedge-fund-method

An **observable Markov regime model** for any ticker, packaged as a Claude Code skill,
plus a companion **TradingView PineScript** indicator.

The Python module:

- Fetches daily OHLCV for any ticker via `yfinance` (free, no key).
- Labels each day **Bull / Bear / Sideways** from a 20-day rolling return.
- Builds the transition matrix by maximum-likelihood counting.
- Forecasts *n*-step ahead by raising the matrix to powers (Chapman–Kolmogorov).
- Solves the stationary distribution (the long-run regime mix).
- Runs a **walk-forward backtest** — re-estimating the matrix at every step using
  only data available before that day — and reports Sharpe and max drawdown.
- Optionally fits a **Hidden Markov Model** via `hmmlearn` (Baum–Welch + Viterbi).
  If `hmmlearn` can't compile (e.g. Windows without MSVC build tools), the HMM
  layer is skipped cleanly and the observable model still works.

> Framework: **Roan (@RohOnChain)**. Packaged as a Claude Code skill by Lewis Jackson.
> Backtests are historical, not forward-looking.

## Install as a Claude Code skill

**Option A — one-line installer (macOS / Linux):**

```bash
git clone https://github.com/quincytrader12/markovmethod.git
cd markovmethod
./install.sh
```

This installs [`uv`](https://docs.astral.sh/uv/) if needed, pins Python 3.12,
installs the dependencies into an isolated `.venv`, copies the skill into
`~/.claude/skills/markov-hedge-fund-method/`, and runs the first demo on SPY 10y.

**Option B — manual:**

```bash
mkdir -p ~/.claude/skills/markov-hedge-fund-method
cp -r SKILL.md pyproject.toml markov_hedge_fund_method ~/.claude/skills/markov-hedge-fund-method/
cd ~/.claude/skills/markov-hedge-fund-method
uv venv --python 3.12 .venv
uv pip install "yfinance>=0.2" "numpy>=1.26" "pandas>=2.0" "scikit-learn>=1.4"
uv pip install "hmmlearn>=0.3"   # optional HMM layer
```

## Run it

From the skill directory:

```bash
uv run python -m markov_hedge_fund_method.run --ticker SPY --years 10
```

Flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--ticker` | `SPY` | Any Yahoo Finance symbol (`AAPL`, `BTC-USD`, `QQQ`, …). |
| `--years` | `10` | Years of daily history to fetch. |
| `--window` | `20` | Rolling-return window (trading days) for regime labels. |
| `--threshold` | `0.02` | Rolling-return threshold separating Bull / Bear from Sideways. |
| `--no-hmm` | off | Skip the HMM fit even if `hmmlearn` is available. |

Inside a Claude Code session you can just say things like *"run the
markov-hedge-fund-method skill on AAPL with a 60-day lookback"* or *"fit the HMM
on BTC-USD"*.

Every run prints: the header (ticker / date range / row count), the 3×3
transition matrix with its persistence diagonal, the stationary distribution,
and the walk-forward Sharpe + max drawdown. If `hmmlearn` is installed it also
prints the HMM regime mean returns.

## TradingView indicator

`tradingview/markov_regime.pine` is a Pine v5 overlay that recreates the same
idea live on-chart: it labels every bar Bull/Bear/Sideways from a rolling
log-return rule and prints the transition matrix and stationary distribution
as corner tables built from visible chart history. Paste it into the
TradingView Pine Editor and add it to a chart (BTCUSDT daily is a good start).

## Layout

```
.
├── SKILL.md                         # Claude Code skill manifest
├── pyproject.toml                   # Python 3.12 + dependency pins
├── install.sh                       # macOS / Linux installer
├── markov_hedge_fund_method/
│   ├── __init__.py
│   ├── regime.py                    # labels, transition matrix, stationary, backtest
│   ├── hmm_extension.py             # optional Gaussian HMM (lazy hmmlearn import)
│   └── run.py                       # CLI entry point
└── tradingview/
    └── markov_regime.pine           # Pine v5 on-chart companion
```

## Notes / caveats

- The walk-forward backtest is deliberately simple (sign of `P(Bull) − P(Bear)`
  from the current state, held one day) and takes **no** position sizing or
  transaction costs into account. It's a measurement of the regime signal, not a
  turnkey trading strategy.
- Baum–Welch (the HMM fit) converges to *local* maxima. For serious use, fit
  across several `random_state` seeds and keep the best by log-likelihood.
- Data comes from Yahoo Finance via `yfinance`; occasional empty responses are
  rate-limiting — the fetcher retries once, then asks you to try again shortly.
