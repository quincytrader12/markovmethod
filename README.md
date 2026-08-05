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

> Author: **Quincy Gininda**. Packaged as a Claude Code skill.
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

## Markov 2.0 — the three corrections

The observable model above is v1. `markov2.py` layers on three documented
fixes, and the live terminal uses them by default.

**Fix 1 — stride sampling (the autocorrelation flaw).** Labels come from a
20-day *rolling* return, so consecutive days share 19 of 20 days — counting
day-to-day transitions manufactures persistence on the diagonal. v2 builds
*both* matrices: the legacy overlapping one **and** a stride-sampled
(non-overlapping, stride = window) one, and shows them side by side. Only the
stride-sampled matrix is statistically honest. In practice the fake diagonal
runs ~90% while the true one is ~40% — a ~50-point illusion.

**Fix 2 — label verification.** Before any matrix/chart renders,
`verify_labels` checks the mapping against the data's own extremes: the
highest-return window must be Bull, the lowest Bear, the flattest Sideways. A
swapped Bull/Bear display fails this check instead of shipping.

**Fix 3 — two explicit modes.**
- **FILTER** (default): the regime *gates* a strategy — long only when the
  signal clears `+threshold`, short below `−threshold`, flat in chop.
- **STANDALONE**: trade the signal directly, position sized to `|signal|` up
  to a cap.

Generate the before/after proof (equity curves + the persistence bars):

```bash
markov-proof --ticker SPY                 # real data (needs network)
markov-proof --demo --image proof.png     # offline synthetic demo
```

> "Backtests flatter. The fixed matrix shows uglier, truer numbers — those are
> the only ones worth trading."

## Mamba Terminal (Alpaca)

A Textual TUI dashboard that runs the 2.0 model live and shows the honest
matrix, the Fix-1 comparison, the Fix-2 verification badge, the signal, and
the target position — plus your Alpaca account/positions when connected.

```bash
mamba-terminal --ticker SPY                    # dashboard, read-only
mamba-terminal --ticker AAPL --demo            # offline synthetic data
mamba-terminal --ticker SPY --strategy standalone
mamba-terminal --mode paper                    # paper auto-trade (needs keys)
mamba-terminal --save-keys <KEY_ID> <SECRET> --account swing   # save keys to a profile
mamba-terminal --account swing --mode paper    # run against a named account
```

**Execution is gated by mode.** `dashboard` (default) is read-only and never
places an order — it shows the *target* position only. `paper` and `live`
route the exact same logic through the broker seam (`broker.py`); the order
method is a hard error in dashboard mode, so trading is genuinely one flag
away rather than accidentally on. Keys are read from the environment or OS
keychain — never bundled into the binary. Install the extra with
`pip install -e ".[terminal]"` (add `viz` for the proof charts).

### Execution panel — manual order entry

The bottom of the terminal is a live **execution panel** that submits real
orders through the gated broker seam, covering the **full Alpaca order
taxonomy**:

| Field | Choices |
| --- | --- |
| **Type** | `market` · `limit` · `stop` · `stop_limit` · `trailing_stop` |
| **Class** | `simple` · `bracket` · `oco` · `oto` |
| **Time-in-force** | `day` · `gtc` · `opg` · `cls` · `ioc` · `fok` |
| **Side** | `buy` · `sell` |
| **Sizing** | `qty` (fractional shares) or `notional` ($, market/day only) |
| **Prices** | limit, stop, trailing (`$` or `%`), take-profit + stop-loss legs |
| **Flags** | extended-hours (limit orders) |

Fill the row, press **Submit** (or `s` to jump to the form); **Cancel all**
(or `x`) cancels every open order. Tickets are validated *before* they leave
the terminal — a bad combination (e.g. a limit order with no price, or a
notional bracket) is rejected with a plain-English reason in the log, so only
well-formed requests reach Alpaca. The panel is **inert in `dashboard`/demo
mode** and only arms under `--mode paper` or `--mode live`. Open orders are
listed in the account panel and refresh after every submit/cancel.

### Accounts — connect multiple portfolios

Each Alpaca account is a named **profile** with its own keys and a paper/live
flag. Keys are stored in the **OS keychain** (via `keyring`); only the profile
names + paper/live flags live in a small local registry
(`~/.config/mamba-terminal/accounts.json`, or `%APPDATA%\mamba-terminal\` on
Windows). Secrets are never written to disk in plaintext or bundled into the
exe.

**From the CLI:**

```bash
mamba-terminal --save-keys <KEY_ID> <SECRET> --account swing          # paper profile
mamba-terminal --save-keys <KEY_ID> <SECRET> --account live1 --live   # live-key profile
mamba-terminal --list-accounts                                        # show profiles (* = active)
mamba-terminal --remove-account swing                                 # delete a profile
mamba-terminal --account live1 --mode live                            # launch on a profile
```

**From inside the terminal:** press **`a`** to open the **Accounts screen**,
where you can add a profile (name / key / secret / paper), **Use** one to
switch the active portfolio, or **Remove** one. Switching rebuilds the live
broker on the fly, so you can flip between portfolios without restarting. The
connected account (and whether its keys are paper or live) is shown in the
account panel and the title bar.

The profile's paper/live flag selects the Alpaca **endpoint**; `--mode` still
independently gates whether orders are actually placed (`dashboard` stays
read-only regardless of which account is connected).

### Tests

```bash
pip install -e ".[terminal,test]"
pytest -q        # order-taxonomy + execution-panel wiring (headless Textual)
```

`tests/test_orders.py` builds every order type/class/TIF against the real
alpaca-py request classes; `tests/test_execution_panel.py` drives the TUI
headlessly with a fake broker to prove the panel collects a ticket, validates
it, and submits through the gated seam; `tests/test_accounts.py` and
`tests/test_accounts_tui.py` cover multi-account storage and live switching.

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
│   ├── regime.py                    # v1: labels, transition matrix (+stride), stationary, backtest
│   ├── markov2.py                   # 2.0: stride comparison, label verification, modes, metrics
│   ├── hmm_extension.py             # optional Gaussian HMM (lazy hmmlearn import)
│   ├── run.py                       # v1 CLI entry point
│   ├── engine.py                    # pure close-series -> Snapshot / SnapshotV2
│   ├── market_data.py               # Alpaca / yfinance / synthetic price sources
│   ├── config.py                    # Settings, execution Mode, Strategy, key handling
│   ├── broker.py                    # gated Alpaca execution seam (submit/cancel/list)
│   ├── orders.py                    # OrderTicket -> alpaca-py request (full taxonomy)
│   ├── accounts.py                  # named multi-account profiles (keychain-backed)
│   ├── tui.py                       # Textual dashboard + execution + accounts screen
│   ├── terminal.py                  # `mamba-terminal` entry point
│   └── proof.py                     # `markov-proof` before/after figure + metrics
├── tests/
│   ├── test_orders.py               # every order type/class/TIF builds correctly
│   ├── test_execution_panel.py      # headless TUI wiring (fake broker)
│   ├── test_accounts.py             # multi-account store (fake keyring)
│   └── test_accounts_tui.py         # accounts screen + live switching
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
