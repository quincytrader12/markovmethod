#!/usr/bin/env bash
# =============================================================================
# markov-hedge-fund-method — installer (macOS / Linux)
# =============================================================================
# Installs the skill into ~/.claude/skills/markov-hedge-fund-method/ using uv
# (Astral's Python toolchain), pins Python 3.12, installs the core stack, tries
# the optional HMM layer, and runs the first demo on SPY 10y.
#
# Idempotent: an existing install is backed up (never deleted) before a fresh
# copy is laid down, so the script is safe to re-run after an interruption.
#
# Author: Quincy Gininda. Packaged as a Claude Code skill.
# =============================================================================
set -euo pipefail

SKILL_NAME="markov-hedge-fund-method"
SKILL_DIR="$HOME/.claude/skills/$SKILL_NAME"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "✓ Running in agent mode — $SKILL_NAME install starting."

# ── Phase 1.1 — OS detection ────────────────────────────────────────────────
UNAME_S="$(uname -s 2>/dev/null || echo Windows_NT)"
case "$UNAME_S" in
  Darwin) OS_KIND=mac ;;
  Linux)  OS_KIND=linux ;;
  *)      OS_KIND=windows ;;
esac
echo "OS detected: $OS_KIND"

# ── Phase 1.2 — idempotency: back up any existing install ───────────────────
if [ -d "$SKILL_DIR" ]; then
  STAMP="$(date +%Y%m%d-%H%M%S)"
  mv "$SKILL_DIR" "$HOME/.claude/skills/.$SKILL_NAME.bak.$STAMP"
  echo "Previous install backed up to ~/.claude/skills/.$SKILL_NAME.bak.$STAMP. Running fresh install."
fi

# ── Phase 1.3 — ensure uv is present ────────────────────────────────────────
if command -v uv >/dev/null 2>&1; then
  echo "✓ uv already installed"
else
  echo "Installing uv (Astral)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # refresh PATH for this shell
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ── Phase 2 — scaffold: copy the skill payload into place ───────────────────
mkdir -p "$SKILL_DIR/markov_hedge_fund_method" "$SKILL_DIR/data"
cp "$SRC_DIR/SKILL.md"        "$SKILL_DIR/"
cp "$SRC_DIR/pyproject.toml"  "$SKILL_DIR/"
cp "$SRC_DIR/.gitignore"      "$SKILL_DIR/" 2>/dev/null || true
cp "$SRC_DIR"/markov_hedge_fund_method/*.py "$SKILL_DIR/markov_hedge_fund_method/"

# ── Phase 2.3 — pin Python 3.12 and create the venv ─────────────────────────
cd "$SKILL_DIR"
uv python install 3.12
uv venv --python 3.12 .venv
uv run python --version

# ── Phase 3.1 — required core stack (failure here is a real failure) ────────
uv pip install "yfinance>=0.2" "numpy>=1.26" "pandas>=2.0" "scikit-learn>=1.4"

# ── Phase 3.2 — optional HMM layer (never fatal) ────────────────────────────
if uv pip install "hmmlearn>=0.3"; then
  echo "true" > .hmm_available
  echo "✓ HMM extension installed — both observable and hidden models available."
else
  echo "false" > .hmm_available
  echo "HMM extension skipped (optional); observable Markov model installed successfully."
  echo "The transition matrix, stationary distribution, and walk-forward backtest will all run normally."
  echo "To enable the HMM layer later, install a C++ toolchain (MSVC Build Tools on Windows) and re-run."
fi

# ── Phase 4 — first run on SPY 10y ──────────────────────────────────────────
uv run python -m markov_hedge_fund_method.run --ticker SPY --years 10

# ── Phase 5 — confirmation ──────────────────────────────────────────────────
HMM_STATE="skipped (optional)"
[ "$(cat .hmm_available 2>/dev/null)" = "true" ] && HMM_STATE="installed"
cat <<EOF
================================================================
 ✓ $SKILL_NAME skill installed at ~/.claude/skills/$SKILL_NAME/

 Installed:
   • Observable Markov model (transition matrix, n-step forecast,
     stationary distribution, walk-forward backtest)
   • HMM extension: $HMM_STATE

 First run on SPY 10y: complete.

 You can now ask Claude — in any Claude Code session — to:
   • run the markov-hedge-fund-method skill on AAPL
   • run the markov-hedge-fund-method skill on BTC-USD with a 60-day lookback
   • fit the HMM on QQQ

 Author: Quincy Gininda. Packaged as a Claude Code skill.
 Backtests are historical, not forward-looking.
================================================================
EOF
