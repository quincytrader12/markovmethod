"""PyInstaller entry point for the mamba-terminal executable."""

import sys

from markov_hedge_fund_method.terminal import main

if __name__ == "__main__":
    sys.exit(main())
