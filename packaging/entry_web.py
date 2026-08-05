"""PyInstaller entry point for the mamba-web executable."""

import sys

from markov_hedge_fund_method.web import main

if __name__ == "__main__":
    sys.exit(main())
