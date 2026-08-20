"""Keep the test run out of the user's real config directory.

`AppState` builds a `PriceStore` pointed at the per-user config dir, which is
correct in the app and wrong in a test: a suite run would write price shards
into %APPDATA%\\mamba-terminal, and tests that depend on what is already stored
would then pass or fail according to what a previous run happened to leave
behind. Every test gets a throwaway config dir instead.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_config_dir(tmp_path, monkeypatch):
    import markov_hedge_fund_method.accounts as accounts
    import markov_hedge_fund_method.pricestore as pricestore

    home = tmp_path / "config"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(accounts, "default_config_dir", lambda: str(home))
    monkeypatch.setattr(pricestore, "default_config_dir", lambda: str(home))
    yield
