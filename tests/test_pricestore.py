"""The on-disk price store, and the incremental refresh it makes possible.

The sweep used to re-download ten years of history for the whole market on every
pass, to learn about one new trading day. These tests cover the store that makes
that unnecessary — and, more importantly, the ways it could silently corrupt
prices, which is far worse than being slow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.market_data import synthetic_ohlc
from markov_hedge_fund_method.pricestore import PriceStore, _shard_for
from markov_hedge_fund_method.web import AppState


@pytest.fixture
def store(tmp_path):
    return PriceStore(config_dir=str(tmp_path))


# ── the round trip must be exact where it matters ───────────────────────────
def test_dates_survive_the_round_trip(store):
    """The bug this exists to prevent: pandas 3 stores datetimes at microsecond
    resolution where pandas 2 used nanoseconds, so a hard-coded divisor turned
    2026 dates into 1970 — which made every symbol look stale forever."""
    df = synthetic_ohlc(500, seed=1)
    store.put("TEST", df)
    back = store.get("TEST")
    assert (back.index == df.index).all()
    assert store.last_date("TEST") == df.index[-1]
    assert store.last_date("TEST").year >= 2000, "dates fell back to the epoch"


def test_prices_survive_the_round_trip(store):
    df = synthetic_ohlc(500, seed=2)
    store.put("TEST", df)
    back = store.get("TEST")
    for col in ("Open", "High", "Low", "Close"):
        # float32 storage: exact to about seven digits, far tighter than the
        # engine's own 2% labelling threshold.
        assert np.allclose(back[col], df[col], rtol=1e-6)


def test_row_count_is_preserved(store):
    df = synthetic_ohlc(1200, seed=3)
    store.put("TEST", df)
    assert len(store.get("TEST")) == len(df)


def test_an_unknown_symbol_reads_as_nothing(store):
    assert store.get("NOPE") is None
    assert store.last_date("NOPE") is None


# ── appending ───────────────────────────────────────────────────────────────
def test_appending_extends_rather_than_replaces(store):
    df = synthetic_ohlc(500, seed=4)
    store.put("TEST", df.iloc[:-10])
    assert len(store.get("TEST")) == 490
    store.put("TEST", df.iloc[-10:])
    back = store.get("TEST")
    assert len(back) == 500
    assert back.index[-1] == df.index[-1]


def test_overlapping_appends_do_not_duplicate_days(store):
    df = synthetic_ohlc(300, seed=5)
    store.put("TEST", df)
    store.put("TEST", df.iloc[-50:])          # re-send the tail
    back = store.get("TEST")
    assert len(back) == 300
    assert not back.index.duplicated().any()


def test_a_later_append_wins_on_a_revised_bar(store):
    """Exchanges do revise bars. The newer value must replace the older one."""
    df = synthetic_ohlc(100, seed=6)
    store.put("TEST", df)
    fixed = df.iloc[-1:].copy()
    fixed.loc[fixed.index[0], "Close"] = 999.0
    store.put("TEST", fixed)
    assert store.get("TEST")["Close"].iloc[-1] == pytest.approx(999.0, rel=1e-4)


def test_the_result_stays_sorted(store):
    df = synthetic_ohlc(200, seed=7)
    store.put("TEST", df.iloc[100:])
    store.put("TEST", df.iloc[:100])          # out of order on purpose
    assert store.get("TEST").index.is_monotonic_increasing


def test_empty_and_malformed_frames_are_ignored(store):
    store.put("TEST", pd.DataFrame())
    store.put("TEST", None)
    store.put("TEST", pd.DataFrame({"Close": [1.0]}))    # missing OHL
    assert store.get("TEST") is None


# ── persistence ─────────────────────────────────────────────────────────────
def test_data_survives_a_restart(tmp_path):
    a = PriceStore(config_dir=str(tmp_path))
    df = synthetic_ohlc(400, seed=8)
    a.put("TEST", df)
    a.flush()

    b = PriceStore(config_dir=str(tmp_path))          # a fresh process
    back = b.get("TEST")
    assert back is not None and len(back) == 400
    assert b.last_date("TEST") == df.index[-1]


def test_nothing_is_written_before_a_flush(tmp_path):
    a = PriceStore(config_dir=str(tmp_path))
    a.put("TEST", synthetic_ohlc(100, seed=9))
    assert PriceStore(config_dir=str(tmp_path)).get("TEST") is None
    a.flush()
    assert PriceStore(config_dir=str(tmp_path)).get("TEST") is not None


def test_a_corrupt_shard_is_discarded_not_guessed_at(tmp_path):
    a = PriceStore(config_dir=str(tmp_path))
    a.put("TEST", synthetic_ohlc(100, seed=10))
    a.flush()
    with open(a._path(_shard_for("TEST")), "wb") as f:
        f.write(b"this is not a gzip stream")

    b = PriceStore(config_dir=str(tmp_path))
    assert b.get("TEST") is None, "a corrupt shard must not yield prices"


def test_symbols_are_sharded_by_first_character():
    assert _shard_for("AAPL") == "A" and _shard_for("aapl") == "A"
    assert _shard_for("9XYZ") == "9"
    assert _shard_for("") == "_"


def test_stats_report_what_is_held(tmp_path):
    s = PriceStore(config_dir=str(tmp_path))
    s.put("AAA", synthetic_ohlc(300, seed=11))
    s.put("BBB", synthetic_ohlc(300, seed=12))
    s.flush()
    st = PriceStore(config_dir=str(tmp_path)).stats()
    assert st["symbols"] == 2 and st["bars"] == 600 and st["diskBytes"] > 0


def test_clear_empties_the_store(tmp_path):
    s = PriceStore(config_dir=str(tmp_path))
    s.put("AAA", synthetic_ohlc(100, seed=13))
    s.flush()
    s.clear()
    assert PriceStore(config_dir=str(tmp_path)).get("AAA") is None


# ── the payoff: incremental refresh ─────────────────────────────────────────
def _wired(tmp_path, monkeypatch, symbols, history):
    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER,
                              api_key="k", api_secret="s"), demo=True)
    state.demo = False
    state.prices = PriceStore(config_dir=str(tmp_path))
    calls = {"full": 0, "incr": 0, "bars": 0}

    def fake(syms, years, k, sec, chunk=200, since=None):
        out = {}
        for s in syms:
            df = history[s]
            sub = df[df.index >= pd.Timestamp(since)] if since is not None else df
            calls["incr" if since is not None else "full"] += 1
            calls["bars"] += len(sub)
            out[s] = sub
        return out

    import markov_hedge_fund_method.market_data as md
    monkeypatch.setattr(md, "batch_alpaca_ohlc", fake)
    return state, calls


def test_the_first_pass_downloads_full_history(tmp_path, monkeypatch):
    hist = {"AAA": synthetic_ohlc(2520, seed=20)}
    state, calls = _wired(tmp_path, monkeypatch, ["AAA"], hist)
    stats = state.prefetch_ohlc(["AAA"])
    assert stats["fetched"] == 1 and calls["full"] == 1
    assert calls["bars"] == 2520


def test_a_second_pass_downloads_nothing_when_already_current(tmp_path, monkeypatch):
    hist = {"AAA": synthetic_ohlc(2520, seed=21)}
    state, calls = _wired(tmp_path, monkeypatch, ["AAA"], hist)
    state.prefetch_ohlc(["AAA"])
    before = calls["bars"]
    state._ohlc_cache.clear()                     # as if the app restarted
    stats = state.prefetch_ohlc(["AAA"])
    assert calls["bars"] == before, "it re-downloaded data it already had"
    assert stats["restored"] == 1 and stats["fetched"] == 0


def test_a_stale_symbol_is_topped_up_not_refetched(tmp_path, monkeypatch):
    """The steady state: come back after a few days, download a few days."""
    full = synthetic_ohlc(2520, seed=22)
    state, calls = _wired(tmp_path, monkeypatch, ["AAA"], {"AAA": full})
    state.prices.put("AAA", full.iloc[:-7])       # as if last run a week ago
    state.prices.flush()

    stats = state.prefetch_ohlc(["AAA"])
    assert stats["toppedUp"] == 1 and stats["fetched"] == 0
    assert calls["full"] == 0, "a top-up must never pull the whole history"
    assert calls["bars"] <= 20, f"transferred {calls['bars']} bars for a week of data"


def test_a_top_up_leaves_the_history_complete(tmp_path, monkeypatch):
    full = synthetic_ohlc(2520, seed=23)
    state, calls = _wired(tmp_path, monkeypatch, ["AAA"], {"AAA": full})
    state.prices.put("AAA", full.iloc[:-7])
    state.prices.flush()
    state.prefetch_ohlc(["AAA"])

    held = state._ohlc_cache["AAA"][1]
    assert len(held) == 2520, "topping up must extend the history, not replace it"
    assert held.index[-1] == full.index[-1]


def test_a_mixed_universe_takes_the_cheapest_path_per_symbol(tmp_path, monkeypatch):
    hist = {s: synthetic_ohlc(2520, seed=i) for i, s in enumerate(("AAA", "BBB", "CCC"))}
    state, calls = _wired(tmp_path, monkeypatch, list(hist), hist)
    state.prices.put("AAA", hist["AAA"])              # current
    state.prices.put("BBB", hist["BBB"].iloc[:-5])    # stale
    state.prices.flush()                              # CCC unknown

    stats = state.prefetch_ohlc(["AAA", "BBB", "CCC"])
    assert stats["restored"] == 1 and stats["toppedUp"] == 1 and stats["fetched"] == 1


def test_a_failed_top_up_still_serves_the_stored_history(tmp_path, monkeypatch):
    """Slow is acceptable; blank is not."""
    full = synthetic_ohlc(2520, seed=24)
    state, _ = _wired(tmp_path, monkeypatch, ["AAA"], {"AAA": full})
    state.prices.put("AAA", full.iloc[:-7])
    state.prices.flush()

    import markov_hedge_fund_method.market_data as md
    monkeypatch.setattr(md, "batch_alpaca_ohlc",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    state.prefetch_ohlc(["AAA"])
    held = state._ohlc_cache.get("AAA")
    assert held is not None and len(held[1]) == 2513


def test_demo_mode_never_touches_the_store(tmp_path, monkeypatch):
    hist = {"AAA": synthetic_ohlc(500, seed=25)}
    state, calls = _wired(tmp_path, monkeypatch, ["AAA"], hist)
    state.demo = True
    state.prefetch_ohlc(["AAA"])
    assert calls["full"] == 0 and state.prices.get("AAA") is None
