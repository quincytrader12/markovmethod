"""Performance work + new timeframes + lot-based sizing.

Covers the incremental walk-forward (same numbers, far less work), the batched
parallel quotes endpoint, the 1H/4H intraday timeframes, and sizing orders in
lots instead of shares.
"""

from __future__ import annotations

import time

import numpy as np
from fastapi.testclient import TestClient

from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.market_data import synthetic_intraday
from markov_hedge_fund_method.orders import (
    DEFAULT_LOT_SIZE,
    OrderTicket,
    OrderValidationError,
    build_order_request,
    describe,
)
from markov_hedge_fund_method.regime import (
    build_transition_matrix,
    label_regimes,
    signal_from_matrix,
    walk_forward_backtest,
)
from markov_hedge_fund_method.market_data import synthetic_close
from markov_hedge_fund_method.web import AppState, create_app


def _demo_client():
    state = AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)
    return TestClient(create_app(state)), state


# ── the optimisation must not change the maths ───────────────────────────────
def test_incremental_backtest_matches_naive_implementation():
    close = synthetic_close(seed=11)
    labels = label_regimes(close)
    daily = close.pct_change().dropna()
    ci = labels.index.intersection(daily.index)
    lb, dr = labels.loc[ci], daily.loc[ci]

    naive = []
    for t in range(252, len(lb) - 1):                      # the old O(n^2) loop
        P = build_transition_matrix(lb.iloc[:t])
        naive.append(float(np.sign(signal_from_matrix(P, int(lb.iloc[t])))) * float(dr.iloc[t + 1]))

    fast = walk_forward_backtest(close, labels)
    assert np.allclose((1 + np.array(naive)).cumprod(), np.array(fast["equity"]), atol=1e-12)


def test_backtest_is_fast():
    close = synthetic_close(seed=4)
    labels = label_regimes(close)
    t0 = time.perf_counter()
    walk_forward_backtest(close, labels)
    assert time.perf_counter() - t0 < 0.5   # was ~0.4s+ per symbol before; now ~0.015s


# ── batched quotes ───────────────────────────────────────────────────────────
def test_batch_quotes_endpoint():
    client, _ = _demo_client()
    d = client.get("/api/quotes", params={"symbols": "spy,aapl,msft"}).json()
    quotes = d["quotes"]
    assert {q["ticker"] for q in quotes} == {"SPY", "AAPL", "MSFT"}
    for q in quotes:
        assert q["regime"] in ("bull", "bear", "sideways")
        assert isinstance(q["lastPrice"], (int, float))
    assert any(q["name"] for q in quotes)


def test_batch_quotes_empty_and_bad_symbols():
    client, _ = _demo_client()
    assert client.get("/api/quotes", params={"symbols": ""}).json()["quotes"] == []
    # unknown tickers still resolve (synthetic fallback) and never 500
    assert client.get("/api/quotes", params={"symbols": "ZZZZ"}).status_code == 200


# ── 1H / 4H timeframes ───────────────────────────────────────────────────────
def test_synthetic_intraday_new_timeframes():
    for tf, delta in (("1H", 3600), ("4H", 4 * 3600)):
        df = synthetic_intraday(tf, seed=2)
        assert len(df) >= 50
        step = (df.index[-1] - df.index[-2]).total_seconds()
        assert step == delta


def test_candles_endpoint_1h_4h():
    client, _ = _demo_client()
    for tf in ("1H", "4H"):
        d = client.get("/api/candles", params={"symbol": "AAPL", "tf": tf}).json()
        assert d["tf"] == tf and len(d["bars"]) >= 50
        bar = d["bars"][-1]
        assert bar["l"] <= bar["o"] <= bar["h"] and ":" in bar["t"]


# ── lots ─────────────────────────────────────────────────────────────────────
def test_lots_convert_to_shares():
    t = OrderTicket(symbol="AAPL", order_type="market", lots=3)
    assert t.normalized().qty == 3 * DEFAULT_LOT_SIZE == 300
    assert getattr(build_order_request(t), "qty") == 300.0
    assert "3 lots (300 sh)" in describe(t)


def test_custom_lot_size_and_singular_label():
    t = OrderTicket(symbol="X", order_type="market", lots=2, lot_size=50)
    assert t.normalized().qty == 100
    assert "1 lot (" in describe(OrderTicket(symbol="X", order_type="market", lots=1, lot_size=10))


def test_bad_lots_rejected():
    for bad in (OrderTicket(symbol="X", order_type="market", lots=0),
                OrderTicket(symbol="X", order_type="market", lots=-1)):
        try:
            bad.normalized()
            raise AssertionError("expected rejection")
        except OrderValidationError:
            pass
    try:
        OrderTicket(symbol="X", order_type="market", lots=1, lot_size=0).normalized()
        raise AssertionError("expected rejection")
    except OrderValidationError:
        pass


def test_shares_path_still_works():
    assert OrderTicket(symbol="X", order_type="market", qty=7).normalized().qty == 7
    assert "7 sh" in describe(OrderTicket(symbol="X", order_type="market", qty=7))


def test_order_endpoint_accepts_lots():
    from markov_hedge_fund_method.broker import OrderResult

    class B:
        def __init__(self):
            self.seen = []

        def submit_ticket(self, ticket):
            build_order_request(ticket)
            self.seen.append(ticket)
            return OrderResult(id="1", status="accepted", summary=describe(ticket))

    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER), demo=False)
    state.broker = B()
    client = TestClient(create_app(state))
    r = client.post("/api/orders", json={"symbol": "AAPL", "order_type": "market",
                                         "side": "buy", "lots": 2, "lot_size": 100})
    assert r.status_code == 200
    assert state.broker.seen[0].normalized().qty == 200
    assert "2 lots" in r.json()["summary"]


def test_index_has_lots_and_new_timeframes_and_watch_button():
    client, _ = _demo_client()
    html = client.get("/").text
    assert 'id="o-lots"' in html and 'id="o-lotsize"' in html
    assert 'value="1H"' in html and 'value="4H"' in html
    assert "watchFromScan" in html and "/api/quotes" in html
