"""Closing positions honestly, and intraday fetch falling back rather than dying."""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import markov_hedge_fund_method.market_data as md
from markov_hedge_fund_method.broker import AlpacaBroker, OpenOrder, Position
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.web import AppState, create_app


# ── closing a position ───────────────────────────────────────────────────────
class _Order:
    def __init__(self, oid="o9", status="accepted", qty="10"):
        self.id, self.status, self.qty = oid, status, qty


class _Client:
    """Mimics Alpaca: shares held by a working order cannot be liquidated."""
    def __init__(self, open_orders=(), fills=True):
        self.open_orders = list(open_orders)
        self.fills = fills
        self.cancelled = []
        self.closed = []
        self.position_gone = False

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)
        self.open_orders = [o for o in self.open_orders if o.id != oid]

    def close_position(self, symbol):
        if self.open_orders:
            raise RuntimeError("insufficient qty available for order")
        self.closed.append(symbol)
        if self.fills:
            self.position_gone = True
        return _Order()


def _broker(client):
    b = AlpacaBroker(Settings(ticker="SPY", mode=Mode.PAPER))
    b._client = client
    return b


def test_close_cancels_working_orders_first(monkeypatch):
    """The classic first-attempt failure: shares reserved by an open order."""
    c = _Client(open_orders=[_Order("o1")])
    b = _broker(c)
    monkeypatch.setattr(b, "list_open_orders",
                        lambda: [OpenOrder("o1", "SPY", "sell", "limit", "10", "new")])
    res = b.close_position("SPY")
    assert c.cancelled == ["o1"], "must clear the working order before liquidating"
    assert c.closed == ["SPY"]
    assert res["cancelledOrders"] == 1 and res["orderId"] == "o9"


def test_close_returns_order_details_not_a_guess(monkeypatch):
    c = _Client()
    b = _broker(c)
    monkeypatch.setattr(b, "list_open_orders", lambda: [])
    res = b.close_position("AAPL")
    assert res["status"] == "accepted" and res["orderId"] == "o9"


class _VerifyBroker:
    """Close submits an order; whether it fills is controlled by `fills`."""
    def __init__(self, fills):
        self.fills, self.flat = fills, False

    def get_position(self, symbol):
        if self.flat:
            return None
        return Position(symbol, 10, 1900.0, 50.0, "long", avg_entry=185.0,
                        current_price=190.0, unrealized_plpc=0.02)

    def position_qty(self, symbol):
        return 0.0 if self.flat else 10.0

    def close_position(self, symbol, **kw):
        if self.fills:
            self.flat = True
        return {"symbol": symbol, "orderId": "x1", "status": "accepted",
                "qty": "10", "cancelledOrders": 0}

    def list_positions(self):
        return [] if self.flat else [self.get_position("SPY")]

    def cancel_all_orders(self):
        return 0


def _client_with(fills):
    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER), demo=False)
    state.broker = _VerifyBroker(fills)
    return TestClient(create_app(state))


def test_filled_close_reports_flat():
    r = _client_with(True).post("/api/positions/close", json={"symbol": "SPY"}).json()
    assert r["flat"] is True and r["qtyRemaining"] == 0.0
    assert "flat" in r["message"].lower()


def test_unfilled_close_says_pending_not_closed():
    """Market shut: the order queues. We must not claim the position is closed."""
    r = _client_with(False).post("/api/positions/close", json={"symbol": "SPY"}).json()
    assert r["flat"] is False and r["qtyRemaining"] == 10.0
    msg = r["message"].lower()
    assert "submitted" in msg and "still holding" in msg
    assert "closed — position is flat" not in msg


def test_broker_rejection_surfaces_the_real_reason():
    class Rejecting:
        def get_position(self, s): return None
        def close_position(self, s, **k): raise RuntimeError("insufficient qty available for order")
    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER), demo=False)
    state.broker = Rejecting()
    r = TestClient(create_app(state)).post("/api/positions/close", json={"symbol": "SPY"})
    assert r.status_code == 502
    assert "insufficient qty" in r.json()["detail"]


# ── intraday fetching ────────────────────────────────────────────────────────
def _bars():
    idx = pd.date_range("2026-08-14 13:30", periods=40, freq="5min")
    return pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5}, index=idx)


def test_intraday_falls_back_to_yahoo_when_alpaca_fails(monkeypatch):
    monkeypatch.setattr(md, "_alpaca_intraday",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("subscription does not permit")))
    monkeypatch.setattr(md, "_yfinance_intraday", lambda *a, **k: _bars())
    s = Settings(ticker="SPY", mode=Mode.PAPER, api_key="k", api_secret="s")
    assert not md.get_intraday_ohlc(s, "1D").empty


def test_intraday_falls_back_when_alpaca_returns_nothing(monkeypatch):
    monkeypatch.setattr(md, "_alpaca_intraday", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(md, "_yfinance_intraday", lambda *a, **k: _bars())
    s = Settings(ticker="SPY", mode=Mode.PAPER, api_key="k", api_secret="s")
    assert len(md.get_intraday_ohlc(s, "1H")) == 40


def test_intraday_error_names_both_sources(monkeypatch):
    monkeypatch.setattr(md, "_alpaca_intraday",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("subscription does not permit")))
    monkeypatch.setattr(md, "_yfinance_intraday",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no data")))
    s = Settings(ticker="SPY", mode=Mode.PAPER, api_key="k", api_secret="s")
    with pytest.raises(RuntimeError) as e:
        md.get_intraday_ohlc(s, "1D")
    assert "Alpaca" in str(e.value) and "Yahoo" in str(e.value)


def test_candles_endpoint_reports_why_it_fell_back(monkeypatch):
    monkeypatch.setattr("markov_hedge_fund_method.web.get_intraday_ohlc",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("subscription does not permit")))
    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER, api_key="k", api_secret="s"), demo=False)
    d = TestClient(create_app(state)).get("/api/candles?symbol=SPY&tf=1D").json()
    assert d["source"].startswith("synthetic")
    assert "subscription does not permit" in d["source"], "the reason must reach the UI"
    assert d["bars"], "the chart still renders rather than blanking"
