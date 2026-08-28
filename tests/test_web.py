"""Web HUD backend (FastAPI) — endpoints exercised with TestClient.

Demo mode drives the data endpoints offline (synthetic prices); a fake broker
+ injected account store cover portfolio/orders/accounts with no network.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from markov_hedge_fund_method.accounts import AccountStore
from markov_hedge_fund_method.broker import Account, OrderResult, Position
from markov_hedge_fund_method.config import Mode, Settings
from markov_hedge_fund_method.orders import build_order_request
from markov_hedge_fund_method.web import AppState, create_app


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def set_password(self, s, u, p):
        self.store[(s, u)] = p

    def get_password(self, s, u):
        return self.store.get((s, u))

    def delete_password(self, s, u):
        self.store.pop((s, u), None)


class FakeBroker:
    def __init__(self):
        self.submitted = []
        self.cancelled = 0

    def get_account(self):
        return Account(cash=1000.0, equity=2500.0, buying_power=5000.0, status="ACTIVE")

    def get_position(self, symbol):
        return Position(symbol=symbol, qty=3, market_value=900.0, unrealized_pl=42.0, side="long")

    def list_open_orders(self):
        return []

    def submit_ticket(self, ticket):
        build_order_request(ticket)  # prove validity
        self.submitted.append(ticket)
        return OrderResult(id="web-1", status="accepted", summary="ok")

    def cancel_all_orders(self):
        self.cancelled += 1
        return 2


def _demo_client():
    state = AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)
    return TestClient(create_app(state)), state


def _paper_client(tmp_path):
    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER), demo=False)
    state.accounts = AccountStore(keyring=FakeKeyring(), config_dir=str(tmp_path))
    state.broker = FakeBroker()
    return TestClient(create_app(state)), state


def test_config_and_state_demo():
    client, _ = _demo_client()
    cfg = client.get("/api/config").json()
    assert cfg["demo"] is True and cfg["mode"] == "dashboard"

    st = client.get("/api/state", params={"symbol": "spy"}).json()
    assert st["ticker"] == "SPY"
    assert st["regime"] in ("bull", "bear", "sideways")
    assert 0 <= st["greedFear"]["score"] <= 100
    assert len(st["chart"]["bars"]) > 50
    assert all(abs(sum(row) - 1.0) < 1e-6 for row in st["matrix"])
    assert "dataSource" in st


def test_state_has_candles_and_name():
    client, _ = _demo_client()
    st = client.get("/api/state", params={"symbol": "AAPL"}).json()
    assert st["name"] == "Apple Inc."
    bar = st["chart"]["bars"][-1]
    assert {"o", "h", "l", "c", "up", "regime", "t"} <= set(bar)
    assert bar["l"] <= bar["o"] <= bar["h"] and bar["l"] <= bar["c"] <= bar["h"]
    assert isinstance(bar["up"], bool)


def test_intraday_candles_1d_and_1w():
    client, _ = _demo_client()
    for tf, floor in (("1D", 40), ("1W", 80)):
        d = client.get("/api/candles", params={"symbol": "AAPL", "tf": tf}).json()
        assert d["tf"] == tf and len(d["bars"]) >= floor
        bar = d["bars"][-1]
        assert {"o", "h", "l", "c", "up", "t"} <= set(bar)
        assert bar["l"] <= bar["o"] <= bar["h"] and bar["l"] <= bar["c"] <= bar["h"]
        assert ":" in bar["t"]                       # intraday timestamps carry a time
        assert len(d["ma20"]) == len(d["bars"])


def test_quote_includes_name():
    client, _ = _demo_client()
    q = client.get("/api/quote", params={"symbol": "MSFT"}).json()
    assert q["name"] == "Microsoft Corporation"


def test_index_served():
    client, _ = _demo_client()
    r = client.get("/")
    assert r.status_code == 200
    assert "MAMBA" in r.text and "canvas" in r.text


def test_orders_require_connection():
    client, _ = _demo_client()               # demo → no broker
    r = client.post("/api/orders", json={"symbol": "SPY", "qty": 1})
    assert r.status_code == 403


def test_order_submit_and_validation(tmp_path):
    client, state = _paper_client(tmp_path)
    ok = client.post("/api/orders", json={"symbol": "SPY", "order_type": "market", "qty": 5})
    assert ok.status_code == 200 and ok.json()["id"] == "web-1"
    assert len(state.broker.submitted) == 1

    bad = client.post("/api/orders", json={"symbol": "SPY", "order_type": "limit", "qty": 5})
    assert bad.status_code == 400                       # limit without a price
    assert len(state.broker.submitted) == 1             # never reached the broker

    cancel = client.post("/api/orders/cancel_all")
    assert cancel.status_code == 200 and cancel.json()["cancelled"] == 2


def test_portfolio(tmp_path):
    client, _ = _paper_client(tmp_path)
    pf = client.get("/api/portfolio", params={"symbol": "SPY"}).json()
    assert pf["connected"] is True
    assert pf["account"]["equity"] == 2500.0
    assert pf["position"]["symbol"] == "SPY"


def test_accounts_crud(tmp_path):
    client, state = _paper_client(tmp_path)
    assert client.get("/api/accounts").json()["accounts"] == []

    r = client.post("/api/accounts", json={"name": "swing", "key_id": "K1", "secret": "S1", "paper": True})
    body = r.json()
    assert [a["name"] for a in body["accounts"]] == ["swing"]
    assert body["active"] == "swing"

    client.post("/api/accounts", json={"name": "live1", "key_id": "K2", "secret": "S2", "paper": False})
    active = client.post("/api/accounts/active", json={"name": "live1"}).json()
    assert active["active"] == "live1"
    assert state.settings.account == "live1" and state.settings.account_paper is False

    gone = client.delete("/api/accounts/swing").json()
    assert [a["name"] for a in gone["accounts"]] == ["live1"]


def test_add_account_validation(tmp_path):
    client, _ = _paper_client(tmp_path)
    r = client.post("/api/accounts", json={"name": "bad/name", "key_id": "K", "secret": "S"})
    assert r.status_code == 400


def test_quote_is_cheap_and_shaped():
    client, _ = _demo_client()
    q = client.get("/api/quote", params={"symbol": "nvda"}).json()
    assert q["ticker"] == "NVDA"
    assert q["regime"] in ("bull", "bear", "sideways")
    assert isinstance(q["lastPrice"], (int, float))
    assert "chart" not in q  # the cheap endpoint omits the heavy series


def test_search_prefix_and_custom_ticker():
    client, _ = _demo_client()
    res = client.get("/api/search", params={"q": "aa"}).json()["results"]
    syms = [r["symbol"] for r in res]
    assert "AAPL" in syms and all("name" in r for r in res)
    assert any(r["symbol"] == "AAPL" and "Apple" in r["name"] for r in res)  # names attached
    # a ticker not in the bundled universe is still offered verbatim
    custom = client.get("/api/search", params={"q": "ZZZZ"}).json()["results"]
    assert custom[0]["symbol"] == "ZZZZ"


def test_state_chart_is_wide_enough_for_timeframes():
    client, _ = _demo_client()
    bars = client.get("/api/state", params={"symbol": "SPY"}).json()["chart"]["bars"]
    assert len(bars) >= 504  # enough history for the 2Y timeframe, sliced client-side


def test_news_demo_sample_feed():
    client, _ = _demo_client()
    body = client.get("/api/news", params={"symbol": "TSLA"}).json()
    items = body["items"]
    assert body["symbol"] == "TSLA" and len(items) >= 1
    assert all(i["sentiment"] in ("bullish", "bearish", "neutral") for i in items)
    assert all(i["sample"] is True for i in items)  # offline → sample feed


def test_news_sentiment_classifier():
    from markov_hedge_fund_method.news import classify
    assert classify("Company beats earnings and shares surge to record") == "bullish"
    assert classify("Stock plunges after downgrade and lawsuit probe") == "bearish"
    assert classify("Company holds annual meeting today") == "neutral"


def test_state_has_forecast_and_timeline_and_equity():
    client, _ = _demo_client()
    st = client.get("/api/state", params={"symbol": "SPY"}).json()

    # n-step regime forecast — probabilities that sum to ~1 per horizon
    fc = st["forecast"]
    assert [row["h"] for row in fc] == [1, 5, 20]
    for row in fc:
        assert abs(row["bear"] + row["sideways"] + row["bull"] - 1.0) < 1e-3  # 4-dp rounding

    # regime timeline
    tl = st["regimeTimeline"]
    assert tl["daysInRegime"] >= 1
    assert isinstance(tl["recentFlips"], list)
    if tl["recentFlips"]:
        f = tl["recentFlips"][0]
        assert set(f) == {"date", "from", "to"}

    # walk-forward equity + win rate
    m = st["metrics"]
    assert isinstance(m["equity"], list) and len(m["equity"]) >= 2
    assert 0.0 <= m["winRate"] <= 1.0


def test_search_returns_connected_flag_and_letter_matches():
    client, _ = _demo_client()
    d = client.get("/api/search", params={"q": "A"}).json()
    assert d["connected"] is False                       # demo → not Alpaca-connected
    syms = [r["symbol"] for r in d["results"]]
    assert any(s.startswith("A") for s in syms)          # letter search lists A-tickers


def test_mode_switch_endpoint():
    state = AppState(Settings(ticker="SPY", mode=Mode.DASHBOARD), demo=True)
    client = TestClient(create_app(state))
    assert client.get("/api/config").json()["mode"] == "dashboard"
    r = client.post("/api/mode", json={"mode": "paper"})
    assert r.status_code == 200
    assert r.json()["mode"] == "paper" and r.json()["canTrade"] is True
    assert state.settings.mode is Mode.PAPER            # backend state actually changed
    assert client.post("/api/mode", json={"mode": "turbo"}).status_code == 400


def test_validate_offline_is_unverified():
    client, _ = _demo_client()
    v = client.get("/api/validate", params={"symbol": "ZZZZ"}).json()
    # offline we can't check Alpaca, so we don't block (source=unverified)
    assert v["valid"] is True and v["source"] == "unverified"


def test_news_sentiment_from_headline_text():
    # bullish/bearish wording drives the tag regardless of source
    from markov_hedge_fund_method.news import classify
    assert classify("shares soar to record high after upgrade") == "bullish"
    assert classify("shares tumble on downgrade and probe") == "bearish"


def test_forecast_converges_to_stationary():
    # far-horizon forecast should approach the stationary distribution
    from markov_hedge_fund_method.market_data import synthetic_close
    from markov_hedge_fund_method.webstate import market_state
    st = market_state(synthetic_close(seed=5), "SPY")
    far = st["forecast"][-1]  # 20 steps
    stat = st["stationary"]   # [bear, sideways, bull]
    assert abs(far["bear"] - stat[0]) < 0.05
    assert abs(far["bull"] - stat[2]) < 0.05


def test_search_never_blocks_while_universe_loads():
    """Typing must answer instantly even while Alpaca's asset list is fetching."""
    import time as _t

    class SlowBroker:
        def list_tradable_assets(self):
            _t.sleep(1.5)
            return [{"symbol": "AAPL", "name": "Apple Inc."},
                    {"symbol": "AACG", "name": "ATA Creativity"}]

    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER), demo=False)
    state.broker = SlowBroker()
    client = TestClient(create_app(state))

    t0 = _t.time()
    d = client.get("/api/search", params={"q": "AA"}).json()
    assert _t.time() - t0 < 0.5, "search must not wait on the asset download"
    assert d["loading"] is True          # told the client the universe is warming
    assert d["results"]                  # still answered, from the bundled list


def test_search_uses_alpaca_index_once_loaded():
    class Broker:
        def list_tradable_assets(self):
            return [{"symbol": s, "name": s + " Corp"} for s in ("AAPL", "AACG", "AMD", "ZZZ")]

    state = AppState(Settings(ticker="SPY", mode=Mode.PAPER), demo=False)
    state.broker = Broker()
    state._ensure_alpaca_universe()       # simulate the background load finishing
    client = TestClient(create_app(state))

    d = client.get("/api/search", params={"q": "AA"}).json()
    syms = [r["symbol"] for r in d["results"]]
    assert d["connected"] is True and d["loading"] is False
    assert syms == ["AACG", "AAPL"]       # prefix hits, sorted
    assert all(r["name"] for r in d["results"])
    # the prefix index is built once, not rebuilt per keystroke
    assert state._alpaca_index["A"] == ["AACG", "AAPL", "AMD"]
