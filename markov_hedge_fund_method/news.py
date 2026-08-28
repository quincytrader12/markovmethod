"""Headlines + a lightweight bullish/bearish tag for a symbol.

Live headlines come from Alpaca's news API when credentials are present;
otherwise a deterministic **sample feed** keeps the HUD's news slot populated
offline. Sentiment is a transparent keyword heuristic (not a model), so the
bullish / bearish / neutral label is explainable and never overclaims.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

_BULLISH = {
    "beats", "beat", "surge", "surges", "soar", "soars", "rally", "rallies",
    "upgrade", "upgraded", "record", "profit", "growth", "outperform", "jumps",
    "jump", "gains", "gain", "tops", "raises", "raise", "buy", "bullish",
    "strong", "expands", "expand", "wins", "win", "breakthrough", "approval",
    "higher", "up", "rises", "rise", "optimistic", "boost", "boosts", "rebound",
}
_BEARISH = {
    "miss", "misses", "plunge", "plunges", "fall", "falls", "drop", "drops",
    "downgrade", "downgraded", "loss", "losses", "cut", "cuts", "lawsuit",
    "probe", "decline", "declines", "warns", "warn", "weak", "slump", "slumps",
    "sinks", "sink", "bearish", "sell", "lower", "recall", "recalls", "halts",
    "halt", "delay", "delays", "investigation", "slashes", "slash", "fears",
    "concern", "concerns", "tumble", "tumbles", "selloff",
}


def classify(text: str) -> str:
    """bullish / bearish / neutral from a transparent keyword tally."""
    words = {w.strip(".,!?:;()[]'\"").lower() for w in (text or "").split()}
    bull = len(words & _BULLISH)
    bear = len(words & _BEARISH)
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    return "neutral"


_TEMPLATES = [
    ("{s} beats quarterly earnings and raises full-year guidance", "bullish"),
    ("Analysts upgrade {s} to Buy on strong demand outlook", "bullish"),
    ("{s} shares rally as revenue tops estimates", "bullish"),
    ("{s} unveils new product line, expands market share", "bullish"),
    ("{s} gains after record quarterly profit", "bullish"),
    ("{s} slips after cautious guidance for next quarter", "bearish"),
    ("Regulators open probe into {s} business practices", "bearish"),
    ("{s} downgraded on margin pressure and rising costs", "bearish"),
    ("{s} recalls product; shares fall in early trading", "bearish"),
    ("{s} tumbles as analysts warn on weak demand", "bearish"),
    ("{s} holds steady as investors await earnings", "neutral"),
    ("{s} in focus ahead of sector rotation", "neutral"),
]


def _synthetic(symbol: str, limit: int) -> list[dict]:
    seed = int(hashlib.sha256(symbol.encode()).hexdigest(), 16)
    now = datetime.now(timezone.utc)
    out, rnd = [], seed
    for i in range(limit):
        rnd = (rnd * 1103515245 + 12345) & 0x7FFFFFFF
        text, sent = _TEMPLATES[rnd % len(_TEMPLATES)]
        out.append({
            "headline": text.format(s=symbol),
            "source": "Sample feed",
            "url": "",
            "createdAt": (now - timedelta(hours=2 * i + 1)).isoformat(),
            "sentiment": sent,
            "sample": True,
        })
    return out


def _from_alpaca(settings, symbol: str, limit: int) -> list[dict]:
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    client = NewsClient(settings.api_key, settings.api_secret)
    res = client.get_news(NewsRequest(symbols=[symbol], limit=limit))
    items = getattr(res, "news", None)
    if items is None and hasattr(res, "data"):
        items = res.data.get("news", [])
    out = []
    for n in (items or [])[:limit]:
        head = getattr(n, "headline", "") or ""
        summary = getattr(n, "summary", "") or ""
        created = getattr(n, "created_at", None)
        out.append({
            "headline": head,
            "source": getattr(n, "source", "") or "Alpaca",
            "url": getattr(n, "url", "") or "",
            "createdAt": created.isoformat() if created else "",
            "sentiment": classify(f"{head} {summary}"),
            "sample": False,
        })
    return out


def _from_yahoo_rss(symbol: str, limit: int) -> list[dict]:
    """Free real headlines from Yahoo Finance's RSS feed — a plain XML GET with
    no API key and no crumb/cookie dance, so it's far more reliable than the
    yfinance news endpoint."""
    import urllib.request
    import xml.etree.ElementTree as ET

    url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline"
           f"?s={symbol}&region=US&lang=en-US")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    out = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        desc = (item.findtext("description") or "").strip()
        out.append({
            "headline": title,
            "source": "Yahoo Finance",
            "url": (item.findtext("link") or "").strip(),
            "createdAt": (item.findtext("pubDate") or "").strip(),
            "sentiment": classify(f"{title} {desc}"),
            "sample": False,
        })
        if len(out) >= limit:
            break
    return out


def _from_yfinance(symbol: str, limit: int) -> list[dict]:
    """Free real headlines from Yahoo Finance (no API key). Format is defensive
    because yfinance has changed its news schema across versions."""
    import yfinance as yf
    from datetime import datetime, timezone

    raw = yf.Ticker(symbol).news or []
    out = []
    for it in raw:
        content = it.get("content") if isinstance(it.get("content"), dict) else it
        title = content.get("title") or it.get("title") or ""
        if not title:
            continue
        prov = content.get("provider")
        source = (prov.get("displayName") if isinstance(prov, dict) else None) \
            or it.get("publisher") or "Yahoo Finance"
        url = ""
        for key in ("canonicalUrl", "clickThroughUrl"):
            u = content.get(key)
            if isinstance(u, dict) and u.get("url"):
                url = u["url"]
                break
        url = url or it.get("link") or ""
        created = ""
        pub = content.get("pubDate") or content.get("displayTime")
        if pub:
            created = str(pub)
        elif it.get("providerPublishTime"):
            created = datetime.fromtimestamp(it["providerPublishTime"], timezone.utc).isoformat()
        summary = content.get("summary") or content.get("description") or ""
        out.append({
            "headline": title,
            "source": source,
            "url": url,
            "createdAt": created,
            "sentiment": classify(f"{title} {summary}"),
            "sample": False,
        })
        if len(out) >= limit:
            break
    return out


def fetch_news(settings, symbol: str, demo: bool = False, limit: int = 6) -> list[dict]:
    """Recent headlines for `symbol`, tagged bullish/bearish/neutral.

    Source order: Alpaca (if connected) → Yahoo RSS (free, no key) → yfinance →
    a deterministic sample feed. News is independent of the `demo` price-data
    mode, so real headlines show whenever there's internet — even in demo. When
    offline, every network source fails fast and the sample feed fills in.
    """
    symbol = (symbol or "").upper()
    if getattr(settings, "has_credentials", False):
        try:
            items = _from_alpaca(settings, symbol, limit)
            if items:
                return items
        except Exception:  # noqa: BLE001
            pass
    for source in (_from_yahoo_rss, _from_yfinance):
        try:
            items = source(symbol, limit)
            if items:
                return items
        except Exception:  # noqa: BLE001
            pass
    return _synthetic(symbol, limit)
