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

    Source order: Alpaca (if connected) → Yahoo Finance (free, no key) →
    a deterministic sample feed (offline). Demo mode stays fully offline.
    """
    symbol = (symbol or "").upper()
    if not demo:
        if getattr(settings, "has_credentials", False):
            try:
                items = _from_alpaca(settings, symbol, limit)
                if items:
                    return items
            except Exception:  # noqa: BLE001
                pass
        try:
            items = _from_yfinance(symbol, limit)
            if items:
                return items
        except Exception:  # noqa: BLE001
            pass
    return _synthetic(symbol, limit)
