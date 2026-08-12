"""Telegram delivery for scanner reports and regime-flip alerts.

Uses the Bot API over plain stdlib HTTP — no extra dependency, so it works
inside the frozen Windows executable. Configuration (bot token + chat id) lives
alongside the account registry; the token is a secret, so it is stored with the
same care as API keys.

Set-up, for the user:
  1. message @BotFather on Telegram, /newbot, copy the token
  2. message your new bot once (so it may write to you)
  3. paste the token in the terminal — the chat id is discovered automatically
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from .accounts import default_config_dir

API = "https://api.telegram.org/bot{token}/{method}"
TIMEOUT = 10


class TelegramError(RuntimeError):
    pass


def _call(token: str, method: str, params: dict | None = None) -> dict:
    url = API.format(token=token, method=method)
    data = urllib.parse.urlencode(params or {}).encode() if params else None
    try:
        with urllib.request.urlopen(url, data=data, timeout=TIMEOUT) as r:
            payload = json.loads(r.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — network/HTTP/JSON all surface the same
        raise TelegramError(f"could not reach Telegram: {exc}") from exc
    if not payload.get("ok"):
        raise TelegramError(payload.get("description", "Telegram rejected the request"))
    return payload.get("result", {})


class TelegramNotifier:
    """Stores the bot config and sends messages."""

    def __init__(self, config_dir: str | None = None):
        self.config_dir = config_dir or default_config_dir()
        self.path = os.path.join(self.config_dir, "telegram.json")

    # ── config ──────────────────────────────────────────────────────────────
    def load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, ValueError, OSError):
            return {}

    def save(self, **fields) -> dict:
        cfg = self.load()
        cfg.update({k: v for k, v in fields.items() if v is not None})
        os.makedirs(self.config_dir, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, self.path)
        return cfg

    def status(self) -> dict:
        """Config for the UI — never returns the token itself."""
        cfg = self.load()
        token = cfg.get("token") or ""
        return {
            "configured": bool(token and cfg.get("chatId")),
            "hasToken": bool(token),
            "chatId": cfg.get("chatId"),
            "sendScans": bool(cfg.get("sendScans", True)),
            "sendFlips": bool(cfg.get("sendFlips", True)),
            "minScore": int(cfg.get("minScore", 70)),
        }

    @property
    def enabled(self) -> bool:
        cfg = self.load()
        return bool(cfg.get("token") and cfg.get("chatId"))

    # ── chat discovery ──────────────────────────────────────────────────────
    def discover_chat_id(self, token: str) -> str | None:
        """Find the chat id from the most recent message sent to the bot."""
        result = _call(token, "getUpdates", {"limit": 10})
        for update in reversed(result if isinstance(result, list) else []):
            msg = update.get("message") or update.get("channel_post") or {}
            chat = msg.get("chat") or {}
            if chat.get("id") is not None:
                return str(chat["id"])
        return None

    def connect(self, token: str, chat_id: str | None = None) -> dict:
        """Verify the token, resolve the chat id, and persist both."""
        token = (token or "").strip()
        if not token:
            raise TelegramError("a bot token is required")
        me = _call(token, "getMe")            # raises if the token is bad
        chat_id = (chat_id or "").strip() or self.discover_chat_id(token)
        if not chat_id:
            raise TelegramError(
                "Send your bot a message on Telegram first (say 'hi' to "
                f"@{me.get('username', 'your bot')}), then connect again.")
        self.save(token=token, chatId=str(chat_id), botName=me.get("username", ""))
        return self.status()

    def disconnect(self) -> dict:
        self.save(token="", chatId="")
        return self.status()

    # ── sending ─────────────────────────────────────────────────────────────
    def send(self, text: str) -> bool:
        cfg = self.load()
        token, chat_id = cfg.get("token"), cfg.get("chatId")
        if not token or not chat_id:
            raise TelegramError("Telegram is not connected yet")
        _call(token, "sendMessage", {
            "chat_id": chat_id, "text": text[:4000],
            "parse_mode": "HTML", "disable_web_page_preview": "true"})
        return True


# ── message formatting ──────────────────────────────────────────────────────
def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def format_scan(results: list[dict], *, min_score: int = 70, limit: int = 5) -> str | None:
    """A short 'here is what the scanner likes' report. None when nothing qualifies."""
    picks = [r for r in results if r.get("score", 0) >= min_score][:limit]
    if not picks:
        return None
    lines = [f"<b>⚡ Mamba Terminal — {len(picks)} opportunity"
             f"{'' if len(picks) == 1 else 'ies'}</b>"]
    for r in picks:
        fresh = r.get("daysInRegime", 0)
        tag = f" · 🌱 {fresh}d new" if 0 < fresh <= 5 else ""
        lines.append(
            f"\n<b>{_esc(r['symbol'])}</b> — {_esc(r.get('name', ''))}\n"
            f"{_esc(r.get('verdict', ''))} · score {r.get('score')} · "
            f"${r.get('lastPrice')}{tag}\n"
            f"<i>{_esc(r.get('rationale', ''))}</i>")
    lines.append("\n<i>Research screen — not investment advice.</i>")
    return "\n".join(lines)


def format_flip(event: dict) -> str:
    arrow = "📈" if event.get("to") == "bull" else "📉" if event.get("to") == "bear" else "↔️"
    return (f"{arrow} <b>{_esc(event.get('symbol', ''))}</b> regime flip: "
            f"{_esc(str(event.get('from', '')).upper())} → "
            f"<b>{_esc(str(event.get('to', '')).upper())}</b>")
