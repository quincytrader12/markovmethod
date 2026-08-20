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
    def send(self, text: str, buttons: list | None = None) -> dict:
        """Send a message, optionally with tappable buttons under it.

        `buttons` is a list of rows, each row a list of (label, data) pairs.
        Telegram calls this an inline keyboard: tapping one sends a callback
        back to the bot rather than posting a reply into the chat, which is what
        lets an alert offer "add this to my watchlist" as a single tap.
        """
        cfg = self.load()
        token, chat_id = cfg.get("token"), cfg.get("chatId")
        if not token or not chat_id:
            raise TelegramError("Telegram is not connected yet")
        params = {"chat_id": chat_id, "text": text[:4000],
                  "parse_mode": "HTML", "disable_web_page_preview": "true"}
        if buttons:
            params["reply_markup"] = json.dumps({"inline_keyboard": [
                [{"text": label, "callback_data": data[:64]} for label, data in row]
                for row in buttons if row]})
        return _call(token, "sendMessage", params) or {}

    # ── replying to a tapped button ─────────────────────────────────────────
    def poll_callbacks(self) -> list[dict]:
        """Fetch button taps since the last poll.

        Long polling rather than a webhook: a webhook needs a public URL, and
        this is a desktop terminal. `offset` is the acknowledgement — asking for
        `last_id + 1` is what tells Telegram the earlier updates are handled, so
        a tap is never delivered twice.
        """
        cfg = self.load()
        token = cfg.get("token")
        if not token:
            return []
        params = {"limit": 20, "timeout": 0, "allowed_updates": json.dumps(["callback_query"])}
        offset = cfg.get("updateOffset")
        if offset:
            params["offset"] = int(offset)
        result = _call(token, "getUpdates", params)
        updates = result if isinstance(result, list) else []
        taps, last = [], None
        for u in updates:
            last = u.get("update_id", last)
            cq = u.get("callback_query")
            if not cq:
                continue
            taps.append({
                "id": cq.get("id"),
                "data": cq.get("data") or "",
                "messageId": (cq.get("message") or {}).get("message_id"),
                "chatId": ((cq.get("message") or {}).get("chat") or {}).get("id"),
            })
        if last is not None:
            self.save(updateOffset=int(last) + 1)
        return taps

    def answer_callback(self, callback_id: str, text: str = "") -> bool:
        """Acknowledge a tap so the button stops spinning on the phone.

        Telegram shows the text as a small toast. Skipping this leaves the
        button looking stuck even though the work was done.
        """
        cfg = self.load()
        token = cfg.get("token")
        if not token or not callback_id:
            return False
        try:
            _call(token, "answerCallbackQuery",
                  {"callback_query_id": callback_id, "text": text[:200]})
        except TelegramError:
            return False
        return True

    def edit_buttons(self, chat_id, message_id, buttons: list) -> bool:
        """Redraw a message's buttons — used to mark one as done."""
        cfg = self.load()
        token = cfg.get("token")
        if not token or message_id is None:
            return False
        try:
            _call(token, "editMessageReplyMarkup", {
                "chat_id": chat_id or cfg.get("chatId"), "message_id": message_id,
                "reply_markup": json.dumps({"inline_keyboard": [
                    [{"text": label, "callback_data": data[:64]} for label, data in row]
                    for row in buttons if row]})})
        except TelegramError:
            return False
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
        # The sector matters on a phone: three names from one sector is a
        # concentrated bet, and that is invisible from tickers alone.
        sec = r.get("sector") or ""
        sec_line = f" · {_esc(sec)}" if sec else ""
        lines.append(
            f"\n<b>{_esc(r['symbol'])}</b> — {_esc(r.get('name', ''))}\n"
            f"{_esc(r.get('verdict', ''))} · score {r.get('score')} · "
            f"${r.get('lastPrice')}{tag}{sec_line}\n"
            f"<i>{_esc(r.get('rationale', ''))}</i>")
    lines.append("\n<i>Research screen — not investment advice.</i>")
    return "\n".join(lines)


def format_flip(event: dict) -> str:
    from .sectors import sector_name

    arrow = "📈" if event.get("to") == "bull" else "📉" if event.get("to") == "bear" else "↔️"
    sec = sector_name(event.get("symbol", ""))
    tail = f"\n<i>{_esc(sec)}</i>" if sec else ""
    return (f"{arrow} <b>{_esc(event.get('symbol', ''))}</b> regime flip: "
            f"{_esc(str(event.get('from', '')).upper())} → "
            f"<b>{_esc(str(event.get('to', '')).upper())}</b>{tail}")


# ── buttons ─────────────────────────────────────────────────────────────────
ADD_PREFIX = "add:"
DONE_MARK = "✓ "


def watch_buttons(symbols: list, per_row: int = 3, added: set | None = None) -> list:
    """One "add to watchlist" button per symbol, laid out in rows.

    Every name in a report gets its own button rather than one button for the
    whole batch: an alert usually contains a few names and you rarely want all
    of them. A symbol already on the list is shown ticked and inert, so the
    keyboard doubles as a record of what you have already taken.
    """
    added = {s.upper() for s in (added or set())}
    rows, row = [], []
    for raw in symbols:
        sym = str(raw).strip().upper()
        if not sym:
            continue
        if sym in added:
            row.append((f"{DONE_MARK}{sym}", f"{ADD_PREFIX}{sym}"))
        else:
            row.append((f"➕ {sym}", f"{ADD_PREFIX}{sym}"))
        if len(row) >= per_row:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return rows


def parse_callback(data: str) -> tuple[str, str]:
    """('add', 'AAPL') from 'add:AAPL'. ('', '') for anything unrecognised."""
    text = (data or "").strip()
    if text.startswith(ADD_PREFIX):
        sym = text[len(ADD_PREFIX):].strip().upper()
        return ("add", sym) if sym else ("", "")
    return ("", "")
