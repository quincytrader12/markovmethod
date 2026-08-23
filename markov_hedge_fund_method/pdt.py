"""Pattern-day-trader guard.

FINRA's rule: in a margin account, four or more day trades inside five rolling
business days makes you a pattern day trader, and a pattern day trader must keep
$25,000 of equity. Fall short and the account is restricted — typically ninety
days, or until the balance is met. A day trade is opening and closing the same
security in the same session; holding overnight is not one, however fast the
trade was decided.

That last sentence is why this is a safety net rather than a cage. Someone
holding positions for days is not day trading and will never approach the limit.
The failure mode worth protecting against is the accidental one: buying
something in the morning, changing your mind after lunch, and discovering weeks
later that three such afternoons cost you the use of your account.

So the guard says nothing at all until a specific order would actually create a
day trade, and only then does it count. The rest of the time it is a number in
the corner.

Nothing here talks to a broker. It takes the count Alpaca already maintains and
the day's fills, and answers one question: would submitting this close the door.
"""

from __future__ import annotations

from dataclasses import dataclass

# FINRA's numbers. A pattern day trader must hold this much equity, and below it
# you get three day trades per rolling five business days — the fourth is what
# flags the account.
EQUITY_FLOOR = 25_000.0
FREE_DAY_TRADES = 3
WINDOW_DAYS = 5


@dataclass(frozen=True)
class Verdict:
    """What the guard thinks of one proposed order."""

    allowed: bool
    is_day_trade: bool
    used: int
    remaining: int
    headline: str
    detail: str
    severity: str          # "ok" | "info" | "warn" | "block"

    @property
    def blocking(self) -> bool:
        return not self.allowed


def exempt(equity: float, pattern_day_trader: bool = False) -> bool:
    """Above the floor the rule stops binding, so the guard stops speaking.

    An account already flagged as a pattern day trader is exempt only while it
    stays above the floor — that is exactly the condition it must keep.
    """
    return float(equity or 0.0) >= EQUITY_FLOOR


def remaining_trades(used: int, equity: float) -> int | None:
    """Day trades left before the fourth one flags the account.

    None means the limit does not apply, which is not the same as "many left"
    and should not be rendered as a number.
    """
    if exempt(equity):
        return None
    return max(0, FREE_DAY_TRADES - int(used or 0))


def would_be_day_trade(symbol: str, side: str, opened_today: set[str] | dict) -> bool:
    """Does closing this position today complete a round trip opened today?

    `opened_today` is the set of symbols with a fill on the opposite side during
    this session. A day trade needs both legs in one session; either leg alone is
    just a position.
    """
    if not symbol:
        return False
    sym = symbol.strip().upper()
    if sym not in {s.strip().upper() for s in opened_today}:
        return False
    # Buying more of something bought today adds to a position; it does not
    # close anything. Only the opposite side completes the round trip.
    return str(side or "").strip().lower() in ("sell", "short")


def evaluate(*, symbol: str, side: str, equity: float, used: int,
             opened_today: set[str] | dict | None = None,
             pattern_day_trader: bool = False,
             closing: bool | None = None) -> Verdict:
    """Judge one proposed order.

    `closing` overrides the fill-history inference when the caller already knows
    the order reduces a position — useful because the day's fills are not always
    reachable, and a missing history should not silently turn the guard off.
    """
    used = int(used or 0)
    left = remaining_trades(used, equity)

    if exempt(equity, pattern_day_trader):
        return Verdict(True, False, used, FREE_DAY_TRADES, "",
                       f"Equity is above ${EQUITY_FLOOR:,.0f}, so the day-trade "
                       "limit does not apply.", "ok")

    opened = set(opened_today or ())
    is_dt = bool(closing) if closing is not None else would_be_day_trade(symbol, side, opened)

    if not is_dt:
        return Verdict(True, False, used, left or 0, "",
                       f"Not a day trade — nothing bought in this session is being "
                       f"closed. {left} of {FREE_DAY_TRADES} left in the rolling "
                       f"{WINDOW_DAYS}-day window.", "ok")

    if left is not None and left <= 0:
        return Verdict(
            False, True, used, 0,
            "This would be your 4th day trade",
            f"You have used {used} day trades in the last {WINDOW_DAYS} business "
            f"days and your equity is under ${EQUITY_FLOOR:,.0f}. A fourth flags "
            "the account as a pattern day trader and restricts it for 90 days. "
            "Holding this overnight instead costs nothing and is not a day trade.",
            "block")

    if left == 1:
        return Verdict(
            True, True, used, left,
            "Last day trade before the limit",
            f"This is day trade {used + 1} of {FREE_DAY_TRADES}. One more inside "
            f"{WINDOW_DAYS} business days flags the account.", "warn")

    return Verdict(
        True, True, used, left,
        f"Day trade {used + 1} of {FREE_DAY_TRADES}",
        f"Closing a position opened today. {left - 1} will remain in the rolling "
        f"{WINDOW_DAYS}-business-day window.", "info")


def summary(equity: float, used: int, pattern_day_trader: bool = False) -> dict:
    """The always-on counter, for the header."""
    left = remaining_trades(used, equity)
    return {
        "applies": left is not None,
        "used": int(used or 0),
        "limit": FREE_DAY_TRADES,
        "remaining": left,
        "equityFloor": EQUITY_FLOOR,
        "patternDayTrader": bool(pattern_day_trader),
        "text": ("day-trade limit does not apply above "
                 f"${EQUITY_FLOOR:,.0f}" if left is None
                 else f"{used} of {FREE_DAY_TRADES} day trades used"),
    }
