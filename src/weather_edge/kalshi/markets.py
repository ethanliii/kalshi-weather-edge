"""Parse Kalshi temperature markets into integer-degree brackets.

The settlement value is an integer °F. Each market in an event is one bracket:

  strike_type  example ticker            subtitle        reported temps covered
  -----------  ------------------------  --------------  ----------------------
  less         KXHIGHNY-26OCT02-T81      "80° or below"  (-inf, cap-1]
  between      KXHIGHNY-26OCT02-B83.5    "83° to 84°"    [floor, cap]
  greater      KXHIGHNY-26OCT02-T88      "89° or above"  [floor+1, +inf)

An event's brackets must tile the integers with no gaps or overlaps so that
exactly one bracket settles YES. `validate_partition` checks that.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1
)}  # fmt: skip
_EVENT_DATE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})$")


@dataclass(frozen=True)
class Bracket:
    ticker: str
    lo: float  # lowest reported integer temperature that resolves YES (-inf for 'less')
    hi: float  # highest reported integer temperature that resolves YES (+inf for 'greater')

    def contains(self, temp_f: int) -> bool:
        return self.lo <= temp_f <= self.hi

    @property
    def label(self) -> str:
        if math.isinf(self.lo):
            return f"≤{int(self.hi)}"
        if math.isinf(self.hi):
            return f"≥{int(self.lo)}"
        return f"{int(self.lo)}-{int(self.hi)}"


def event_date(event_ticker: str) -> date:
    """'KXHIGHNY-26OCT02' -> date(2026, 10, 2)."""
    m = _EVENT_DATE.search(event_ticker)
    if not m:
        raise ValueError(f"cannot parse date from event ticker {event_ticker!r}")
    yy, mon, dd = m.groups()
    return date(2000 + int(yy), _MONTHS[mon], int(dd))


def bracket_from_market(market: dict) -> Bracket:
    kind = market.get("strike_type")
    floor, cap = market.get("floor_strike"), market.get("cap_strike")
    if kind == "less":
        return Bracket(market["ticker"], -math.inf, float(cap) - 1)
    if kind == "greater":
        return Bracket(market["ticker"], float(floor) + 1, math.inf)
    if kind == "between":
        return Bracket(market["ticker"], float(floor), float(cap))
    raise ValueError(f"unsupported strike_type {kind!r} for {market.get('ticker')}")


def sorted_brackets(markets: list[dict]) -> list[Bracket]:
    return sorted((bracket_from_market(m) for m in markets), key=lambda b: b.lo)


def validate_partition(brackets: list[Bracket]) -> None:
    """Raise ValueError unless brackets tile the integers exactly once."""
    if not brackets:
        raise ValueError("no brackets")
    bs = sorted(brackets, key=lambda b: b.lo)
    if not math.isinf(bs[0].lo) or not math.isinf(bs[-1].hi):
        raise ValueError("brackets do not cover both tails")
    for a, b in zip(bs, bs[1:], strict=False):
        if b.lo != a.hi + 1:
            raise ValueError(f"gap/overlap between {a.ticker} and {b.ticker}")
    for b in bs:
        if b.lo > b.hi:
            raise ValueError(f"empty bracket {b.ticker}")


def winning_bracket(brackets: list[Bracket], temp_f: int) -> Bracket:
    hits = [b for b in brackets if b.contains(temp_f)]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} brackets contain {temp_f}")
    return hits[0]


def settlement_source(rules_primary: str) -> str:
    """Classify which data source a market's rules say it settles on."""
    text = rules_primary or ""
    if "Weather Company" in text:
        return "TWC"
    if "National Weather Service" in text or "Climatological Report" in text:
        return "NWS_CLI"
    return "UNKNOWN"


def close_time(market: dict) -> datetime | None:
    from weather_edge.kalshi.client import parse_ts

    return parse_ts(market.get("close_time"))
