"""Read-only Kalshi REST client for market data.

Public market-data endpoints need no authentication. The client handles:
  * cursor pagination,
  * 429 back-off (Kalshi sends no Retry-After header, so we back off exponentially),
  * a process-wide request spacing so threaded callers stay under the rate limit,
  * the live/historical data split: settled markets (and their candlesticks) older
    than `GET /historical/cutoff` are only served by `/historical/...` endpoints,
  * the two candlestick schemas (pre-2026 `close`, current `close_dollars`).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import requests

log = logging.getLogger(__name__)

PROD_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiAPIError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.status = status


class _Throttle:
    """Enforce a minimum interval between requests across threads."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.min_interval_s
        if delay > 0:
            time.sleep(delay)


class KalshiClient:
    """Thin wrapper over the Kalshi Trade API v2 market-data endpoints."""

    def __init__(
        self,
        base_url: str = PROD_BASE_URL,
        session: requests.Session | None = None,
        min_interval_s: float = 0.08,
        max_retries: int = 8,
        timeout_s: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.throttle = _Throttle(min_interval_s)
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._cutoff: dict[str, datetime] | None = None

    # ------------------------------------------------------------------ transport
    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries):
            self.throttle.wait()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout_s)
            except requests.RequestException as exc:
                log.warning("network error %s (attempt %d): %s", url, attempt + 1, exc)
                time.sleep(min(2**attempt, 30))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(min(0.5 * 2**attempt, 30))
                continue
            if resp.status_code != 200:
                raise KalshiAPIError(resp.status_code, resp.text, resp.url)
            return resp.json()
        raise KalshiAPIError(429, "retries exhausted", url)

    def paginate(
        self, path: str, key: str, params: dict[str, Any] | None = None, page_size: int = 1000
    ) -> Iterator[dict]:
        params = dict(params or {})
        params.setdefault("limit", page_size)
        while True:
            page = self.get(path, params)
            yield from page.get(key) or []
            cursor = page.get("cursor")
            if not cursor:
                return
            params["cursor"] = cursor

    # ------------------------------------------------------------------ metadata
    def get_series(self, series_ticker: str) -> dict:
        return self.get(f"/series/{series_ticker}")["series"]

    def get_event(self, event_ticker: str) -> dict:
        """Event with nested markets (only markets newer than the historical cutoff)."""
        return self.get(f"/events/{event_ticker}", {"with_nested_markets": "true"})["event"]

    def historical_cutoff(self) -> dict[str, datetime]:
        if self._cutoff is None:
            raw = self.get("/historical/cutoff")
            self._cutoff = {k: parse_ts(v) for k, v in raw.items()}
        return self._cutoff

    # ------------------------------------------------------------------ markets
    def list_markets(
        self, series_ticker: str | None = None, status: str | None = None, **params: Any
    ) -> list[dict]:
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        return list(self.paginate("/markets", "markets", params))

    def list_historical_markets(self, series_ticker: str, **params: Any) -> list[dict]:
        params["series_ticker"] = series_ticker
        return list(self.paginate("/historical/markets", "markets", params))

    def all_settled_markets(self, series_ticker: str) -> list[dict]:
        """Every settled market in a series, merging the historical and live tiers."""
        seen: dict[str, dict] = {}
        for m in self.list_historical_markets(series_ticker):
            seen[m["ticker"]] = m
        for m in self.list_markets(series_ticker, status="settled"):
            seen.setdefault(m["ticker"], m)
        return list(seen.values())

    def get_market(self, ticker: str) -> dict:
        """One market, falling back to the historical tier for old settled markets."""
        try:
            return self.get(f"/markets/{ticker}")["market"]
        except KalshiAPIError as exc:
            if exc.status != 404:
                raise
            return self.get(f"/historical/markets/{ticker}")["market"]

    def open_markets(self, series_ticker: str) -> list[dict]:
        return self.list_markets(series_ticker, status="open")

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """Order book as {'yes': [(price, size), ...], 'no': [...]} sorted best-first.

        Kalshi returns resting *bids* only, for YES and for NO. A NO bid at p is
        an offer to sell YES at 1 - p, so best YES ask = 1 - best NO bid.
        """
        raw = self.get(f"/markets/{ticker}/orderbook", {"depth": depth})
        book = raw.get("orderbook_fp") or raw.get("orderbook") or {}

        def levels(side: str) -> list[tuple[float, float]]:
            rows = book.get(f"{side}_dollars") or []
            out = [(float(p), float(q)) for p, q in rows]
            return sorted(out, key=lambda r: -r[0])

        return {"yes": levels("yes"), "no": levels("no")}

    # ------------------------------------------------------------------ candles
    def candlesticks(
        self,
        series_ticker: str,
        market_ticker: str,
        start: datetime,
        end: datetime,
        period_minutes: int = 60,
        settled_at: datetime | None = None,
    ) -> list[dict]:
        """Normalised candlesticks for one market.

        Markets that settled before the historical cutoff are served from
        `/historical/markets/{ticker}/candlesticks`; everything else from the
        live series endpoint.
        """
        params = {
            "start_ts": int(start.timestamp()),
            "end_ts": int(end.timestamp()),
            "period_interval": period_minutes,
        }
        cutoff = self.historical_cutoff()["market_settled_ts"]
        if settled_at is not None and settled_at < cutoff:
            path = f"/historical/markets/{market_ticker}/candlesticks"
        else:
            path = f"/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        raw = self.get(path, params).get("candlesticks") or []
        return [normalise_candle(c) for c in raw]


# ---------------------------------------------------------------------- helpers
def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _num(x: Any) -> float | None:
    return None if x is None or x == "" else float(x)


def _ohlc(d: dict | None, field: str) -> float | None:
    if not d:
        return None
    if f"{field}_dollars" in d:
        return _num(d[f"{field}_dollars"])
    return _num(d.get(field))


def normalise_candle(c: dict) -> dict:
    """Flatten a candlestick from either API schema into floats (dollars)."""
    return {
        "end_ts": datetime.fromtimestamp(c["end_period_ts"], UTC),
        "yes_bid_close": _ohlc(c.get("yes_bid"), "close"),
        "yes_ask_close": _ohlc(c.get("yes_ask"), "close"),
        "price_close": _ohlc(c.get("price"), "close"),
        "volume": _num(c.get("volume_fp", c.get("volume"))) or 0.0,
        "open_interest": _num(c.get("open_interest_fp", c.get("open_interest"))) or 0.0,
    }
