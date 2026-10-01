"""Authenticated client for Kalshi's DEMO exchange — and only the demo exchange.

Safety: the host is pinned. The constructor refuses any base URL whose host is
not one of Kalshi's documented demo API hosts, every request re-checks the URL,
and redirects are disabled, so a production endpoint cannot be reached through
configuration mistakes or redirects.
Credentials come from .env (KALSHI_DEMO_API_KEY_ID, KALSHI_DEMO_PRIVATE_KEY_PATH)
and must be demo keys; production keys do not work on demo and vice versa.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any
from urllib.parse import urlparse

import requests

from weather_edge.config import env
from weather_edge.kalshi.auth import auth_headers, load_private_key

log = logging.getLogger(__name__)

DEMO_BASE_URL = "https://external-api.demo.kalshi.co/trade-api/v2"
# Documented demo REST hosts (docs.kalshi.com/getting_started/demo_env).
DEMO_HOSTS = frozenset({"external-api.demo.kalshi.co", "demo-api.kalshi.co"})


class NotDemoError(RuntimeError):
    """Raised whenever anything tries to talk to a non-demo host."""


def assert_demo(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in DEMO_HOSTS:
        host = parsed.hostname
        raise NotDemoError(f"refusing non-demo host {host!r}; paper trading is demo-only")


class MissingCredentials(RuntimeError):
    pass


class DemoClient:
    def __init__(
        self,
        key_id: str | None = None,
        private_key_path: str | None = None,
        base_url: str = DEMO_BASE_URL,
        session: requests.Session | None = None,
        private_key=None,
    ):
        assert_demo(base_url)
        self.base_url = base_url.rstrip("/")
        self.key_id = key_id or env("KALSHI_DEMO_API_KEY_ID")
        key_path = private_key_path or env("KALSHI_DEMO_PRIVATE_KEY_PATH")
        if not self.key_id or not (key_path or private_key):
            raise MissingCredentials(
                "Set KALSHI_DEMO_API_KEY_ID and KALSHI_DEMO_PRIVATE_KEY_PATH in .env "
                "(create a key at https://demo.kalshi.co -> Account & security -> API Keys)."
            )
        self.private_key = private_key or load_private_key(key_path)
        self.session = session or requests.Session()
        self.session.max_redirects = 0

    # ------------------------------------------------------------------ transport
    def request(self, method: str, path: str, params: dict | None = None,
                json: dict | None = None, retries: int = 4) -> dict:
        url = f"{self.base_url}{path}"
        assert_demo(url)
        for attempt in range(retries):
            headers = auth_headers(self.key_id, self.private_key, method, url)
            if json is not None:
                headers["Content-Type"] = "application/json"
            resp = self.session.request(method, url, params=params, json=json, headers=headers,
                                        timeout=30, allow_redirects=False)
            assert_demo(resp.url or url)
            if resp.status_code == 429:
                time.sleep(0.5 * 2**attempt)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"demo {method} {path} -> {resp.status_code}: {resp.text[:300]}")
            return resp.json() if resp.content else {}
        raise RuntimeError(f"demo {method} {path}: rate limited")

    # ------------------------------------------------------------------ reads
    def balance(self) -> float:
        r = self.request("GET", "/portfolio/balance")
        if "balance_dollars" in r:
            return float(r["balance_dollars"])
        return float(r["balance"]) / 100  # legacy integer cents

    def positions(self) -> list[dict]:
        out, cursor = [], None
        while True:
            params = {"limit": 1000, **({"cursor": cursor} if cursor else {})}
            r = self.request("GET", "/portfolio/positions", params=params)
            out += r.get("market_positions") or []
            cursor = r.get("cursor")
            if not cursor:
                return out

    def resting_orders(self) -> list[dict]:
        r = self.request("GET", "/portfolio/orders", params={"status": "resting", "limit": 1000})
        return r.get("orders") or []

    def fills(self, min_ts: int | None = None) -> list[dict]:
        params: dict[str, Any] = {"limit": 1000}
        if min_ts:
            params["min_ts"] = min_ts
        return self.request("GET", "/portfolio/fills", params=params).get("fills") or []

    def open_markets(self, series_ticker: str) -> list[dict]:
        params = {"series_ticker": series_ticker, "status": "open", "limit": 1000}
        return self.request("GET", "/markets", params=params).get("markets") or []

    def orderbook(self, ticker: str, depth: int = 5) -> dict:
        raw = self.request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})
        book = raw.get("orderbook_fp") or {}
        levels = lambda side: sorted(  # noqa: E731
            ((float(p), float(q)) for p, q in book.get(f"{side}_dollars") or []),
            key=lambda r: -r[0])
        return {"yes": levels("yes"), "no": levels("no")}

    # ------------------------------------------------------------------ writes
    def place_limit_order(self, ticker: str, side: str, price: float, count: int,
                          time_in_force: str = "immediate_or_cancel",
                          client_order_id: str | None = None) -> dict:
        """Limit order via Create Order (V2). `side` is the contract we buy: 'yes' or 'no'.

        The V2 book is quoted from the YES side: buying YES at p is a `bid` at p;
        buying NO at q is an `ask` (sell YES) at 1 - q.
        """
        if side not in ("yes", "no"):
            raise ValueError(side)
        if not 0 < price < 1 or count <= 0:
            raise ValueError(f"bad order price={price} count={count}")
        yes_price = price if side == "yes" else 1 - price
        body = {
            "ticker": ticker,
            "side": "bid" if side == "yes" else "ask",
            "count": f"{int(count)}.00",
            "price": f"{yes_price:.4f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": "taker_at_cross",
            "client_order_id": client_order_id or str(uuid.uuid4()),
        }
        return {"request": body, "response": self.request("POST", "/portfolio/events/orders",
                                                          json=body)}

    def cancel_order(self, order_id: str, ticker: str | None = None) -> dict:
        params = {"market_ticker": ticker} if ticker else None
        return self.request("DELETE", f"/portfolio/events/orders/{order_id}", params=params)
