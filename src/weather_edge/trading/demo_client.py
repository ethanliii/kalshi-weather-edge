"""Authenticated Kalshi clients, each pinned to one environment.

* DemoClient: Kalshi's DEMO exchange only. Reads and writes (paper trading).
  Credentials: KALSHI_DEMO_API_KEY_ID, KALSHI_DEMO_PRIVATE_KEY_PATH.
* ProductionReadOnlyClient: the real exchange, GET requests only. Used by
  shadow mode, which logs the orders it *would* send. Any POST/DELETE raises
  before a request is built, so this client cannot place or cancel orders.
  Credentials: KALSHI_PROD_API_KEY_ID, KALSHI_PROD_PRIVATE_KEY_PATH.

Safety: each client's host is pinned. The constructor refuses any base URL
outside that environment's documented hosts (https only), every request
re-checks the URL, and redirects are disabled.
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
PROD_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
# Production REST hosts listed in the Trade API OpenAPI `servers` block.
PROD_HOSTS = frozenset({"external-api.kalshi.com", "api.elections.kalshi.com"})


class WrongHostError(RuntimeError):
    """Raised whenever a client is pointed at a host outside its environment."""


NotDemoError = WrongHostError  # backwards-compatible name


class ReadOnlyError(RuntimeError):
    """Raised when a read-only client is asked to send anything but GET."""


class MissingCredentials(RuntimeError):
    pass


def _assert_host(url: str, hosts: frozenset[str], env_name: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in hosts:
        raise WrongHostError(f"refusing host {parsed.hostname!r}: this client is {env_name}-only")


def assert_demo(url: str) -> None:
    _assert_host(url, DEMO_HOSTS, "demo")


class _SignedClient:
    hosts: frozenset[str] = frozenset()
    env_name = ""
    env_prefix = ""
    default_base_url = ""
    key_help = ""
    read_only = False

    def __init__(
        self,
        key_id: str | None = None,
        private_key_path: str | None = None,
        base_url: str | None = None,
        session: requests.Session | None = None,
        private_key=None,
    ):
        base_url = base_url or self.default_base_url
        self._check_host(base_url)
        self.base_url = base_url.rstrip("/")
        self.key_id = key_id or env(f"{self.env_prefix}_API_KEY_ID")
        key_path = private_key_path or env(f"{self.env_prefix}_PRIVATE_KEY_PATH")
        if not self.key_id or not (key_path or private_key):
            raise MissingCredentials(
                f"Set {self.env_prefix}_API_KEY_ID and {self.env_prefix}_PRIVATE_KEY_PATH in "
                f".env ({self.key_help})."
            )
        self.private_key = private_key or load_private_key(key_path)
        self.session = session or requests.Session()
        self.session.max_redirects = 0

    def _check_host(self, url: str) -> None:
        _assert_host(url, self.hosts, self.env_name)

    # ------------------------------------------------------------------ transport
    def request(self, method: str, path: str, params: dict | None = None,
                json: dict | None = None, retries: int = 4) -> dict:
        method = method.upper()
        if self.read_only and method != "GET":
            raise ReadOnlyError(f"{self.env_name} client is read-only; refused {method} {path}")
        url = f"{self.base_url}{path}"
        self._check_host(url)
        for attempt in range(retries):
            headers = auth_headers(self.key_id, self.private_key, method, url)
            if json is not None:
                headers["Content-Type"] = "application/json"
            resp = self.session.request(method, url, params=params, json=json, headers=headers,
                                        timeout=30, allow_redirects=False)
            self._check_host(resp.url or url)
            if resp.status_code == 429:
                time.sleep(0.5 * 2**attempt)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"{self.env_name} {method} {path} -> {resp.status_code}: "
                                   f"{resp.text[:300]}")
            return resp.json() if resp.content else {}
        raise RuntimeError(f"{self.env_name} {method} {path}: rate limited")

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
        body = order_body(ticker, side, price, count, time_in_force, client_order_id)
        return {"request": body, "response": self.request("POST", "/portfolio/events/orders",
                                                          json=body)}

    def cancel_order(self, order_id: str, ticker: str | None = None) -> dict:
        params = {"market_ticker": ticker} if ticker else None
        return self.request("DELETE", f"/portfolio/events/orders/{order_id}", params=params)


def order_body(ticker: str, side: str, price: float, count: int,
               time_in_force: str = "immediate_or_cancel",
               client_order_id: str | None = None) -> dict:
    """Create Order (V2) body. `side` is the contract we buy: 'yes' or 'no'.

    The V2 book is quoted from the YES side: buying YES at p is a `bid` at p;
    buying NO at q is an `ask` (sell YES) at 1 - q.
    """
    if side not in ("yes", "no"):
        raise ValueError(side)
    if not 0 < price < 1 or count <= 0:
        raise ValueError(f"bad order price={price} count={count}")
    yes_price = price if side == "yes" else 1 - price
    return {
        "ticker": ticker,
        "side": "bid" if side == "yes" else "ask",
        "count": f"{int(count)}.00",
        "price": f"{yes_price:.4f}",
        "time_in_force": time_in_force,
        "self_trade_prevention_type": "taker_at_cross",
        "client_order_id": client_order_id or str(uuid.uuid4()),
    }


class DemoClient(_SignedClient):
    hosts = DEMO_HOSTS
    env_name = "demo"
    env_prefix = "KALSHI_DEMO"
    default_base_url = DEMO_BASE_URL
    key_help = "create a key at https://demo.kalshi.co -> Account & security -> API Keys"


class ProductionReadOnlyClient(_SignedClient):
    """Real-exchange client that can only read. Orders and cancels are impossible."""

    hosts = PROD_HOSTS
    env_name = "production"
    env_prefix = "KALSHI_PROD"
    default_base_url = PROD_BASE_URL
    key_help = "create a key at https://kalshi.com -> Account & security -> API Keys"
    read_only = True

    def place_limit_order(self, *args, **kwargs) -> dict:
        raise ReadOnlyError("production client is read-only (shadow mode); no orders are sent")

    def cancel_order(self, *args, **kwargs) -> dict:
        raise ReadOnlyError("production client is read-only (shadow mode); nothing is cancelled")
