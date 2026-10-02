import math
from datetime import UTC, datetime

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from weather_edge.kalshi.markets import Bracket
from weather_edge.live import FairValue
from weather_edge.trading import shadow_report
from weather_edge.trading.demo_client import (
    PROD_BASE_URL,
    ProductionReadOnlyClient,
    ReadOnlyError,
    WrongHostError,
)
from weather_edge.trading.paper import PaperTrader
from weather_edge.trading.risk import KillSwitch, Ledger
from weather_edge.trading.sizing import SizingConfig

KEY = ed25519.Ed25519PrivateKey.generate()


class NoNetworkSession:
    """Fails the test if any request is attempted."""

    calls = 0

    def request(self, *a, **k):
        NoNetworkSession.calls += 1
        raise AssertionError("network request attempted")


# ----------------------------------------------------- read-only production client
@pytest.mark.parametrize("method", ["POST", "DELETE", "PUT", "post"])
def test_production_client_refuses_writes_before_any_request(method):
    c = ProductionReadOnlyClient(key_id="k", private_key=KEY, session=NoNetworkSession())
    with pytest.raises(ReadOnlyError):
        c.request(method, "/portfolio/events/orders", json={"ticker": "X"})
    assert NoNetworkSession.calls == 0


def test_production_client_has_no_order_path():
    c = ProductionReadOnlyClient(key_id="k", private_key=KEY, session=NoNetworkSession())
    with pytest.raises(ReadOnlyError):
        c.place_limit_order("KXHIGHNY-26OCT02-B83.5", "yes", 0.4, 1)
    with pytest.raises(ReadOnlyError):
        c.cancel_order("o1")


@pytest.mark.parametrize("url", ["https://external-api.demo.kalshi.co/trade-api/v2",
                                 "https://kalshi.com.evil.net/trade-api/v2",
                                 "http://external-api.kalshi.com/trade-api/v2"])
def test_production_client_host_pinned(url):
    with pytest.raises(WrongHostError):
        ProductionReadOnlyClient(key_id="k", private_key=KEY, base_url=url)


def test_production_hosts_accepted():
    ProductionReadOnlyClient(key_id="k", private_key=KEY)
    assert PROD_BASE_URL.startswith("https://external-api.kalshi.com")
    ProductionReadOnlyClient(key_id="k", private_key=KEY,
                             base_url="https://api.elections.kalshi.com/trade-api/v2")


# --------------------------------------------------------------- shadow trader
NOW = datetime(2026, 10, 2, 16, 0, tzinfo=UTC)
EVENT = "KXHIGHNY-26OCT02"
BRACKETS = [Bracket(f"{EVENT}-T81", -math.inf, 80), Bracket(f"{EVENT}-B81.5", 81, 82),
            Bracket(f"{EVENT}-T82", 83, math.inf)]


class RealAccount:
    """Production stand-in: reads work; any write fails the test."""

    def __init__(self):
        self.books = {BRACKETS[0].ticker: {"yes": [(0.04, 50)], "no": [(0.94, 50)]},
                      BRACKETS[1].ticker: {"yes": [(0.40, 50)], "no": [(0.58, 50)]},
                      BRACKETS[2].ticker: {"yes": [(0.45, 50)], "no": [(0.53, 50)]}}

    def request(self, method, path, **kw):
        assert method == "GET"
        return {"balance_dollars": "12.00", "portfolio_value": 0}

    def balance(self):
        return 12.0

    def positions(self):
        return []

    def fills(self):
        return []

    def resting_orders(self):
        return [{"order_id": "users-own-order", "ticker": "SOMETHING"}]

    def open_markets(self, series):
        return [{"event_ticker": EVENT, "ticker": b.ticker} for b in BRACKETS] if series == "KXHIGHNY" else []

    def orderbook(self, ticker, depth=5):
        return self.books[ticker]

    def place_limit_order(self, *a, **k):
        raise AssertionError("shadow mode tried to place a real order")

    def cancel_order(self, *a, **k):
        raise AssertionError("shadow mode tried to cancel a real order")


class FV:
    def fair_value(self, station, ev, day, lead, markets):
        return FairValue("KXHIGHNY", ev, day, lead, 81.8, 1.5, BRACKETS, [0.05, 0.75, 0.20], [0, 1, 0])


def shadow_trader(tmp_path, bankroll=1000.0):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    cfg = SizingConfig(max_market_exposure=20, max_event_exposure=30)
    return PaperTrader(RealAccount(), ledger, KillSwitch(tmp_path, ledger=ledger), cfg, FV(),
                       series=["KXHIGHNY"], now=lambda: NOW, shadow=True,
                       bankroll_override=bankroll), ledger


def test_shadow_logs_orders_without_sending(tmp_path, monkeypatch):
    monkeypatch.delenv("WEATHER_EDGE_KILL", raising=False)
    t, ledger = shadow_trader(tmp_path)
    out = t.run_once()
    shadow = ledger.records("shadow_order")
    assert out and len(shadow) == len(out) and not ledger.records("order")
    assert {r["ticker"] for r in shadow} == {BRACKETS[1].ticker, BRACKETS[2].ticker}
    assert all(r["request"]["ticker"] == r["ticker"] for r in shadow)


def test_shadow_respects_caps_across_runs(tmp_path, monkeypatch):
    monkeypatch.delenv("WEATHER_EDGE_KILL", raising=False)
    t, ledger = shadow_trader(tmp_path)
    t.run_once()
    first = len(ledger.records("shadow_order"))
    t.run_once()  # event cap already used by the first run's would-be orders
    assert len(ledger.records("shadow_order")) == first


def test_shadow_kill_switch_never_cancels_real_orders(tmp_path, monkeypatch):
    monkeypatch.delenv("WEATHER_EDGE_KILL", raising=False)
    t, ledger = shadow_trader(tmp_path)
    (tmp_path / "KILL").touch()
    assert t.run_once() == []  # RealAccount.cancel_order would raise if called


# --------------------------------------------------------------------- report
def test_shadow_report_pnl(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.log("shadow_order", ticker="A", side="yes", price=0.40, count=10, prob=0.6, edge=0.18,
               fee=0.17, cost=4.17, request={})
    ledger.log("shadow_order", ticker="B", side="no", price=0.55, count=10, prob=0.8, edge=0.23,
               fee=0.18, cost=5.68, request={})
    ledger.log("shadow_order", ticker="C", side="yes", price=0.30, count=5, prob=0.5, edge=0.18,
               fee=0.08, cost=1.58, request={})

    class Client:
        def get_market(self, t):
            return {"A": {"result": "yes"}, "B": {"result": "yes"}, "C": {"result": ""}}[t]

    df = shadow_report.score(ledger, Client()).set_index("ticker")
    assert df.loc["A", "pnl"] == pytest.approx(10 - 4.17)  # YES won
    assert df.loc["B", "pnl"] == pytest.approx(-5.68)  # bought NO, YES won
    assert df.loc["C", "pnl"] is None or df["pnl"].isna()["C"]  # pending
    text = shadow_report.summary(df.reset_index())
    assert "2 settled, 1 pending" in text
