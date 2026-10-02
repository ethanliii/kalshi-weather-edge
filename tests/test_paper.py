import base64
import math
from datetime import UTC, date, datetime

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from weather_edge.kalshi.auth import auth_headers, sign
from weather_edge.kalshi.markets import Bracket
from weather_edge.live import FairValue, lead_for
from weather_edge.stations import STATIONS
from weather_edge.trading.demo_client import (
    DEMO_BASE_URL,
    DemoClient,
    MissingCredentials,
    NotDemoError,
    assert_demo,
)
from weather_edge.trading.paper import PaperTrader, top_of_book
from weather_edge.trading.risk import KillSwitch, KillSwitchConfig, Ledger
from weather_edge.trading.sizing import SizingConfig

RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


# --------------------------------------------------------------- signing
def test_rsa_pss_signature_verifies_and_ignores_query():
    sig = sign(RSA_KEY, "1703123456789", "get",
               "https://external-api.demo.kalshi.co/trade-api/v2/portfolio/orders?limit=5")
    RSA_KEY.public_key().verify(
        base64.b64decode(sig), b"1703123456789GET/trade-api/v2/portfolio/orders",
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256())


def test_ed25519_signature_verifies():
    key = ed25519.Ed25519PrivateKey.generate()
    sig = sign(key, "1", "POST", "/trade-api/v2/portfolio/events/orders")
    key.public_key().verify(base64.b64decode(sig), b"1POST/trade-api/v2/portfolio/events/orders")


def test_auth_headers_shape():
    h = auth_headers("key-id", RSA_KEY, "GET", DEMO_BASE_URL + "/portfolio/balance")
    assert set(h) == {"KALSHI-ACCESS-KEY", "KALSHI-ACCESS-TIMESTAMP", "KALSHI-ACCESS-SIGNATURE"}
    assert len(h["KALSHI-ACCESS-TIMESTAMP"]) == 13  # milliseconds


# ------------------------------------------------------------ demo pinning
@pytest.mark.parametrize("url", [
    "https://api.elections.kalshi.com/trade-api/v2",
    "https://external-api.kalshi.com/trade-api/v2",
    "https://demo.kalshi.co.evil.com/trade-api/v2",
    "https://kalshi.co/trade-api/v2",
    "http://external-api.demo.kalshi.co/trade-api/v2",  # plaintext
])
def test_production_and_lookalike_hosts_refused(url):
    with pytest.raises(NotDemoError):
        assert_demo(url)
    with pytest.raises(NotDemoError):
        DemoClient(key_id="k", private_key=RSA_KEY, base_url=url)


def test_demo_hosts_allowed():
    assert_demo(DEMO_BASE_URL)
    assert_demo("https://demo-api.kalshi.co/trade-api/v2")


@pytest.mark.parametrize("content,msg", [
    (b"", "empty"),
    (b"a952bcbe-ec3b-4b5b-b8f9-11dae589608c\n", "no -----BEGIN"),
    (b"-----BEGIN PRIVATE KEY-----MIIEv...one line-----END PRIVATE KEY-----", "could not parse"),
])
def test_bad_key_files_give_clear_errors(tmp_path, content, msg):
    from weather_edge.kalshi.auth import KeyFileError, load_private_key

    key = tmp_path / "demo.key"
    key.write_bytes(content)
    with pytest.raises(KeyFileError, match=msg):
        load_private_key(key)
    with pytest.raises(KeyFileError, match="not found"):
        load_private_key(tmp_path / "missing.key")


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("KALSHI_DEMO_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_DEMO_PRIVATE_KEY_PATH", raising=False)
    with pytest.raises(MissingCredentials):
        DemoClient()


class RecordingSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, params=None, json=None, headers=None, timeout=None,
                allow_redirects=True):
        self.calls.append((method, url, json, headers))

        class R:
            status_code, content = 201, b"{}"
            text = "{}"

            def json(self):
                return {"order_id": "o1", "fill_count": "0.00", "remaining_count": "0.00"}
        r = R()
        r.url = url
        return r


def test_buy_no_is_sent_as_yes_ask_at_complement():
    sess = RecordingSession()
    c = DemoClient(key_id="k", private_key=RSA_KEY, session=sess)
    out = c.place_limit_order("KXHIGHNY-26OCT02-B83.5", "no", 0.62, 7)
    body = out["request"]
    assert body["side"] == "ask" and body["price"] == "0.3800" and body["count"] == "7.00"
    assert body["time_in_force"] == "immediate_or_cancel"
    assert sess.calls[0][1] == DEMO_BASE_URL + "/portfolio/events/orders"
    with pytest.raises(ValueError):
        c.place_limit_order("T", "yes", 1.0, 1)


# ------------------------------------------------------------- kill switch
def test_kill_switch_triggers_and_persists(tmp_path, monkeypatch):
    monkeypatch.delenv("WEATHER_EDGE_KILL", raising=False)
    ks = KillSwitch(tmp_path, KillSwitchConfig(max_daily_loss=50, max_consecutive_errors=3))
    ks.check_pnl(-49.99)
    assert not ks.tripped
    ks.check_pnl(-50)
    assert ks.tripped and "daily loss" in ks.reason()
    assert KillSwitch(tmp_path).tripped  # survives restart


def test_kill_switch_errors_reset_on_success(tmp_path, monkeypatch):
    monkeypatch.delenv("WEATHER_EDGE_KILL", raising=False)
    ks = KillSwitch(tmp_path, KillSwitchConfig(max_consecutive_errors=3))
    ks.record_error(RuntimeError("a"))
    ks.record_error(RuntimeError("b"))
    ks.record_success()
    ks.record_error(RuntimeError("c"))
    assert not ks.tripped
    ks.record_error(RuntimeError("d"))
    ks.record_error(RuntimeError("e"))
    assert ks.tripped


def test_error_count_survives_restarts(tmp_path, monkeypatch):
    monkeypatch.delenv("WEATHER_EDGE_KILL", raising=False)
    cfg = KillSwitchConfig(max_consecutive_errors=3)
    for _ in range(3):  # a fresh process per failure, as under cron
        assert not KillSwitch(tmp_path, cfg).tripped
        KillSwitch(tmp_path, cfg).record_error(RuntimeError("boom"))
    assert KillSwitch(tmp_path, cfg).tripped


def test_trader_api_error_is_counted(tmp_path, monkeypatch):
    monkeypatch.delenv("WEATHER_EDGE_KILL", raising=False)
    ex = FakeExchange()
    ex.balance = lambda: (_ for _ in ()).throw(RuntimeError("503"))
    t, ledger = trader(tmp_path, ex)
    with pytest.raises(RuntimeError):
        t.run_once()
    assert t.kill.consecutive_errors == 1 and ledger.records("error")


def test_trader_ignores_events_after_decision_window(tmp_path, monkeypatch):
    monkeypatch.delenv("WEATHER_EDGE_KILL", raising=False)
    ex = FakeExchange()
    t, _ = trader(tmp_path, ex)
    t.now = lambda: datetime(2026, 10, 2, 21, 0, tzinfo=UTC)  # 16:00 EST: high already observed
    t.run_once()
    assert ex.orders == []


def test_manual_kill(tmp_path, monkeypatch):
    monkeypatch.delenv("WEATHER_EDGE_KILL", raising=False)
    ks = KillSwitch(tmp_path)
    (tmp_path / "KILL").touch()
    assert ks.tripped
    (tmp_path / "KILL").unlink()
    monkeypatch.setenv("WEATHER_EDGE_KILL", "1")
    assert ks.tripped


# ------------------------------------------------------------ trader loop
NOW = datetime(2026, 10, 2, 16, 0, tzinfo=UTC)  # 11:00 EST in NYC: inside the lead-1 window
EVENT = "KXHIGHNY-26OCT02"
BRACKETS = [Bracket(f"{EVENT}-T81", -math.inf, 80), Bracket(f"{EVENT}-B81.5", 81, 82),
            Bracket(f"{EVENT}-T82", 83, math.inf)]


class FakeExchange:
    def __init__(self, resting=None):
        self.orders, self.cancelled = [], []
        self.resting = resting or []
        self.books = {
            BRACKETS[0].ticker: {"yes": [(0.04, 50)], "no": [(0.94, 50)]},  # ask 0.06
            BRACKETS[1].ticker: {"yes": [(0.40, 50)], "no": [(0.58, 50)]},  # ask 0.42
            BRACKETS[2].ticker: {"yes": [(0.45, 50)], "no": [(0.53, 50)]},  # ask 0.47
        }

    def request(self, method, path, **kw):
        return {"balance_dollars": "1000.00", "portfolio_value": 0}

    def balance(self):
        return 1000.0

    def positions(self):
        return []

    def fills(self):
        return [{"fill_id": "f1", "ticker": BRACKETS[1].ticker}]

    def resting_orders(self):
        return self.resting

    def cancel_order(self, order_id, ticker=None):
        self.cancelled.append(order_id)
        return {"order_id": order_id}

    def open_markets(self, series):
        return [{"event_ticker": EVENT, "ticker": b.ticker} for b in BRACKETS] if series == "KXHIGHNY" else []

    def orderbook(self, ticker, depth=5):
        return self.books[ticker]

    def place_limit_order(self, ticker, side, price, count):
        self.orders.append((ticker, side, price, count))
        return {"request": {"ticker": ticker}, "response": {"order_id": f"o{len(self.orders)}"}}


class FakeFV:
    def fair_value(self, station, ev, day, lead, markets):
        # Model: 5% / 75% / 20%. Bracket 1 YES is cheap at 0.42; bracket 2 NO is cheap at 0.55.
        return FairValue("KXHIGHNY", ev, day, lead, 81.8, 1.5, BRACKETS, [0.05, 0.75, 0.20], [0, 1, 0])


def trader(tmp_path, exchange, **sizing):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    cfg = SizingConfig(**{"max_market_exposure": 20, "max_event_exposure": 30, **sizing})
    return PaperTrader(exchange, ledger, KillSwitch(tmp_path, ledger=ledger), cfg, FakeFV(),
                       series=["KXHIGHNY"], now=lambda: NOW), ledger


def test_trader_places_capped_orders_and_logs(tmp_path, monkeypatch):
    monkeypatch.delenv("WEATHER_EDGE_KILL", raising=False)
    ex = FakeExchange()
    t, ledger = trader(tmp_path, ex)
    t.run_once()
    sides = {o[0]: (o[1], o[2]) for o in ex.orders}
    assert sides[BRACKETS[1].ticker] == ("yes", 0.42)
    assert sides[BRACKETS[2].ticker] == ("no", 0.55)
    assert BRACKETS[0].ticker not in sides  # 5% vs bid 0.04 / ask 0.06: no edge after fees
    for _, _, price, count in ex.orders:
        assert count * price <= 20  # per-market cap
    assert sum(c * p for _, _, p, c in ex.orders) <= 30 + 1  # per-event cap (plus fees)
    kinds = [r["kind"] for r in ledger.records()]
    assert kinds.count("order") == len(ex.orders) and "fill" in kinds and "fair_value" in kinds


def test_trader_halts_and_cancels_when_killed(tmp_path, monkeypatch):
    monkeypatch.delenv("WEATHER_EDGE_KILL", raising=False)
    ex = FakeExchange(resting=[{"order_id": "r1", "ticker": BRACKETS[0].ticker}])
    t, ledger = trader(tmp_path, ex)
    (tmp_path / "KILL").touch()
    assert t.run_once() == []
    assert ex.orders == [] and ex.cancelled == ["r1"]
    assert ledger.records("halted")


def test_trader_respects_displayed_size(tmp_path, monkeypatch):
    monkeypatch.delenv("WEATHER_EDGE_KILL", raising=False)
    ex = FakeExchange()
    ex.books[BRACKETS[1].ticker] = {"yes": [(0.40, 50)], "no": [(0.58, 3)]}  # only 3 offered
    t, _ = trader(tmp_path, ex)
    t.run_once()
    assert [o[3] for o in ex.orders if o[0] == BRACKETS[1].ticker] == [3]


def test_top_of_book():
    assert top_of_book({"yes": [(0.40, 5)], "no": [(0.58, 7)]}) == (0.40, 5, 0.42, 7)
    assert top_of_book({"yes": [], "no": []}) == (None, 0.0, None, 0.0)


def test_lead_for_respects_decision_time():
    ny = STATIONS["KXHIGHNY"]
    assert lead_for(ny, date(2026, 10, 2), datetime(2026, 10, 2, 14, 59, tzinfo=UTC)) is None
    assert lead_for(ny, date(2026, 10, 2), datetime(2026, 10, 2, 15, 0, tzinfo=UTC)) == 1
    assert lead_for(ny, date(2026, 10, 3), datetime(2026, 10, 2, 15, 0, tzinfo=UTC)) == 2
    assert lead_for(ny, date(2026, 10, 5), datetime(2026, 10, 2, 15, 0, tzinfo=UTC)) is None
    # The window closes two hours after the decision time.
    assert lead_for(ny, date(2026, 10, 2), datetime(2026, 10, 2, 16, 59, tzinfo=UTC)) == 1
    assert lead_for(ny, date(2026, 10, 2), datetime(2026, 10, 2, 17, 0, tzinfo=UTC)) is None
    assert lead_for(ny, date(2026, 10, 2), datetime(2026, 10, 3, 4, 59, tzinfo=UTC)) is None


def test_daily_snapshot_only_inside_window(monkeypatch):
    from weather_edge import daily

    class Client:
        def open_markets(self, s):
            if s != "KXHIGHNY":
                return []
            return [{"event_ticker": EVENT, "ticker": b.ticker, "yes_bid_dollars": "0.30",
                     "yes_ask_dollars": "0.32"} for b in BRACKETS]

    # 10:00 EST decision = 15:00Z. 16:00Z is inside the 2h window, 18:00Z is not.
    inside = daily.snapshot(Client(), FakeFV(), datetime(2026, 10, 2, 16, 0, tzinfo=UTC))
    late = daily.snapshot(Client(), FakeFV(), datetime(2026, 10, 2, 18, 0, tzinfo=UTC))
    assert len(inside) == 3 and inside["p_market"].sum() == pytest.approx(1, abs=1e-3)
    assert late.empty
