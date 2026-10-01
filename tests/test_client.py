from datetime import UTC, datetime

import pytest

from weather_edge.kalshi.client import KalshiAPIError, KalshiClient, normalise_candle


class FakeResponse:
    def __init__(self, status, payload, url="u"):
        self.status_code, self._payload, self.url = status, payload, url
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def client(responses):
    return KalshiClient(session=FakeSession(responses), min_interval_s=0)


def test_paginate_follows_cursor():
    c = client([FakeResponse(200, {"markets": [{"t": 1}], "cursor": "abc"}),
                FakeResponse(200, {"markets": [{"t": 2}], "cursor": ""})])
    assert [m["t"] for m in c.paginate("/markets", "markets")] == [1, 2]
    assert c.session.calls[1][1]["cursor"] == "abc"


def test_retries_on_429(monkeypatch):
    monkeypatch.setattr("weather_edge.kalshi.client.time.sleep", lambda s: None)
    c = client([FakeResponse(429, {}), FakeResponse(200, {"ok": True})])
    assert c.get("/x") == {"ok": True}


def test_raises_on_404():
    with pytest.raises(KalshiAPIError):
        client([FakeResponse(404, {"error": "nf"})]).get("/x")


def test_orderbook_sorted_best_first():
    c = client([FakeResponse(200, {"orderbook_fp": {
        "yes_dollars": [["0.42", "10"], ["0.46", "5"]], "no_dollars": [["0.49", "3"], ["0.53", "2"]]}})])
    book = c.get_orderbook("T")
    assert book["yes"][0] == (0.46, 5.0)
    assert book["no"][0] == (0.53, 2.0)  # best YES ask = 1 - 0.53 = 0.47


def test_normalise_candle_handles_both_schemas():
    new = {"end_period_ts": 1789470000, "yes_bid": {"close_dollars": "0.55"},
           "yes_ask": {"close_dollars": "0.56"}, "price": {"close_dollars": None},
           "volume_fp": "595.90", "open_interest_fp": "4302.18"}
    old = {"end_period_ts": 1689451200, "yes_bid": {"close": "0.16"}, "yes_ask": {"close": "0.99"},
           "price": {"close": None}, "volume": "0.00", "open_interest": "333.00"}
    n, o = normalise_candle(new), normalise_candle(old)
    assert (n["yes_bid_close"], n["yes_ask_close"], n["price_close"]) == (0.55, 0.56, None)
    assert (o["yes_bid_close"], o["yes_ask_close"], o["volume"]) == (0.16, 0.99, 0.0)
    assert n["end_ts"] == datetime.fromtimestamp(1789470000, UTC)


def test_candles_route_to_historical_before_cutoff():
    c = client([FakeResponse(200, {"market_settled_ts": "2026-08-02T00:00:00Z"}),
                FakeResponse(200, {"candlesticks": []}),
                FakeResponse(200, {"candlesticks": []})])
    t0, t1 = datetime(2025, 7, 15, tzinfo=UTC), datetime(2025, 7, 16, tzinfo=UTC)
    c.candlesticks("KXHIGHNY", "M", t0, t1, settled_at=datetime(2025, 7, 16, tzinfo=UTC))
    c.candlesticks("KXHIGHNY", "M", t0, t1, settled_at=datetime(2026, 9, 16, tzinfo=UTC))
    assert c.session.calls[1][0].endswith("/historical/markets/M/candlesticks")
    assert c.session.calls[2][0].endswith("/series/KXHIGHNY/markets/M/candlesticks")
