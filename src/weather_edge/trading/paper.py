"""Paper-trading loop on the Kalshi DEMO exchange.

One `run_once()` call:
  1. stops (and cancels resting orders) if the kill switch is tripped;
  2. syncs new fills into the ledger and updates the daily P&L check;
  3. for every open event whose decision time has passed, computes EMOS fair
     values, reads the demo order book and sends IOC limit orders where the
     edge after fees exceeds the margin, sized by fractional Kelly and capped
     per market, per event and in total (and by displayed size).

Every decision, order, response, fill and error goes to the JSONL ledger.
Demo-exchange prices and liquidity are not representative of production
(Kalshi says so); this module tests the plumbing and risk controls, not alpha.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime

from weather_edge.kalshi.markets import event_date
from weather_edge.live import FairValueService, lead_for
from weather_edge.stations import STATIONS
from weather_edge.trading.demo_client import DemoClient
from weather_edge.trading.fees import order_fee
from weather_edge.trading.risk import KillSwitch, Ledger
from weather_edge.trading.sizing import SizingConfig, best_opportunity, size_order

log = logging.getLogger(__name__)


def _event_of(ticker: str) -> str:
    return ticker.rsplit("-", 1)[0]


def top_of_book(book: dict) -> tuple[float | None, float, float | None, float]:
    """(yes_bid, size, yes_ask, size). A NO bid at q is a YES offer at 1 - q."""
    bid, bid_sz = book["yes"][0] if book["yes"] else (None, 0.0)
    no_bid, no_sz = book["no"][0] if book["no"] else (None, 0.0)
    ask = round(1 - no_bid, 4) if no_bid is not None else None
    return bid, bid_sz, ask, no_sz


class PaperTrader:
    def __init__(
        self,
        client: DemoClient,
        ledger: Ledger,
        kill: KillSwitch,
        sizing: SizingConfig | None = None,
        fair_values: FairValueService | None = None,
        series: list[str] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.client = client
        self.ledger = ledger
        self.kill = kill
        self.sizing = sizing or SizingConfig()
        self.fv = fair_values or FairValueService()
        self.series = series or sorted(STATIONS)
        self.now = now

    # ----------------------------------------------------------------- helpers
    def _call(self, fn, *args, **kwargs):
        try:
            out = fn(*args, **kwargs)
        except Exception as exc:
            self.kill.record_error(exc)
            self.ledger.log("error", call=getattr(fn, "__name__", str(fn)), error=str(exc))
            raise
        self.kill.record_success()
        return out

    def cancel_all(self) -> None:
        for o in self._call(self.client.resting_orders):
            r = self._call(self.client.cancel_order, o["order_id"], o.get("ticker"))
            self.ledger.log("cancel", order_id=o["order_id"], ticker=o.get("ticker"), response=r)

    def sync_fills(self) -> list[dict]:
        seen = {r["fill"]["fill_id"] for r in self.ledger.records("fill")}
        new = [f for f in self._call(self.client.fills) if f.get("fill_id") not in seen]
        for f in new:
            self.ledger.log("fill", fill=f)
        return new

    def equity(self) -> float:
        r = self._call(self.client.request, "GET", "/portfolio/balance")
        cash = float(r["balance_dollars"]) if "balance_dollars" in r else r["balance"] / 100
        return cash + float(r.get("portfolio_value", 0)) / 100

    def day_start_equity(self, equity: float) -> float:
        today = self.now().date().isoformat()
        for r in reversed(self.ledger.records("day_start")):
            if r["day"] == today:
                return r["equity"]
        self.ledger.log("day_start", day=today, equity=equity)
        return equity

    # ----------------------------------------------------------------- main loop
    def run_once(self) -> list[dict]:
        if self.kill.tripped:
            self.ledger.log("halted", reason=self.kill.reason())
            self.cancel_all()
            return []
        self.sync_fills()
        equity = self.equity()
        self.kill.check_pnl(equity - self.day_start_equity(equity))
        if self.kill.tripped:
            self.cancel_all()
            return []

        bankroll = self._call(self.client.balance)
        exposure = defaultdict(float)  # per ticker, dollars at cost
        for p in self._call(self.client.positions):
            exposure[p["ticker"]] += abs(float(p.get("market_exposure_dollars") or 0))
        event_exp = defaultdict(float)
        for t, v in exposure.items():
            event_exp[_event_of(t)] += v
        total = sum(exposure.values())

        orders = []
        for s in self.series:
            station = STATIONS[s]
            by_event = defaultdict(list)
            for m in self._call(self.client.open_markets, s):
                by_event[m["event_ticker"]].append(m)
            for ev, markets in sorted(by_event.items()):
                day = event_date(ev)
                lead = lead_for(station, day, self.now())
                if lead is None:
                    continue
                try:
                    fv = self.fv.fair_value(station, ev, day, lead, markets)
                except Exception as exc:  # bad partition / missing forecast: skip, don't halt
                    self.ledger.log("skip_event", event=ev, reason=str(exc))
                    continue
                self.ledger.log("fair_value", event=ev, lead=lead, mu=fv.mu, sigma=fv.sigma,
                                brackets=[b.ticker for b in fv.brackets], p_emos=fv.p_emos)
                for bracket, p in zip(fv.brackets, fv.p_emos, strict=True):
                    if self.kill.tripped:
                        return orders
                    bid, bid_sz, ask, ask_sz = top_of_book(
                        self._call(self.client.orderbook, bracket.ticker))
                    opp = best_opportunity(p, bid, ask, self.sizing)
                    if opp is None:
                        continue
                    n = size_order(opp, bankroll, self.sizing, exposure[bracket.ticker],
                                   event_exp[ev], total)
                    n = min(n, int(ask_sz if opp.side == "yes" else bid_sz))
                    if n <= 0:
                        continue
                    self.ledger.log("decision", ticker=bracket.ticker, side=opp.side,
                                    price=opp.price, prob=opp.prob, edge=opp.edge, count=n,
                                    yes_bid=bid, yes_ask=ask)
                    result = self._call(self.client.place_limit_order, bracket.ticker,
                                        opp.side, opp.price, n)
                    self.ledger.log("order", ticker=bracket.ticker, **result)
                    spent = n * opp.price + order_fee(opp.price, n)
                    exposure[bracket.ticker] += spent
                    event_exp[ev] += spent
                    total += spent
                    orders.append(result)
        return orders
