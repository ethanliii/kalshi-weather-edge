"""Edge detection and fractional-Kelly position sizing for binary contracts.

A YES contract bought at all-in cost c (price + fee per contract) pays 1 with
probability p. The Kelly-optimal fraction of bankroll for this single bet is

    f* = (p - c) / (1 - c)

Buying NO is the same bet on 1 - p at cost (1 - yes_bid) + fee. Full Kelly is
very sensitive to errors in p, so we bet a fraction (default 1/4) and cap
exposure per market, per event and in total. Brackets of one event are mutually
exclusive, so per-event caps stop the sizer from stacking correlated bets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from weather_edge.trading.fees import fee_per_contract

Side = Literal["yes", "no"]


@dataclass(frozen=True)
class SizingConfig:
    kelly_fraction: float = 0.25
    min_edge: float = 0.03  # required edge after fees, in dollars per contract
    max_contracts_per_order: int = 100
    max_market_exposure: float = 25.0  # dollars at risk per market
    max_event_exposure: float = 50.0  # dollars at risk per event (all brackets of a day/city)
    max_total_exposure: float = 250.0  # dollars at risk across everything
    fee_type: str = "quadratic"
    maker: bool = False  # True when resting a limit order that does not cross the spread


@dataclass(frozen=True)
class Opportunity:
    side: Side
    price: float  # limit price paid per contract for the chosen side, in dollars
    prob: float  # model probability that the chosen side wins
    edge: float  # prob - price - fee per contract


def kelly_fraction(prob: float, cost: float) -> float:
    """Full-Kelly bankroll fraction for paying `cost` to win 1 with probability `prob`."""
    if not 0 < cost < 1:
        return 0.0
    return max(0.0, (prob - cost) / (1 - cost))


def best_opportunity(
    p_yes: float,
    yes_bid: float | None,
    yes_ask: float | None,
    cfg: SizingConfig,
    ref_count: int = 10,
) -> Opportunity | None:
    """Return the better of buy-YES-at-ask / buy-NO-at-(1-bid) if its edge clears the margin.

    Fees depend on order size through the per-order cent round-up; `ref_count`
    is the size used to evaluate the fee when screening.
    """
    cands = []
    if yes_ask is not None and 0 < yes_ask < 1:
        fee = fee_per_contract(yes_ask, ref_count, maker=cfg.maker, fee_type=cfg.fee_type)
        cands.append(Opportunity("yes", yes_ask, p_yes, p_yes - yes_ask - fee))
    if yes_bid is not None and 0 < yes_bid < 1:
        no_price = round(1 - yes_bid, 4)
        fee = fee_per_contract(no_price, ref_count, maker=cfg.maker, fee_type=cfg.fee_type)
        cands.append(Opportunity("no", no_price, 1 - p_yes, (1 - p_yes) - no_price - fee))
    if not cands:
        return None
    best = max(cands, key=lambda o: o.edge)
    return best if best.edge > cfg.min_edge else None


def size_order(
    opp: Opportunity,
    bankroll: float,
    cfg: SizingConfig,
    market_exposure: float = 0.0,
    event_exposure: float = 0.0,
    total_exposure: float = 0.0,
) -> int:
    """Number of contracts to buy, after fractional Kelly and every exposure cap."""
    if bankroll <= 0:
        return 0
    count = cfg.max_contracts_per_order
    # Kelly with the fee at the provisional size, then re-check the fee at the final size.
    for _ in range(3):
        cost = opp.price + fee_per_contract(opp.price, max(count, 1), maker=cfg.maker,
                                            fee_type=cfg.fee_type)
        f = cfg.kelly_fraction * kelly_fraction(opp.prob, cost)
        room = min(
            f * bankroll,
            cfg.max_market_exposure - market_exposure,
            cfg.max_event_exposure - event_exposure,
            cfg.max_total_exposure - total_exposure,
        )
        new_count = max(0, min(cfg.max_contracts_per_order, int(room // cost)))
        if new_count == count:
            break
        count = new_count
    if count == 0:
        return 0
    final_edge = opp.prob - opp.price - fee_per_contract(opp.price, count, maker=cfg.maker,
                                                         fee_type=cfg.fee_type)
    return count if final_edge > cfg.min_edge else 0
