"""Kalshi trading fees.

From the Kalshi fee schedule (kalshi.com/fee-schedule) and API series metadata:

    taker fee = ceil_to_cent( M * 0.07   * C * P * (1 - P) )
    maker fee = ceil_to_cent( M * 0.0175 * C * P * (1 - P) )   only if fee_type is
                                                               'quadratic_with_maker_fees'

C = contracts, P = price in dollars, M = series fee multiplier (1 for every
weather series as of 2026-10-01, all with fee_type 'quadratic', so resting
maker orders pay no fee). There is no settlement fee. The fee is charged per
order, so the cent round-up matters most for small orders.
"""

from __future__ import annotations

import math

TAKER_RATE = 0.07
MAKER_RATE = 0.0175


def _ceil_cents(x: float) -> float:
    # Round to 1e-9 first so float noise (e.g. 0.07000000000000001) doesn't add a cent.
    return math.ceil(round(x * 100, 9)) / 100


def order_fee(
    price: float,
    count: int,
    *,
    maker: bool = False,
    fee_type: str = "quadratic",
    multiplier: float = 1.0,
) -> float:
    """Total fee in dollars for one order of `count` contracts at `price`."""
    if not 0 < price < 1:
        raise ValueError(f"price must be in (0, 1), got {price}")
    if count <= 0:
        return 0.0
    if maker:
        if fee_type not in ("quadratic_with_maker_fees", "quadratic_with_combo_maker_fees"):
            return 0.0
        rate = MAKER_RATE
    else:
        rate = TAKER_RATE
    return _ceil_cents(multiplier * rate * count * price * (1 - price))


def fee_per_contract(price: float, count: int, **kwargs) -> float:
    return order_fee(price, count, **kwargs) / count if count > 0 else 0.0
