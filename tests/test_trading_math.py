import pytest

from weather_edge.trading.fees import fee_per_contract, order_fee
from weather_edge.trading.sizing import (
    Opportunity,
    SizingConfig,
    best_opportunity,
    kelly_fraction,
    size_order,
)


# ------------------------------------------------------------------- fees
def test_fee_schedule_examples():
    # 0.07 * 1 * 0.5 * 0.5 = 0.0175 -> rounds up to 2 cents
    assert order_fee(0.50, 1) == 0.02
    # 0.07 * 100 * 0.5 * 0.5 = 1.75 exactly
    assert order_fee(0.50, 100) == 1.75
    # 0.07 * 10 * 0.1 * 0.9 = 0.063 -> 0.07
    assert order_fee(0.10, 10) == 0.07


def test_fee_float_noise_does_not_add_a_cent():
    # 0.07 * 20 * 0.5 * 0.5 = 0.35 exactly, despite binary floating point
    assert order_fee(0.50, 20) == 0.35


def test_fee_symmetric_in_price():
    assert order_fee(0.23, 37) == order_fee(0.77, 37)


def test_maker_fee_zero_on_standard_series():
    assert order_fee(0.5, 100, maker=True) == 0.0
    assert order_fee(0.5, 100, maker=True, fee_type="quadratic_with_maker_fees") == pytest.approx(0.44)


def test_fee_validation():
    with pytest.raises(ValueError):
        order_fee(1.0, 1)
    assert order_fee(0.5, 0) == 0.0
    assert fee_per_contract(0.5, 100) == pytest.approx(0.0175)


# ------------------------------------------------------------------ kelly
def test_kelly_formula():
    assert kelly_fraction(0.6, 0.5) == pytest.approx(0.2)
    assert kelly_fraction(0.4, 0.5) == 0.0  # no bet without edge
    assert kelly_fraction(0.9, 1.0) == 0.0


# ------------------------------------------------------------- edge screen
CFG = SizingConfig(min_edge=0.03)


def test_buys_yes_when_model_above_ask():
    opp = best_opportunity(0.60, yes_bid=0.48, yes_ask=0.50, cfg=CFG)
    assert opp.side == "yes" and opp.price == 0.50
    assert opp.edge == pytest.approx(0.60 - 0.50 - 0.018)  # 10-contract fee 0.175 -> 0.18


def test_buys_no_when_model_below_bid():
    opp = best_opportunity(0.20, yes_bid=0.35, yes_ask=0.37, cfg=CFG)
    assert opp.side == "no" and opp.price == pytest.approx(0.65)
    assert opp.prob == pytest.approx(0.80)


def test_no_trade_inside_spread_or_below_margin():
    assert best_opportunity(0.50, 0.48, 0.52, CFG) is None
    assert best_opportunity(0.53, 0.40, 0.50, CFG) is None  # 3c gross edge < fee + margin


def test_missing_quotes():
    assert best_opportunity(0.9, None, None, CFG) is None
    assert best_opportunity(0.9, 0.0, 1.0, CFG) is None


# ------------------------------------------------------------------ sizing
def test_quarter_kelly_size():
    cfg = SizingConfig(kelly_fraction=0.25, max_market_exposure=1e9, max_event_exposure=1e9,
                       max_total_exposure=1e9, max_contracts_per_order=10_000)
    opp = Opportunity("yes", 0.50, 0.60, 0.08)
    n = size_order(opp, bankroll=1000, cfg=cfg)
    cost = 0.50 + 0.0175
    expected = int(0.25 * (0.60 - cost) / (1 - cost) * 1000 // cost)
    assert abs(n - expected) <= 1


@pytest.mark.parametrize("cap_field,cap,used_field", [
    ("max_market_exposure", 5.0, "market_exposure"),
    ("max_event_exposure", 5.0, "event_exposure"),
    ("max_total_exposure", 5.0, "total_exposure"),
])
def test_exposure_caps_bind(cap_field, cap, used_field):
    base = dict(max_market_exposure=1e9, max_event_exposure=1e9, max_total_exposure=1e9,
                max_contracts_per_order=10_000)
    base[cap_field] = cap
    cfg = SizingConfig(**base)
    opp = Opportunity("yes", 0.30, 0.60, 0.25)
    n = size_order(opp, bankroll=100_000, cfg=cfg)
    assert n * 0.30 <= cap and n > 0
    assert size_order(opp, bankroll=100_000, cfg=cfg, **{used_field: cap}) == 0


def test_never_exceeds_order_cap_or_trades_without_edge():
    cfg = SizingConfig(max_contracts_per_order=7, max_market_exposure=1e9, max_event_exposure=1e9,
                       max_total_exposure=1e9)
    assert size_order(Opportunity("yes", 0.10, 0.90, 0.78), 1e6, cfg) == 7
    assert size_order(Opportunity("yes", 0.50, 0.50, 0.0), 1e6, cfg) == 0
    assert size_order(Opportunity("yes", 0.50, 0.90, 0.38), 0, cfg) == 0


def test_small_size_fee_roundup_can_kill_the_trade():
    # 1 contract at 0.50 pays a 2c fee; edge 0.035 - 0.02 = 0.015 < 0.03 margin.
    cfg = SizingConfig(min_edge=0.03, max_market_exposure=0.6, max_event_exposure=1e9,
                       max_total_exposure=1e9)
    assert size_order(Opportunity("yes", 0.50, 0.535, 0.017), 1e6, cfg) == 0
