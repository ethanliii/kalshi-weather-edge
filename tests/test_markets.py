import math
from datetime import date

import pytest

from weather_edge.kalshi.markets import (
    Bracket,
    bracket_from_market,
    event_date,
    settlement_source,
    sorted_brackets,
    validate_partition,
    winning_bracket,
)

# Real field values from KXHIGHNY-26OCT02 (fetched 2026-10-01).
NY_OCT02 = [
    {"ticker": "KXHIGHNY-26OCT02-T81", "strike_type": "less", "cap_strike": 81},
    {"ticker": "KXHIGHNY-26OCT02-B81.5", "strike_type": "between", "floor_strike": 81, "cap_strike": 82},
    {"ticker": "KXHIGHNY-26OCT02-B83.5", "strike_type": "between", "floor_strike": 83, "cap_strike": 84},
    {"ticker": "KXHIGHNY-26OCT02-B85.5", "strike_type": "between", "floor_strike": 85, "cap_strike": 86},
    {"ticker": "KXHIGHNY-26OCT02-B87.5", "strike_type": "between", "floor_strike": 87, "cap_strike": 88},
    {"ticker": "KXHIGHNY-26OCT02-T88", "strike_type": "greater", "floor_strike": 88},
]


def test_bracket_bounds_match_subtitles():
    less, b81, *_, greater = sorted_brackets(NY_OCT02)
    assert (less.lo, less.hi) == (-math.inf, 80)  # "80° or below"
    assert (b81.lo, b81.hi) == (81, 82)  # "81° to 82°"
    assert (greater.lo, greater.hi) == (89, math.inf)  # "89° or above"


def test_real_event_is_a_partition():
    validate_partition(sorted_brackets(NY_OCT02))


@pytest.mark.parametrize("temp,expected", [(60, "T81"), (80, "T81"), (81, "B81.5"), (84, "B83.5"),
                                           (88, "B87.5"), (89, "T88"), (120, "T88")])
def test_exactly_one_winner(temp, expected):
    assert winning_bracket(sorted_brackets(NY_OCT02), temp).ticker.endswith(expected)


def test_gap_is_rejected():
    bs = sorted_brackets([m for m in NY_OCT02 if not m["ticker"].endswith("B83.5")])
    with pytest.raises(ValueError, match="gap"):
        validate_partition(bs)


def test_missing_tail_is_rejected():
    with pytest.raises(ValueError, match="tails"):
        validate_partition(sorted_brackets(NY_OCT02[1:]))


def test_unknown_strike_type():
    with pytest.raises(ValueError):
        bracket_from_market({"ticker": "X", "strike_type": "custom"})


def test_event_date_parsing():
    assert event_date("KXHIGHNY-26OCT02") == date(2026, 10, 2)
    assert event_date("HIGHNY-21AUG06") == date(2021, 8, 6)
    with pytest.raises(ValueError):
        event_date("KXHIGHNY")


def test_settlement_source_classification():
    assert settlement_source("... according to The Weather Company, then ...") == "TWC"
    assert settlement_source("as reported by the National Weather Service's Climatological "
                             "Report (Daily), is greater than") == "NWS_CLI"


def test_bracket_label():
    assert Bracket("x", -math.inf, 80).label == "≤80"
    assert Bracket("x", 89, math.inf).label == "≥89"
    assert Bracket("x", 81, 82).label == "81-82"
