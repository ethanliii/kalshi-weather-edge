from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from weather_edge.stations import STATIONS
from weather_edge.weather.daywindow import (
    PUBLICATION_LAG,
    cli_window_utc,
    decision_time,
    latest_input_init,
)
from weather_edge.weather.openmeteo import cli_daily_max, parse_ensemble_column

NYC = STATIONS["KXHIGHNY"]
LAX = STATIONS["KXHIGHLAX"]


def test_cli_window_is_standard_time_even_in_summer():
    # July: EDT is UTC-4 but the CLI day is EST midnight-midnight = 05Z-05Z.
    start, end = cli_window_utc(date(2026, 7, 4), NYC.std_utc_offset_h)
    assert start == datetime(2026, 7, 4, 5, tzinfo=UTC)
    assert end == datetime(2026, 7, 5, 5, tzinfo=UTC)
    # Same UTC window in January (no DST) — the window never shifts.
    start_w, _ = cli_window_utc(date(2026, 1, 4), NYC.std_utc_offset_h)
    assert start_w.hour == 5


def test_west_coast_window():
    start, end = cli_window_utc(date(2026, 7, 4), LAX.std_utc_offset_h)
    assert (start.hour, end - start) == (8, timedelta(days=1))


@pytest.mark.parametrize("station", list(STATIONS.values()))
@pytest.mark.parametrize("lead", [1, 2])
def test_decision_time_never_sees_unpublished_runs(station, lead):
    d = date(2026, 7, 15)
    newest_init = latest_input_init(d, station.std_utc_offset_h, lead)
    assert newest_init + PUBLICATION_LAG < decision_time(d, station.std_utc_offset_h, lead)


def test_decision_time_precedes_measurement_day_end():
    d = date(2026, 7, 15)
    start, _ = cli_window_utc(d, NYC.std_utc_offset_h)
    assert decision_time(d, NYC.std_utc_offset_h, 2) < start
    assert decision_time(d, NYC.std_utc_offset_h, 1) == start + timedelta(hours=10)


def _hourly(day, values, offset_h=-5):
    start, _ = cli_window_utc(day, offset_h)
    idx = pd.date_range(start - timedelta(hours=3), periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_daily_max_respects_window_edges():
    d = date(2026, 7, 4)
    vals = np.full(30, 70.0)
    vals[0:3] = 99.0  # 3 hours before the window opens — must be ignored
    vals[3 + 24:] = 98.0  # after the window closes — must be ignored
    vals[3 + 15] = 85.0  # inside
    out = cli_daily_max(_hourly(d, vals), NYC, [d])
    assert out[d] == 85.0


def test_daily_max_requires_full_coverage():
    d = date(2026, 7, 4)
    vals = np.full(30, 70.0)
    vals[10] = np.nan
    assert np.isnan(cli_daily_max(_hourly(d, vals), NYC, [d])[d])


def test_parse_ensemble_column():
    assert parse_ensemble_column("temperature_2m_member07_ncep_gefs025") == ("ncep_gefs025", 7)
    assert parse_ensemble_column("temperature_2m_ecmwf_ifs025_ensemble") == ("ecmwf_ifs025_ensemble", 0)
