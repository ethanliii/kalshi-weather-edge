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


def test_decision_day_run_assigns_days_to_leads():
    from weather_edge.weather import openmeteo

    run_day = date(2026, 7, 4)
    idx = pd.date_range("2026-07-04T00:00", periods=72, freq="h", tz="UTC")
    vals = np.full(72, 70.0)
    d1_start, _ = cli_window_utc(run_day, NYC.std_utc_offset_h)
    d2_start, _ = cli_window_utc(run_day + timedelta(days=1), NYC.std_utc_offset_h)
    vals[idx.get_loc(d1_start + timedelta(hours=15))] = 88.0  # day-1 peak
    vals[idx.get_loc(d2_start + timedelta(hours=15))] = 91.0  # day-2 peak

    class Session:
        def get(self, url, params=None, timeout=None):
            assert params["run"] == "2026-07-04T00:00" and url == openmeteo.SINGLE_RUNS_URL

            class R:
                status_code = 200

                def json(self):
                    return {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in idx],
                                       "temperature_2m": vals.tolist()}}
            return R()

    assert openmeteo.fetch_decision_day_run(NYC, run_day, session=Session()) == {1: 88.0, 2: 91.0}
