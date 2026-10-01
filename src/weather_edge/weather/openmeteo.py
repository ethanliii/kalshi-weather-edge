"""Open-Meteo forecasts reduced to CLI-day maximum temperature (°F).

Two sources:
  * Previous Runs API: fixed-lead deterministic forecasts from several global
    models, archived since early 2024. This is the training/backtest input,
    a "poor man's ensemble" of independent models.
  * Ensemble API: GEFS and ECMWF ENS members for live forecasting. Open-Meteo only
    retains ~3 months of ensemble history, so this feed is archived going forward
    by the daily collection job.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from weather_edge.stations import Station
from weather_edge.weather.daywindow import cli_window_utc

log = logging.getLogger(__name__)

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Models with lead-1 and lead-2 archives from 2024-03 onward (probed 2026-10-01).
MULTI_MODELS = ("gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "jma_seamless")
ENSEMBLE_MODELS = ("gfs025", "ecmwf_ifs025")
HISTORY_START = date(2024, 3, 1)


def _get(url: str, params: dict, session: requests.Session | None = None, retries: int = 4) -> dict:
    s = session or requests
    for attempt in range(retries):
        try:
            r = s.get(url, params=params, timeout=60)
        except requests.RequestException as exc:
            log.warning("open-meteo network error: %s", exc)
            time.sleep(2**attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(5 * 2**attempt, 60))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"Open-Meteo HTTP {r.status_code}: {r.text[:300]}")
        return r.json()
    raise RuntimeError(f"Open-Meteo retries exhausted for {url}")


def hourly_frame(payload: dict) -> pd.DataFrame:
    """Open-Meteo `hourly` block -> DataFrame indexed by UTC timestamp."""
    hourly = dict(payload["hourly"])
    idx = pd.to_datetime(hourly.pop("time"), utc=True)
    return pd.DataFrame(hourly, index=idx).astype(float)


def cli_daily_max(hourly: pd.Series, station: Station, days: list[date]) -> pd.Series:
    """Max of hourly values within each CLI (LST) day; NaN unless all 24 hours present."""
    out = {}
    for d in days:
        start, end = cli_window_utc(d, station.std_utc_offset_h)
        window = hourly[(hourly.index >= start) & (hourly.index < end)]
        out[d] = window.max() if window.notna().sum() == 24 else np.nan
    return pd.Series(out, dtype=float)


def fetch_previous_runs(
    station: Station,
    start: date,
    end: date,
    models: tuple[str, ...] = MULTI_MODELS,
    leads: tuple[int, ...] = (1, 2),
    chunk_days: int = 180,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Daily CLI-window max per (date, lead, model) from the Previous Runs API.

    Returns long format: columns [date, lead, model, tmax_f].
    """
    variables = ",".join(f"temperature_2m_previous_day{lead}" for lead in leads)
    frames = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end)
        params = {
            "latitude": station.lat,
            "longitude": station.lon,
            "hourly": variables,
            "models": ",".join(models),
            "temperature_unit": "fahrenheit",
            "timezone": "GMT",
            "start_date": chunk_start.isoformat(),
            # CLI windows west of UTC spill into the next UTC day.
            "end_date": (chunk_end + timedelta(days=1)).isoformat(),
        }
        hourly = hourly_frame(_get(PREVIOUS_RUNS_URL, params, session))
        days = [chunk_start + timedelta(days=i) for i in range((chunk_end - chunk_start).days + 1)]
        for lead in leads:
            for model in models:
                col = f"temperature_2m_previous_day{lead}"
                if len(models) > 1:
                    col = f"{col}_{model}"
                if col not in hourly:
                    log.warning("missing column %s for %s", col, station.series)
                    continue
                daily = cli_daily_max(hourly[col], station, days)
                frames.append(pd.DataFrame({"date": daily.index, "lead": lead, "model": model,
                                            "tmax_f": daily.values}))
        chunk_start = chunk_end + timedelta(days=1)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def fetch_decision_day_run(
    station: Station,
    run_day: date,
    model: str = "ecmwf_ifs",
    session: requests.Session | None = None,
) -> dict[int, float]:
    """CLI-day max from the 00Z run of `run_day`, for lead 1 (D = run_day) and lead 2 (D + 1).

    Unlike the fixed-lead Previous Runs fields, this is one specific run, published
    by ~06-08Z — before the 10:00 LST decision time (15-18Z) in every US zone — so
    it is the freshest model information a trader could actually have used.
    ECMWF IFS HRES single runs are archived from 2024-03-14.
    """
    params = {
        "latitude": station.lat,
        "longitude": station.lon,
        "hourly": "temperature_2m",
        "models": model,
        "run": f"{run_day.isoformat()}T00:00",
        "forecast_days": 3,
        "temperature_unit": "fahrenheit",
        "timezone": "GMT",
    }
    hourly = hourly_frame(_get(SINGLE_RUNS_URL, params, session))["temperature_2m"]
    days = [run_day, run_day + timedelta(days=1)]
    daily = cli_daily_max(hourly, station, days)
    return {1: float(daily[days[0]]), 2: float(daily[days[1]])}


def parse_ensemble_column(col: str) -> tuple[str, int]:
    """'temperature_2m_member07_ncep_gefs025' -> ('ncep_gefs025', 7); control run is member 0."""
    rest = col.removeprefix("temperature_2m_")
    if rest.startswith("member"):
        return rest[9:], int(rest[6:8])
    return rest, 0


def fetch_ensemble(
    station: Station,
    forecast_days: int = 3,
    models: tuple[str, ...] = ENSEMBLE_MODELS,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Live ensemble members' CLI-day max. Columns [date, model, member, tmax_f]."""
    params = {
        "latitude": station.lat,
        "longitude": station.lon,
        "hourly": "temperature_2m",
        "models": ",".join(models),
        "temperature_unit": "fahrenheit",
        "timezone": "GMT",
        "forecast_days": forecast_days + 1,
    }
    hourly = hourly_frame(_get(ENSEMBLE_URL, params, session))
    today_utc = hourly.index[0].date()
    days = [today_utc + timedelta(days=i) for i in range(forecast_days)]
    rows = []
    for col in hourly.columns:
        model, member = parse_ensemble_column(col)
        daily = cli_daily_max(hourly[col], station, days)
        rows += [{"date": d, "model": model, "member": member, "tmax_f": v}
                 for d, v in daily.items()]
    return pd.DataFrame(rows).dropna()
