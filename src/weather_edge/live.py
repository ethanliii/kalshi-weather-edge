"""Live fair values for open markets.

For an event on day D the lead is chosen so live inputs match training inputs:
  * D == today (LST)    -> lead 1, valid once 10:00 LST has passed
  * D == tomorrow (LST) -> lead 2, valid once 10:00 LST today has passed
EMOS is refit on all history available at run time (same specification as the
walk-forward evaluation), then applied to the fixed-lead forecast for D.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from weather_edge import dataset
from weather_edge.kalshi.markets import Bracket, sorted_brackets, validate_partition
from weather_edge.model.brackets import probs_members, probs_normal
from weather_edge.model.emos import GaussianEMOS
from weather_edge.stations import Station
from weather_edge.weather import openmeteo
from weather_edge.weather.daywindow import decision_time

log = logging.getLogger(__name__)


def lst_today(station: Station, now: datetime | None = None) -> date:
    now = now or datetime.now(UTC)
    return (now + timedelta(hours=station.std_utc_offset_h)).date()


def lead_for(station: Station, day: date, now: datetime | None = None) -> int | None:
    """Lead whose decision time has passed for `day`, or None if not tradeable by our rules."""
    now = now or datetime.now(UTC)
    lead = (day - lst_today(station, now)).days + 1
    if lead not in (1, 2):
        return None
    return lead if now >= decision_time(day, station.std_utc_offset_h, lead) else None


@dataclass
class FairValue:
    series: str
    event_ticker: str
    day: date
    lead: int
    mu: float
    sigma: float
    brackets: list[Bracket]
    p_emos: list[float]
    p_raw: list[float]


class FairValueService:
    """Caches one fitted EMOS per (series, lead) for the life of the process."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, int], GaussianEMOS] = {}

    def model(self, series: str, lead: int) -> GaussianEMOS:
        key = (series, lead)
        if key not in self._models:
            train = dataset.forecast_table(series, lead).dropna(subset=["obs"])
            self._models[key] = GaussianEMOS(dataset.MEMBERS).fit(train)
        return self._models[key]

    def forecast_row(self, station: Station, day: date, lead: int) -> pd.DataFrame:
        fc = openmeteo.fetch_previous_runs(station, day, day, leads=(lead,))
        wide = fc.pivot_table(index="date", columns="model", values="tmax_f").reset_index()
        missing = [m for m in dataset.MEMBERS if m not in wide or wide[m].isna().any()]
        if missing:
            raise ValueError(f"{station.series} {day}: missing forecasts for {missing}")
        return wide

    def fair_value(self, station: Station, event_ticker: str, day: date, lead: int,
                   markets: list[dict]) -> FairValue:
        brackets = sorted_brackets(markets)
        validate_partition(brackets)
        row = self.forecast_row(station, day, lead)
        mu, sigma = self.model(station.series, lead).predict(row)
        members = row[dataset.MEMBERS].to_numpy(dtype=float)[0]
        return FairValue(
            series=station.series, event_ticker=event_ticker, day=day, lead=lead,
            mu=float(mu[0]), sigma=float(sigma[0]), brackets=brackets,
            p_emos=probs_normal(mu[0], sigma[0], brackets).tolist(),
            p_raw=probs_members(members, brackets).tolist(),
        )
