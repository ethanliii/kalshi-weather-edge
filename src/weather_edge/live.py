"""Live fair values for open markets.

For an event on day D the lead is chosen so live inputs match training inputs:
  * D == today (LST)    -> lead 1, from 10:00 LST for DECISION_WINDOW
  * D == tomorrow (LST) -> lead 2, from 10:00 LST today for DECISION_WINDOW
Outside that window nothing is priced: later in the day the market has seen
observations the forecast hasn't, and the backtest never evaluated those trades.
EMOS parameters come from models/emos_params.json (written by `weather-edge fit`
on all history, same specification as the walk-forward evaluation) and are
applied to the fixed-lead forecast for D. Only that one day's forecast is fetched
at run time, which keeps the CI job light. Without the file, models are fitted
on the fly from the local data cache.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from weather_edge import dataset
from weather_edge.config import ROOT
from weather_edge.kalshi.markets import Bracket, sorted_brackets, validate_partition
from weather_edge.model.brackets import probs_members, probs_normal
from weather_edge.model.emos import GaussianEMOS
from weather_edge.stations import Station
from weather_edge.weather import openmeteo
from weather_edge.weather.daywindow import decision_time

log = logging.getLogger(__name__)
DECISION_WINDOW = timedelta(hours=2)
PARAMS_PATH = ROOT / "models" / "emos_params.json"


def lst_today(station: Station, now: datetime | None = None) -> date:
    now = now or datetime.now(UTC)
    return (now + timedelta(hours=station.std_utc_offset_h)).date()


def lead_for(
    station: Station, day: date, now: datetime | None = None,
    window: timedelta = DECISION_WINDOW,
) -> int | None:
    """Lead for `day` if `now` is inside [decision time, decision time + window), else None."""
    now = now or datetime.now(UTC)
    lead = (day - lst_today(station, now)).days + 1
    if lead not in (1, 2):
        return None
    start = decision_time(day, station.std_utc_offset_h, lead)
    return lead if start <= now < start + window else None


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


def fit_all(series: list[str], leads: tuple[int, ...] = (1, 2)) -> dict:
    """Fit EMOS on all available history per (series, lead); JSON-serialisable."""
    out: dict = {"fitted_utc": datetime.now(UTC).isoformat(timespec="seconds"), "models": {}}
    for s in series:
        for lead in leads:
            train = dataset.forecast_table(s, lead).dropna(subset=[*dataset.MEMBERS, "obs"])
            model = GaussianEMOS(dataset.MEMBERS).fit(train)
            out["models"][f"{s}|{lead}"] = {
                **model.to_dict(), "n_train": int(len(train)),
                "train_start": str(train["date"].min()), "train_end": str(train["date"].max()),
            }
            log.info("fitted %s lead %d on %d days", s, lead, len(train))
    return out


def save_params(params: dict, path=PARAMS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=1))


class FairValueService:
    """EMOS per (series, lead): loaded from the params file, else fitted once per process."""

    def __init__(self, params_path=PARAMS_PATH) -> None:
        self._models: dict[tuple[str, int], GaussianEMOS] = {}
        if params_path is not None and params_path.exists():
            saved = json.loads(params_path.read_text())["models"]
            for key, d in saved.items():
                s, lead = key.split("|")
                self._models[(s, int(lead))] = GaussianEMOS.from_dict(d)

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
