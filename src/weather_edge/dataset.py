"""Assemble modelling and evaluation tables from the raw cache.

  forecast_table(series, lead)   date x model forecasts + observed CLI high
  emos_predictions(lead)          walk-forward EMOS mu/sigma for every station
  bracket_table(lead)             one row per settled market with model, raw-ensemble
                                  and market-implied probabilities and the outcome
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from weather_edge.collect import RAW_DIR
from weather_edge.config import PROCESSED_DIR
from weather_edge.kalshi.markets import Bracket
from weather_edge.model.brackets import probs_members, probs_normal
from weather_edge.model.emos import walk_forward
from weather_edge.stations import STATIONS
from weather_edge.weather.openmeteo import MULTI_MODELS

log = logging.getLogger(__name__)
MEMBERS = list(MULTI_MODELS)


def forecast_table(series: str, lead: int) -> pd.DataFrame:
    st = STATIONS[series]
    fc = pd.read_parquet(RAW_DIR / "forecasts" / f"{series}.parquet")
    fc = fc[fc["lead"] == lead]
    wide = fc.pivot_table(index="date", columns="model", values="tmax_f").reset_index()
    cli = pd.read_parquet(RAW_DIR / "cli" / f"{st.icao}.parquet")
    df = wide.merge(cli[["date", "cli_high_f"]], on="date", how="left")
    df = df.rename(columns={"cli_high_f": "obs"})
    df.insert(0, "series", series)
    return df


def emos_predictions(lead: int, series: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for s in series or sorted(STATIONS):
        df = forecast_table(s, lead)
        preds = walk_forward(df, MEMBERS)
        frames.append(preds)
        log.info("EMOS %s lead %d: %d out-of-sample days", s, lead, len(preds))
    out = pd.concat(frames, ignore_index=True)
    out["lead"] = lead
    return out


def market_implied(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """Mid prices normalised to sum to one across an event's brackets.

    Raw mids typically sum to slightly more than 1 (the market's overround);
    normalising gives a proper probability vector comparable with the model.
    """
    mid = (bid + ask) / 2
    return mid / mid.sum()


def _event_ok(g: pd.DataFrame) -> bool:
    """Usable event: complete partition, one winner, a two-sided quote on every bracket."""
    if g["yes_bid"].isna().any() or g["yes_ask"].isna().any():
        return False
    if (g["yes_ask"] <= g["yes_bid"]).any() or (g["yes_ask"] > 1).any():
        return False
    if (g["result"] == "yes").sum() != 1:
        return False
    lo, hi = g["lo"].to_numpy(), g["hi"].to_numpy()
    if not (np.isinf(lo[0]) and np.isinf(hi[-1]) and np.all(lo[1:] == hi[:-1] + 1)):
        return False
    mids = ((g["yes_bid"] + g["yes_ask"]) / 2).sum()
    return 0.8 <= mids <= 1.3  # stale or one-sided books otherwise


def bracket_table(lead: int, preds: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per bracket of every usable settled event with an out-of-sample forecast."""
    preds = preds if preds is not None else emos_predictions(lead)
    rows = []
    drop_counts = {"no_forecast": 0, "bad_quotes": 0}
    for s in sorted(STATIONS):
        mpath, qpath = RAW_DIR / "markets" / f"{s}.parquet", RAW_DIR / "quotes" / f"{s}.parquet"
        if not (mpath.exists() and qpath.exists()):
            continue
        markets = pd.read_parquet(mpath)
        quotes = pd.read_parquet(qpath)
        quotes = quotes[quotes["lead"] == lead]
        m = markets.merge(quotes, on="ticker", how="inner")
        p = preds[preds["series"] == s].set_index("date")
        for _, g in m.groupby("event_ticker"):
            g = g.sort_values("lo").reset_index(drop=True)
            d = g["date"].iloc[0]
            if d not in p.index:
                drop_counts["no_forecast"] += 1
                continue
            if not _event_ok(g):
                drop_counts["bad_quotes"] += 1
                continue
            f = p.loc[d]
            brackets = [Bracket(t, lo, hi) for t, lo, hi in zip(g["ticker"], g["lo"], g["hi"],
                                                                strict=True)]
            g = g.assign(
                series=s,
                p_emos=probs_normal(f["mu"], f["sigma"], brackets),
                p_raw=probs_members(f[MEMBERS].to_numpy(dtype=float), brackets),
                p_market=market_implied(g["yes_bid"].to_numpy(), g["yes_ask"].to_numpy()),
                y=(g["result"] == "yes").astype(int),
                mu=f["mu"], sigma=f["sigma"], obs=f["obs"],
            )
            rows.append(g)
    log.info("lead %d: dropped events %s", lead, drop_counts)
    out = pd.concat(rows, ignore_index=True)
    out.attrs["dropped"] = drop_counts
    return out


def save(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
