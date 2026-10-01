"""Daily forward-test job (run by GitHub Actions, read-only against production).

Writes compact, committed results under live/:
  live/snapshots/YYYY-MM-DD.csv   fair values vs production quotes for open events
  live/ensemble/YYYY-MM-DD.csv.gz GEFS + ECMWF ENS member highs (Open-Meteo keeps only
                                  ~3 months, so we archive them for future evaluation)
  live/forward_scores.csv         per-event scores once events settle

The job runs hourly from 15:20 to 18:20 UTC (10:00 LST is 15Z in the East, 18Z on
the West Coast). An event is snapshotted only within live.DECISION_WINDOW after its
decision time, so the forward test sees the same information set as the
backtest instead of a late-day market that already knows the observed high.
No orders are placed here; paper trading is a separate, demo-only command.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from weather_edge import collect
from weather_edge.config import ROOT
from weather_edge.dataset import market_implied
from weather_edge.evaluation.metrics import EPS
from weather_edge.kalshi.client import KalshiClient
from weather_edge.kalshi.markets import event_date
from weather_edge.live import FairValueService, lead_for
from weather_edge.stations import STATIONS

log = logging.getLogger(__name__)
LIVE_DIR = ROOT / "live"


def snapshot(client: KalshiClient, fv: FairValueService, now: datetime | None = None
             ) -> pd.DataFrame:
    now = now or datetime.now(UTC)
    rows = []
    for s, station in sorted(STATIONS.items()):
        by_event: dict[str, list[dict]] = defaultdict(list)
        for m in client.open_markets(s):
            by_event[m["event_ticker"]].append(m)
        for ev, markets in sorted(by_event.items()):
            day = event_date(ev)
            lead = lead_for(station, day, now)
            if lead is None:
                continue
            try:
                f = fv.fair_value(station, ev, day, lead, markets)
            except Exception as exc:
                log.warning("skip %s: %s", ev, exc)
                continue
            quotes = {m["ticker"]: m for m in markets}
            bid = np.array([float(quotes[b.ticker]["yes_bid_dollars"]) for b in f.brackets])
            ask = np.array([float(quotes[b.ticker]["yes_ask_dollars"]) for b in f.brackets])
            p_mkt = market_implied(bid, ask)
            for i, b in enumerate(f.brackets):
                rows.append({
                    "run_utc": now.isoformat(timespec="minutes"), "series": s,
                    "event_ticker": ev, "date": day.isoformat(), "lead": lead,
                    "ticker": b.ticker, "lo": b.lo, "hi": b.hi,
                    "mu": round(f.mu, 2), "sigma": round(f.sigma, 2),
                    "p_emos": round(f.p_emos[i], 4), "p_raw": round(f.p_raw[i], 4),
                    "yes_bid": bid[i], "yes_ask": ask[i], "p_market": round(p_mkt[i], 4),
                })
    return pd.DataFrame(rows)


def score_forward(client: KalshiClient) -> pd.DataFrame:
    """Score every snapshot event that has since settled; rewrite forward_scores.csv."""
    snaps = sorted((LIVE_DIR / "snapshots").glob("*.csv"))
    if not snaps:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in snaps], ignore_index=True)
    # Keep the first snapshot of each event/lead (closest to the decision time).
    df = df.sort_values("run_utc").drop_duplicates(["ticker", "lead"], keep="first")
    results = {}
    for ev in df["event_ticker"].unique():
        try:
            event = client.get_event(ev)
        except Exception:
            continue
        for m in event.get("markets") or []:
            if m.get("result") in ("yes", "no"):
                results[m["ticker"]] = int(m["result"] == "yes")
    df["y"] = df["ticker"].map(results)
    rows = []
    for (ev, lead), g in df.groupby(["event_ticker", "lead"]):
        if g["y"].isna().any() or g["y"].sum() != 1:
            continue
        row = {"event_ticker": ev, "lead": lead, "date": g["date"].iloc[0]}
        for col in ("p_emos", "p_raw", "p_market"):
            p, y = g[col].to_numpy(float), g["y"].to_numpy(float)
            row[f"brier_{col[2:]}"] = round(float(np.sum((p - y) ** 2)), 5)
            row[f"logloss_{col[2:]}"] = round(float(-np.log(max(p[y == 1][0], EPS))), 5)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["date", "event_ticker", "lead"])
    out.to_csv(LIVE_DIR / "forward_scores.csv", index=False)
    return out


def run() -> None:
    LIVE_DIR.mkdir(exist_ok=True)
    (LIVE_DIR / "snapshots").mkdir(exist_ok=True)
    (LIVE_DIR / "ensemble").mkdir(exist_ok=True)
    stations = list(STATIONS.values())
    client = KalshiClient()
    now = datetime.now(UTC)
    snap = snapshot(client, FairValueService(), now)
    if not snap.empty:
        path = LIVE_DIR / "snapshots" / f"{now.date().isoformat()}.csv"
        if path.exists():  # a manual re-run on the same day appends rather than overwrites
            snap = pd.concat([pd.read_csv(path), snap], ignore_index=True)
        snap.to_csv(path, index=False)
    ens_path = LIVE_DIR / "ensemble" / f"{now.date().isoformat()}.csv.gz"
    ens = pd.DataFrame()
    if not ens_path.exists():  # once per day
        ens = collect.collect_ensemble_snapshot(stations)
        ens["tmax_f"] = ens["tmax_f"].round(1)
        ens.to_csv(ens_path, index=False)
    scores = score_forward(client)
    log.info("snapshot rows %d, ensemble rows %d, scored events %d", len(snap), len(ens),
             len(scores))
