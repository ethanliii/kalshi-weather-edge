"""Data collection: forecasts, observed CLI highs, settled markets and quotes.

Everything is cached under data/raw/ as parquet and collected incrementally, so
re-running only fetches what is missing. Layout:

  data/raw/cli/{ICAO}.parquet              observed CLI highs
  data/raw/forecasts/{SERIES}.parquet      multi-model CLI-day max, leads 1-2
  data/raw/markets/{SERIES}.parquet        settled market metadata + results
  data/raw/quotes/{SERIES}.parquet         YES bid/ask at each decision time
  data/raw/ensemble/{YYYY-MM-DD}.parquet   live ensemble snapshots (forward archive)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from weather_edge.config import RAW_DIR
from weather_edge.kalshi.client import KalshiClient, parse_ts
from weather_edge.kalshi.markets import bracket_from_market, event_date, settlement_source
from weather_edge.stations import STATIONS, Station
from weather_edge.weather import nws, openmeteo
from weather_edge.weather.daywindow import decision_time

log = logging.getLogger(__name__)
LEADS = (1, 2)


def _path(kind: str, name: str) -> Path:
    p = RAW_DIR / kind / f"{name}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def _yesterday() -> date:
    return datetime.now(UTC).date() - timedelta(days=1)


# --------------------------------------------------------------------------- CLI
def collect_cli(station: Station, start: date = openmeteo.HISTORY_START) -> pd.DataFrame:
    path = _path("cli", station.icao)
    old = _read(path)
    end = _yesterday()
    if old is not None and not old.empty:
        # Re-pull the last two weeks: CLI products are occasionally corrected.
        start = max(start, old["date"].max() - timedelta(days=14))
    new = nws.fetch_cli_history(station.icao, start, end)
    df = new if old is None else pd.concat([old[old["date"] < start], new])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(path, index=False)
    log.info("CLI %s: %d days through %s", station.icao, len(df), df["date"].max())
    return df


# --------------------------------------------------------------------- forecasts
def collect_forecasts(station: Station, start: date = openmeteo.HISTORY_START) -> pd.DataFrame:
    path = _path("forecasts", station.series)
    old = _read(path)
    end = _yesterday() + timedelta(days=2)  # include tomorrow's lead-1/lead-2 values
    if old is not None and not old.empty:
        complete = old.dropna(subset=["tmax_f"])
        # Refetch recent days, whose windows may have been partially in the future.
        start = max(start, complete["date"].max() - timedelta(days=3))
    new = openmeteo.fetch_previous_runs(station, start, end)
    df = new if old is None else pd.concat([old[old["date"] < start], new])
    df = df.sort_values(["date", "lead", "model"]).reset_index(drop=True)
    df.to_parquet(path, index=False)
    log.info("forecasts %s: %d rows through %s", station.series, len(df), df["date"].max())
    return df


FRESH_START = date(2024, 3, 15)


def collect_fresh_runs(station: Station, start: date = FRESH_START) -> pd.DataFrame:
    """Decision-day 00Z ECMWF HRES highs (ablation input). Rows: date, lead, model, tmax_f."""
    path = _path("fresh", station.series)
    old = _read(path)
    have = set() if old is None else set(old["run_day"])
    end = datetime.now(UTC).date()
    todo = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in todo if d not in have]
    rows: list[dict] = []
    session = __import__("requests").Session()

    def save() -> pd.DataFrame:
        df = pd.DataFrame(rows) if old is None else pd.concat([old, pd.DataFrame(rows)])
        if not df.empty:
            df.to_parquet(path, index=False)
        return df

    for i, run_day in enumerate(todo, 1):
        try:
            highs = openmeteo.fetch_decision_day_run(station, run_day, session=session)
        except Exception as exc:  # missing runs are retried on the next invocation
            log.warning("fresh run %s %s: %s", station.series, run_day, exc)
            continue
        for lead, tmax in highs.items():
            rows.append({"run_day": run_day, "date": run_day + timedelta(days=lead - 1),
                         "lead": lead, "model": "ecmwf_fresh", "tmax_f": tmax})
        time.sleep(0.25)  # stay well inside Open-Meteo's free-tier limits
        if i % 100 == 0:
            log.info("fresh %s: %d/%d", station.series, i, len(todo))
            save()
    df = save()
    log.info("fresh %s: %d rows", station.series, len(df))
    return df


def collect_ensemble_snapshot(stations: list[Station]) -> pd.DataFrame:
    """Archive today's live ensemble so true-ensemble skill can be evaluated later."""
    issued = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    frames = []
    for st in stations:
        df = openmeteo.fetch_ensemble(st)
        df.insert(0, "series", st.series)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["issued_utc"] = issued
    out.to_parquet(_path("ensemble", issued.strftime("%Y-%m-%dT%H")), index=False)
    return out


# ----------------------------------------------------------------------- markets
MARKET_COLS = ["ticker", "event_ticker", "date", "lo", "hi", "result", "expiration_value",
               "source", "close_time", "settlement_ts", "volume"]


def _market_row(m: dict) -> dict:
    b = bracket_from_market(m)
    ev = m.get("expiration_value")
    return {
        "ticker": m["ticker"],
        "event_ticker": m["event_ticker"],
        "date": event_date(m["event_ticker"]),
        "lo": b.lo,
        "hi": b.hi,
        "result": m.get("result"),
        "expiration_value": float(ev) if ev not in (None, "") else None,
        "source": settlement_source(m.get("rules_primary", "")),
        "close_time": parse_ts(m.get("close_time")),
        "settlement_ts": parse_ts(m.get("settlement_ts")) or parse_ts(m.get("expiration_time")),
        "volume": float(m.get("volume_fp", m.get("volume")) or 0),
    }


def collect_markets(
    client: KalshiClient, station: Station, since: date = openmeteo.HISTORY_START
) -> pd.DataFrame:
    rows = []
    for m in client.all_settled_markets(station.series):
        if m.get("result") not in ("yes", "no"):
            continue
        try:
            row = _market_row(m)
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("skip %s: %s", m.get("ticker"), exc)
            continue
        if row["date"] >= since:
            rows.append(row)
    df = pd.DataFrame(rows, columns=MARKET_COLS).sort_values(["date", "lo"])
    df.to_parquet(_path("markets", station.series), index=False)
    log.info("markets %s: %d settled markets in %d events", station.series, len(df),
             df["event_ticker"].nunique())
    return df


# ------------------------------------------------------------------------ quotes
def _quotes_for_market(client: KalshiClient, station: Station, row: pd.Series) -> list[dict]:
    times = {lead: decision_time(row["date"], station.std_utc_offset_h, lead) for lead in LEADS}
    start = min(times.values()) - timedelta(hours=2)
    end = max(times.values())
    candles = client.candlesticks(station.series, row["ticker"], start, end, 60,
                                  settled_at=row["settlement_ts"])
    out = []
    for lead, t in times.items():
        prior = [c for c in candles if c["end_ts"] <= t]
        c = prior[-1] if prior else None
        out.append({
            "ticker": row["ticker"],
            "lead": lead,
            "decision_time": t,
            "candle_end": c["end_ts"] if c else None,
            "yes_bid": c["yes_bid_close"] if c else None,
            "yes_ask": c["yes_ask_close"] if c else None,
            "open_interest": c["open_interest"] if c else None,
        })
    return out


def collect_quotes(client: KalshiClient, station: Station, workers: int = 6) -> pd.DataFrame:
    markets = _read(_path("markets", station.series))
    if markets is None or markets.empty:
        return pd.DataFrame()
    path = _path("quotes", station.series)
    old = _read(path)
    done = set() if old is None else set(old["ticker"])
    todo = markets[~markets["ticker"].isin(done)]
    log.info("quotes %s: %d markets to fetch (%d cached)", station.series, len(todo), len(done))
    rows: list[dict] = []
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(_quotes_for_market, client, station, r) for _, r in todo.iterrows()]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rows += fut.result()
            except Exception as exc:  # keep going; failures are refetched next run
                log.warning("quote fetch failed: %s", exc)
            if i % 500 == 0:
                log.info("quotes %s: %d/%d", station.series, i, len(todo))
                _save_quotes(path, old, rows)
    return _save_quotes(path, old, rows)


def _save_quotes(path: Path, old: pd.DataFrame | None, rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(rows)
    df = new if old is None else pd.concat([old, new], ignore_index=True)
    if not df.empty:
        df = df.drop_duplicates(["ticker", "lead"], keep="last")
        df.to_parquet(path, index=False)
    return df


def stations_for(series: list[str] | None) -> list[Station]:
    return [STATIONS[s] for s in (series or sorted(STATIONS))]
