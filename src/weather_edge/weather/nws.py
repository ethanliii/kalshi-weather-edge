"""Observed daily highs from the NWS Daily Climate Report (CLI).

The Iowa Environmental Mesonet archives every CLI product and exposes a parsed
JSON feed per station and year. For past days the entry reflects the final CLI
issued the following morning; the entry for "today" may be the preliminary
afternoon report, so callers should not treat the current day as settled.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

log = logging.getLogger(__name__)

IEM_CLI_URL = "https://mesonet.agron.iastate.edu/json/cli.py"


def fetch_cli_year(icao: str, year: int, session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests
    r = s.get(IEM_CLI_URL, params={"station": icao, "year": year}, timeout=60)
    r.raise_for_status()
    rows = []
    for rec in r.json().get("results", []):
        high = rec.get("high")
        if high in (None, "M", ""):
            continue
        rows.append({"date": date.fromisoformat(rec["valid"]), "cli_high_f": int(high),
                     "cli_product": rec.get("product")})
    return pd.DataFrame(rows, columns=["date", "cli_high_f", "cli_product"])


def fetch_cli_history(
    icao: str, start: date, end: date, session: requests.Session | None = None
) -> pd.DataFrame:
    frames = [fetch_cli_year(icao, y, session) for y in range(start.year, end.year + 1)]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df.drop_duplicates("date", keep="last").reset_index(drop=True)
