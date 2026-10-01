"""Kalshi daily-high series and the NWS climate stations they settle on.

Every active series' rules name an NWS Daily Climate Report (CLI) site, e.g.
"the maximum temperature recorded at New York City (CLINYC)". The CLI covers a
*local standard time* day: midnight-to-midnight LST all year, so during daylight
saving time the measurement window is 1:00 AM to 1:00 AM local clock time.
We therefore store a fixed standard-time UTC offset per station and never use
DST-aware local time when building forecast windows.

Coordinates, elevations and time zones come from api.weather.gov/stations/{id}
(retrieved 2026-10-01).

Settlement source history (verified from market rules via the Kalshi API):
  * through ~2026-08-13: "National Weather Service's Climatological Report (Daily)"
  * from ~2026-08-16:    "The Weather Company" reporting the same CLI site
The exact source is recorded per event from `rules_primary` when data are built.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    series: str  # Kalshi series ticker
    city: str
    cli_code: str  # NWS CLI product site id, as named in the market rules
    icao: str  # ASOS station id (IEM / api.weather.gov)
    lat: float
    lon: float
    elevation_m: float
    tz: str  # IANA zone (for display only)
    std_utc_offset_h: int  # local *standard* time offset; CLI day boundaries use this

    @property
    def name(self) -> str:
        return self.city


_ROWS = [
    # series,        city,            cli,      icao,   lat,       lon,         elev_m,  tz,                          std
    ("KXHIGHNY",    "New York City", "CLINYC", "KNYC", 40.78333, -73.96667, 46.9, "America/New_York", -5),
    ("KXHIGHCHI",   "Chicago",       "CLIMDW", "KMDW", 41.78417, -87.75528, 188.1, "America/Chicago", -6),
    ("KXHIGHMIA",   "Miami",         "CLIMIA", "KMIA", 25.79056, -80.31639, 3.0, "America/New_York", -5),
    ("KXHIGHAUS",   "Austin",        "CLIAUS", "KAUS", 30.18304, -97.67987, 148.1, "America/Chicago", -6),
    ("KXHIGHDEN",   "Denver",        "CLIDEN", "KDEN", 39.84658, -104.65622, 1647.1, "America/Denver", -7),
    ("KXHIGHLAX",   "Los Angeles",   "CLILAX", "KLAX", 33.93806, -118.38889, 38.1, "America/Los_Angeles", -8),
    ("KXHIGHPHIL",  "Philadelphia",  "CLIPHL", "KPHL", 39.87327, -75.22678, 2.1, "America/New_York", -5),
    ("KXHIGHTHOU",  "Houston",       "CLIHOU", "KHOU", 29.63750, -95.28250, 14.0, "America/Chicago", -6),
    ("KXHIGHTNOLA", "New Orleans",   "CLIMSY", "KMSY", 29.99278, -90.25083, 0.9, "America/Chicago", -6),
    ("KXHIGHTSEA",  "Seattle",       "CLISEA", "KSEA", 47.44472, -122.31361, 130.1, "America/Los_Angeles", -8),
    ("KXHIGHTDAL",  "Dallas",        "CLIDFW", "KDFW", 32.89743, -97.02196, 164.9, "America/Chicago", -6),
    ("KXHIGHTATL",  "Atlanta",       "CLIATL", "KATL", 33.64028, -84.42694, 313.0, "America/New_York", -5),
    ("KXHIGHTSDF",  "Louisville",    "CLISDF", "KSDF", 38.17406, -85.73650, 152.4, "America/Kentucky/Louisville", -5),
    ("KXHIGHTEWR",  "Newark",        "CLIEWR", "KEWR", 40.68250, -74.16944, 4.9, "America/New_York", -5),
    ("KXHIGHTLV",   "Las Vegas",     "CLILAS", "KLAS", 36.07188, -115.16340, 664.5, "America/Los_Angeles", -8),
    ("KXHIGHTPHX",  "Phoenix",       "CLIPHX", "KPHX", 33.42780, -112.00347, 339.9, "America/Phoenix", -7),
    ("KXHIGHTMIN",  "Minneapolis",   "CLIMSP", "KMSP", 44.88306, -93.22889, 256.0, "America/Chicago", -6),
    ("KXHIGHTOKC",  "Oklahoma City", "CLIOKC", "KOKC", 35.38861, -97.60028, 394.1, "America/Chicago", -6),
    ("KXHIGHTSAN",  "San Diego",     "CLISAN", "KSAN", 32.73361, -117.18306, 4.0, "America/Los_Angeles", -8),
    ("KXHIGHTTTN",  "Trenton",       "CLITTN", "KTTN", 40.27639, -74.81639, 64.0, "America/New_York", -5),
    ("KXHIGHTSFO",  "San Francisco", "CLISFO", "KSFO", 37.61961, -122.36558, 3.0, "America/Los_Angeles", -8),
    ("KXHIGHTSATX", "San Antonio",   "CLISAT", "KSAT", 29.53278, -98.46361, 246.0, "America/Chicago", -6),
    ("KXHIGHTDC",   "Washington DC", "CLIDCA", "KDCA", 38.84833, -77.03417, 4.0, "America/New_York", -5),
    ("KXHIGHTBOS",  "Boston",        "CLIBOS", "KBOS", 42.36056, -71.01056, 6.1, "America/New_York", -5),
]  # fmt: skip

STATIONS: dict[str, Station] = {r[0]: Station(*r) for r in _ROWS}


def by_series(series: str) -> Station:
    try:
        return STATIONS[series]
    except KeyError as exc:
        raise KeyError(f"unknown series {series!r}; known: {sorted(STATIONS)}") from exc
