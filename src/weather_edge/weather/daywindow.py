"""Measurement-day windows and leak-free decision times.

The NWS Daily Climate Report (CLI) covers midnight-to-midnight *local standard
time* year-round. During daylight saving time that is 1:00 AM to 1:00 AM on the
local clock, so we work in UTC with a fixed standard offset and never call a
DST-aware conversion.

Forecast inputs come from Open-Meteo's Previous Runs API, where the value
`temperature_2m_previous_dayL` at valid time t was produced by a model run
initialised at least 24*L hours before t. For day D's window [start, end) the
newest run that can contribute is therefore initialised at end - 24*L hours,
which is 00:00 LST on day D-(L-1). Global models are published within ~6 hours
of initialisation, so a decision at 10:00 LST on day D-(L-1) only uses
information that was public at the time. `decision_time` encodes that rule and
`latest_input_init` exposes the bound so tests can check the margin.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

DECISION_HOUR_LST = 10
PUBLICATION_LAG = timedelta(hours=6)  # conservative upper bound for global NWP availability


def lst_midnight_utc(day: date, std_utc_offset_h: int) -> datetime:
    """UTC instant of 00:00 local standard time on `day`."""
    return datetime.combine(day, time(0), tzinfo=UTC) - timedelta(hours=std_utc_offset_h)


def cli_window_utc(day: date, std_utc_offset_h: int) -> tuple[datetime, datetime]:
    """[start, end) of the CLI measurement day in UTC."""
    start = lst_midnight_utc(day, std_utc_offset_h)
    return start, start + timedelta(days=1)


def latest_input_init(day: date, std_utc_offset_h: int, lead_days: int) -> datetime:
    """Newest model initialisation that can feed a lead-`lead_days` forecast for `day`."""
    _, end = cli_window_utc(day, std_utc_offset_h)
    return end - timedelta(days=lead_days)


def decision_time(day: date, std_utc_offset_h: int, lead_days: int) -> datetime:
    """When a lead-`lead_days` forecast for `day` is acted on (10:00 LST, lead-1 days earlier)."""
    ref = day - timedelta(days=lead_days - 1)
    return lst_midnight_utc(ref, std_utc_offset_h) + timedelta(hours=DECISION_HOUR_LST)
