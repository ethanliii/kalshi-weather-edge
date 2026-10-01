"""Order ledger and kill switch for paper trading.

Ledger: append-only JSONL, one record per event (decision, order request and
response, fill, cancel, kill-switch trip). Nothing is ever rewritten, so the
log is an audit trail of exactly what the bot did and why.

Kill switch: trading halts (and resting orders are cancelled) if any of
  * a KILL file exists in the log directory (manual: `touch logs/paper/KILL`),
  * env var WEATHER_EDGE_KILL=1,
  * the day's realised + unrealised loss exceeds `max_daily_loss`,
  * `max_consecutive_errors` API errors happen in a row (counted on disk, so the
    count survives process restarts, e.g. under cron).
Once tripped it stays tripped (state persisted) until a human deletes the
state file — an automated system must never un-trip itself.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


def _jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, datetime | date):
        return x.isoformat()
    return str(x)


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **fields: Any) -> dict:
        record = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), "kind": kind,
                  **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=_jsonable) + "\n")
        return record

    def records(self, kind: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        return [r for r in rows if kind is None or r["kind"] == kind]


@dataclass
class KillSwitchConfig:
    max_daily_loss: float = 100.0
    max_consecutive_errors: int = 5


class KillSwitch:
    def __init__(self, state_dir: Path, cfg: KillSwitchConfig | None = None,
                 ledger: Ledger | None = None):
        self.state_dir = state_dir
        self.cfg = cfg or KillSwitchConfig()
        self.ledger = ledger
        self.state_file = state_dir / "kill_switch.json"
        self.manual_file = state_dir / "KILL"
        self.errors_file = state_dir / "consecutive_errors"
        state_dir.mkdir(parents=True, exist_ok=True)

    @property
    def consecutive_errors(self) -> int:
        return int(self.errors_file.read_text()) if self.errors_file.exists() else 0

    @property
    def tripped(self) -> bool:
        return (self.state_file.exists() or self.manual_file.exists()
                or os.environ.get("WEATHER_EDGE_KILL") == "1")

    def reason(self) -> str | None:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text()).get("reason")
        if self.manual_file.exists():
            return "manual KILL file"
        if os.environ.get("WEATHER_EDGE_KILL") == "1":
            return "WEATHER_EDGE_KILL=1"
        return None

    def trip(self, reason: str) -> None:
        if not self.state_file.exists():
            self.state_file.write_text(json.dumps(
                {"reason": reason, "ts": datetime.now(UTC).isoformat()}))
        if self.ledger:
            self.ledger.log("kill_switch", reason=reason)

    def check_pnl(self, day_pnl: float) -> None:
        if day_pnl <= -abs(self.cfg.max_daily_loss):
            self.trip(f"daily loss {day_pnl:.2f} exceeds {self.cfg.max_daily_loss:.2f}")

    def record_error(self, err: Exception) -> None:
        n = self.consecutive_errors + 1
        self.errors_file.write_text(str(n))
        if n >= self.cfg.max_consecutive_errors:
            self.trip(f"{n} consecutive API errors; last: {err}")

    def record_success(self) -> None:
        if self.errors_file.exists():
            self.errors_file.unlink()
