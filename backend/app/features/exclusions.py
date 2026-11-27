"""Which feature rows must not be used for training or backtests, and why."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

CA_FILE = Path(__file__).resolve().parent.parent / "data" / "corporate_actions.csv"

AFTER_EVENT_SESSIONS = 14   # daily lookbacks (ATR14, prev-day levels) are distorted this long
WARMUP_SESSIONS = 20        # a new series needs this many sessions before indicators settle


def load_corporate_actions(path: Path = CA_FILE) -> dict[str, set[date]]:
    events: dict[str, set[date]] = {}
    if not path.exists():
        return events
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    for row in csv.DictReader(lines):
        sym = (row.get("symbol") or "").strip().upper()
        if sym and row.get("date"):
            events.setdefault(sym, set()).add(date.fromisoformat(row["date"].strip()))
    return events


def excluded_days(sessions: list[date], events: set[date], special_sessions: set[date],
                  after: int = AFTER_EVENT_SESSIONS, warmup: int = WARMUP_SESSIONS) -> dict[date, str]:
    """Map day -> reason for one stock. `sessions` = that stock's trading days, sorted."""
    out: dict[date, str] = {}
    for d in sessions[:warmup]:
        out[d] = "warmup"
    index = {d: i for i, d in enumerate(sessions)}
    for ev in sorted(events):
        start = index.get(ev)
        if start is None:  # event on a day without data: start from the next session we have
            later = [i for d, i in index.items() if d > ev]
            if not later:
                continue
            start = min(later)
        for d in sessions[start:start + after + 1]:
            out[d] = "corporate_action"
    for d in sessions:
        if d in special_sessions:
            out[d] = "special_session"
    return out
