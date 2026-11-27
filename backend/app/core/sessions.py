"""Date-dependent session rules (Closing Auction Session and square-off times).

Since 3 Aug 2026 NSE runs a Closing Auction Session (CAS) for F&O stocks, so their
continuous trading ends earlier than other stocks, and brokers square off intraday
positions in them earlier. Rules live in data/session_rules.json, one period per change.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "session_rules.json"


@dataclass(frozen=True)
class SessionPeriod:
    start: date
    label: str
    continuous_end: dict[str, time]
    broker_mis_square_off: dict[str, time]
    our_square_off: dict[str, time]
    confirmed: bool
    source: str


def _times(d: dict[str, str]) -> dict[str, time]:
    return {k: time.fromisoformat(v) for k, v in d.items()}


class SessionRules:
    def __init__(self, path: Path | str = DEFAULT_RULES_PATH) -> None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        periods = sorted(
            (
                SessionPeriod(
                    start=date.fromisoformat(p["from"]),
                    label=p["label"],
                    continuous_end=_times(p["continuous_end"]),
                    broker_mis_square_off=_times(p["broker_mis_square_off"]),
                    our_square_off=_times(p["our_square_off"]),
                    confirmed=bool(p.get("confirmed", False)),
                    source=p.get("source", ""),
                )
                for p in raw["periods"]
            ),
            key=lambda p: p.start,
        )
        for p in periods:
            for kind in ("cas", "non_cas"):
                if p.our_square_off[kind] >= p.broker_mis_square_off[kind]:
                    raise ValueError(f"{p.label}: our square-off must be before the broker's MIS square-off")
                if p.our_square_off[kind] > p.continuous_end[kind]:
                    raise ValueError(f"{p.label}: our square-off must not be after continuous trading ends")
        self.periods = periods
        self._starts = [p.start for p in periods]

    def period(self, d: date) -> SessionPeriod:
        i = bisect.bisect_right(self._starts, d) - 1
        if i < 0:
            raise ValueError(f"no session rules for {d}")
        return self.periods[i]

    def continuous_end(self, d: date, cas: bool) -> time:
        return self.period(d).continuous_end["cas" if cas else "non_cas"]

    def square_off(self, d: date, cas: bool) -> time:
        return self.period(d).our_square_off["cas" if cas else "non_cas"]
