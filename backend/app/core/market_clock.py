"""NSE market clock: IST time, trading days, sessions and entry windows.

Everything in the app asks this module "what time is it in the market" so that
timezone mistakes cannot creep in. All datetimes returned are timezone-aware IST.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

DEFAULT_CALENDAR_PATH = Path(__file__).resolve().parent.parent / "data" / "nse_holidays.json"


class HolidayCalendarMissing(RuntimeError):
    """Raised when the holiday calendar has no entry for the requested year.

    Callers must treat this as "do not trade": we never guess whether a day is
    a holiday. NSE publishes next year's list by circular each December.
    """


class Phase(StrEnum):
    CLOSED = "closed"
    PRE_OPEN = "pre_open"            # 09:00-09:15
    OPENING_RANGE = "opening_range"  # 09:15-09:30, no new entries
    ENTRY_WINDOW = "entry_window"    # 09:30-14:30, new entries allowed
    MANAGE_ONLY = "manage_only"      # 14:30-15:15, manage open trades only
    SQUARE_OFF = "square_off"        # 15:15-15:30, flatten everything
    POST_CLOSE = "post_close"        # after 15:30 on a trading day


@dataclass(frozen=True)
class SessionTimes:
    pre_open: time = time(9, 0)
    market_open: time = time(9, 15)
    entry_start: time = time(9, 30)
    entry_end: time = time(14, 30)
    square_off: time = time(15, 15)
    market_close: time = time(15, 30)


@dataclass(frozen=True)
class SpecialSession:
    day: date
    start: time
    end: time
    name: str


class MarketClock:
    def __init__(
        self,
        calendar_path: Path | str = DEFAULT_CALENDAR_PATH,
        session: SessionTimes | None = None,
    ) -> None:
        self.session = session or SessionTimes()
        self._holidays: dict[int, set[date]] = {}
        self._special: dict[date, SpecialSession] = {}
        self._load(Path(calendar_path))

    # ------------------------------------------------------------------ loading
    def _load(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for year_str, entry in raw.get("years", {}).items():
            year = int(year_str)
            self._holidays[year] = {date.fromisoformat(h["date"]) for h in entry.get("holidays", [])}
            for s in entry.get("special_sessions", []):
                d = date.fromisoformat(s["date"])
                self._special[d] = SpecialSession(
                    day=d,
                    start=time.fromisoformat(s["start"]),
                    end=time.fromisoformat(s["end"]),
                    name=s.get("name", "special session"),
                )

    # ------------------------------------------------------------------ queries
    @staticmethod
    def now() -> datetime:
        return datetime.now(IST)

    @staticmethod
    def to_ist(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            raise ValueError("naive datetime passed to market clock; attach a timezone")
        return dt.astimezone(IST)

    def has_calendar_for(self, year: int) -> bool:
        return year in self._holidays

    def is_holiday(self, d: date) -> bool:
        if d.year not in self._holidays:
            raise HolidayCalendarMissing(
                f"No NSE holiday calendar for {d.year}. Add it to nse_holidays.json "
                "from the official NSE circular before trading."
            )
        return d in self._holidays[d.year]

    def special_session(self, d: date) -> SpecialSession | None:
        return self._special.get(d)

    def is_trading_day(self, d: date) -> bool:
        """Regular trading day (weekday and not a holiday). Special sessions are separate."""
        if d.weekday() >= 5:
            return False
        return not self.is_holiday(d)

    def phase(self, at: datetime | None = None) -> Phase:
        at = self.to_ist(at) if at else self.now()
        if not self.is_trading_day(at.date()):
            return Phase.CLOSED
        t, s = at.time(), self.session
        if t < s.pre_open:
            return Phase.CLOSED
        if t < s.market_open:
            return Phase.PRE_OPEN
        if t < s.entry_start:
            return Phase.OPENING_RANGE
        if t < s.entry_end:
            return Phase.ENTRY_WINDOW
        if t < s.square_off:
            return Phase.MANAGE_ONLY
        if t < s.market_close:
            return Phase.SQUARE_OFF
        return Phase.POST_CLOSE

    def can_open_new_trades(self, at: datetime | None = None) -> bool:
        return self.phase(at) is Phase.ENTRY_WINDOW

    def next_trading_day(self, d: date) -> date:
        nxt = d + timedelta(days=1)
        for _ in range(30):
            if self.is_trading_day(nxt):
                return nxt
            nxt += timedelta(days=1)
        raise RuntimeError("no trading day found within 30 days; check the calendar")

    def kite_token_expiry(self, issued_at: datetime | None = None) -> datetime:
        """Kite access tokens expire at 06:00 IST the next morning."""
        issued_at = self.to_ist(issued_at) if issued_at else self.now()
        expiry = datetime.combine(issued_at.date(), time(6, 0), tzinfo=IST)
        if issued_at >= expiry:
            expiry += timedelta(days=1)
        return expiry
