"""Historical candles from Kite: chunk planning, rate limiting, retries, completeness.

Kite caps how many days one historical request may span per interval and how
many requests per second you may send. Everything here is deterministic and unit
tested; the Kite client is injected so tests never touch the network.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta

from app.core.market_clock import IST

log = logging.getLogger(__name__)

INTERVALS = ("day", "15minute", "5minute", "minute")

# Maximum calendar days per historical request (Kite Connect limits)
MAX_DAYS_PER_REQUEST = {"minute": 60, "5minute": 100, "15minute": 200, "day": 2000}

STEP = {
    "minute": timedelta(minutes=1),
    "5minute": timedelta(minutes=5),
    "15minute": timedelta(minutes=15),
    "day": timedelta(days=1),
}

# candles per full NSE session (09:15-15:30)
CANDLES_PER_SESSION = {"minute": 375, "5minute": 75, "15minute": 25, "day": 1}


class TokenExpired(RuntimeError):
    """Kite rejected the access token: the morning login is needed."""


def plan_chunks(start: datetime, end: datetime, interval: str) -> list[tuple[datetime, datetime]]:
    """Split [start, end] into non-overlapping request windows within Kite's span limit."""
    if interval not in MAX_DAYS_PER_REQUEST:
        raise ValueError(f"unknown interval {interval}")
    if start > end:
        return []
    span = MAX_DAYS_PER_REQUEST[interval]
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor <= end:
        last_day = cursor.date() + timedelta(days=span - 1)
        chunk_end = min(datetime.combine(last_day, datetime.max.time().replace(microsecond=0),
                                         tzinfo=cursor.tzinfo), end)
        chunks.append((cursor, chunk_end))
        cursor = datetime.combine(chunk_end.date() + timedelta(days=1), datetime.min.time(),
                                  tzinfo=cursor.tzinfo)
    return chunks


def resume_start(last_ts: datetime | None, window_start: date, interval: str) -> datetime:
    """Where fetching should restart: right after the newest stored candle, or the window start."""
    floor = datetime.combine(window_start, datetime.min.time(), tzinfo=IST)
    if last_ts is None:
        return floor
    return max(last_ts.astimezone(IST) + STEP[interval], floor)


def is_complete(candle_ts: datetime, interval: str, now: datetime) -> bool:
    """True if the candle has closed (Kite also returns the candle still forming)."""
    ts = candle_ts.astimezone(IST)
    if interval == "day":
        close = datetime.combine(ts.date(), datetime.min.time(), tzinfo=IST).replace(hour=15, minute=30)
        return now >= close
    return ts + STEP[interval] <= now


class RateLimiter:
    """Evenly spaces calls to at most `rate` per second, across threads."""

    def __init__(self, rate: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.interval = 1.0 / rate
        self.clock, self.sleep = clock, sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self.clock()
            if now < self._next:
                self.sleep(self._next - now)
                now = self._next
            self._next = now + self.interval


_RETRYABLE = {"NetworkException", "DataException", "GeneralException", "ReadTimeout",
              "ConnectTimeout", "ConnectionError", "Timeout"}


def fetch_chunk(kite, token: int, start: datetime, end: datetime, interval: str,
                limiter: RateLimiter, retries: int = 4,
                sleep: Callable[[float], None] = time.sleep) -> list[dict]:
    """One historical request with retries and backoff. Raises TokenExpired on auth failure."""
    attempt = 0
    while True:
        limiter.wait()
        try:
            return kite.historical_data(
                token,
                start.strftime("%Y-%m-%d %H:%M:%S"),
                end.strftime("%Y-%m-%d %H:%M:%S"),
                interval,
            )
        except Exception as exc:
            name = exc.__class__.__name__
            if name in {"TokenException", "PermissionException"}:
                raise TokenExpired(str(exc)) from exc
            if name in {"InputException"}:
                raise  # bad request: retrying will not help
            attempt += 1
            if attempt > retries or (name not in _RETRYABLE and "429" not in str(exc)):
                raise
            backoff = min(2 ** attempt, 30)
            log.warning("historical_data %s %s failed (%s), retry %d in %ss",
                        token, interval, name, attempt, backoff)
            sleep(backoff)


def normalize(rows: list[dict], interval: str, now: datetime) -> list[dict]:
    """Convert Kite rows to our schema, dropping candles that are still forming."""
    out = []
    for r in rows:
        ts = r["date"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=IST)
        if not is_complete(ts, interval, now):
            continue
        out.append({
            "ts": ts, "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]), "volume": int(r.get("volume") or 0),
        })
    return out
