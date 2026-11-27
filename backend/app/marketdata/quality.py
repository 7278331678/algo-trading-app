"""Data-quality checks on stored candles.

Pure functions over lists of candle dicts, so they are easy to test. Reference
trading days come from the NIFTY 50 daily series: if the index traded, every
stock should have traded too.
"""

from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, time

from app.core.market_clock import IST

from .history import CANDLES_PER_SESSION


@dataclass(frozen=True)
class Issue:
    day: date | None
    severity: str  # critical | warning
    issue: str
    detail: str


def check_ohlc(rows: list[dict]) -> list[Issue]:
    out = []
    for r in rows:
        o, h, lo, c = r["open"], r["high"], r["low"], r["close"]
        d = r["ts"].astimezone(IST).date()
        if min(o, h, lo, c) <= 0:
            out.append(Issue(d, "critical", "non_positive_price", f"{r['ts']}: {o},{h},{lo},{c}"))
        elif h < lo or not (lo <= o <= h) or not (lo <= c <= h):
            out.append(Issue(d, "critical", "ohlc_inconsistent", f"{r['ts']}: o={o} h={h} l={lo} c={c}"))
    return out


def check_duplicates(rows: list[dict]) -> list[Issue]:
    counts = Counter(r["ts"] for r in rows)
    return [Issue(ts.astimezone(IST).date(), "critical", "duplicate_candle", str(ts))
            for ts, n in counts.items() if n > 1]


def check_completeness(rows: list[dict], interval: str, reference_days: set[date],
                       is_index: bool = False, reference_times: dict[date, list[time]] | None = None,
                       auction_end: dict[date, time] | None = None) -> list[Issue]:
    """Missing trading days, and intraday days with too few candles.

    reference_times: candle start times the reference index (NIFTY 50) had each day. A stock is
    expected to have the same candles, so short and special sessions are handled automatically.
    auction_end: on days with a Closing Auction Session, continuous trading for auction stocks ends
    at this time. A stock whose candles stop before it is checked only up to it; this follows the
    data, so stocks entering or leaving the F&O list need no separate history.
    A series' first day is not checked: listing days open late after a special auction.
    """
    by_day: dict[date, list[dict]] = defaultdict(list)
    for r in rows:
        by_day[r["ts"].astimezone(IST).date()].append(r)
    if not by_day:
        return [Issue(None, "critical", "no_data", f"no {interval} candles at all")]
    first = min(by_day)
    out = []
    for d in sorted(x for x in reference_days if x >= first):
        if d not in by_day:
            out.append(Issue(d, "critical", "missing_day", f"no {interval} candles on {d}"))
    if interval == "day":
        return out
    full = CANDLES_PER_SESSION[interval]
    for d, day_rows in sorted(by_day.items()):
        if d not in reference_days or d == first:
            continue
        times = (reference_times or {}).get(d)
        expected = len(times) if times is not None else full
        if expected < 0.75 * full:
            continue  # special session (e.g. Muhurat, ~1 hour): not traded, so not checked
        cut = (auction_end or {}).get(d)
        if times is not None and cut is not None and not is_index:
            last = max(r["ts"].astimezone(IST).time() for r in day_rows)
            if last < cut:
                expected = bisect.bisect_left(times, cut)
        n = len(day_rows)
        if n < 0.9 * expected:
            out.append(Issue(d, "critical", "incomplete_day", f"{n}/{expected} {interval} candles"))
        elif n < 0.98 * expected:
            out.append(Issue(d, "warning", "incomplete_day", f"{n}/{expected} {interval} candles"))
        if not is_index:
            zero = sum(1 for r in day_rows if r["volume"] == 0)
            if zero > 0.05 * n:
                out.append(Issue(d, "warning", "zero_volume", f"{zero}/{n} candles with zero volume"))
    return out


def check_jumps(daily_rows: list[dict], warn: float = 0.20, critical: float = 0.35) -> list[Issue]:
    """Large day-to-day moves: usually an unadjusted split/bonus, sometimes a real event."""
    out = []
    rows = sorted(daily_rows, key=lambda r: r["ts"])
    for prev, cur in zip(rows, rows[1:], strict=False):
        move = abs(cur["open"] / prev["close"] - 1)
        if move >= warn:
            sev = "critical" if move >= critical else "warning"
            out.append(Issue(cur["ts"].astimezone(IST).date(), sev, "price_jump",
                             f"open {cur['open']} vs prior close {prev['close']} ({move:.0%}); "
                             "possible unadjusted corporate action: re-fetch with --refresh"))
    return out


def run_checks(rows: list[dict], interval: str, reference_days: set[date], is_index: bool = False,
               reference_times: dict[date, list[time]] | None = None,
               auction_end: dict[date, time] | None = None) -> list[Issue]:
    issues = check_ohlc(rows) + check_duplicates(rows)
    issues += check_completeness(rows, interval, reference_days, is_index=is_index,
                                 reference_times=reference_times, auction_end=auction_end)
    if interval == "day" and not is_index:  # VIX and indices legitimately jump
        issues += check_jumps(rows)
    return issues
