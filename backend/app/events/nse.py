"""NSE corporate announcements and board meetings (nseindia.com website JSON, no official API).

Polite by design: one request per second by default, a browser-like session that first loads
the home page for cookies, at most one session refresh per failure, and a hard stop
(NseBlocked) on repeated 401/403 or a non-JSON page. We never try to get around a block.

Parsing is defensive: NSE renames fields from time to time, so every field is looked up under
the names seen so far, and records without a usable timestamp are counted and skipped.
"""

from __future__ import annotations

import hashlib
import http.cookiejar
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date, datetime, timedelta

from app.core.market_clock import IST
from app.marketdata.history import RateLimiter

from . import classify

log = logging.getLogger(__name__)

BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/131.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
}
ANN_CHUNK_DAYS = 90
BM_CHUNK_DAYS = 365


class NseBlocked(RuntimeError):
    """NSE refused us (401/403/captcha page). Stop; don't retry harder."""


# fetch(url) -> parsed JSON; injected in tests
Fetcher = Callable[[str], object]


class NseSession:
    def __init__(self, rate_per_sec: float = 1.0, timeout: float = 20.0) -> None:
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.limiter = RateLimiter(rate_per_sec)
        self.timeout = timeout
        self.warm = False

    def _open(self, url: str) -> bytes:
        self.limiter.wait()
        req = urllib.request.Request(url, headers=HEADERS)
        with self.opener.open(req, timeout=self.timeout) as r:   # noqa: S310 - fixed https host
            return r.read()

    def _warm_up(self) -> None:
        self._open(BASE + "/")
        self._open(BASE + "/companies-listing/corporate-filings-announcements")
        self.warm = True

    def get_json(self, url: str):
        for attempt in range(3):
            try:
                if not self.warm:
                    self._warm_up()
                raw = self._open(url)
                try:
                    return json.loads(raw) if raw.strip() else []
                except json.JSONDecodeError as exc:
                    raise NseBlocked(f"non-JSON response from {url[:80]} (blocked or changed page)") from exc
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    if attempt == 0:
                        self.warm = False          # cookies expired: refresh once
                        continue
                    raise NseBlocked(f"HTTP {e.code} from NSE after a fresh session") from e
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
        raise NseBlocked("NSE kept refusing the request")


# ------------------------------------------------------------------ parsing
def _records(payload) -> list[dict]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for k in ("data", "Data", "rows", "results"):
            if isinstance(payload.get(k), list):
                return [r for r in payload[k] if isinstance(r, dict)]
    return []


def _pick(r: dict, *names):
    for n in names:
        v = r.get(n)
        if v not in (None, "", "-"):
            return v
    return None


_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%d%m%Y%H%M%S", "%d-%b-%Y %H:%M", "%d-%m-%Y %H:%M:%S")
_DAY_FORMATS = ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d")


def parse_ts(v) -> datetime | None:
    if not v:
        return None
    s = str(v).strip()
    for f in _TS_FORMATS:
        try:
            return datetime.strptime(s, f).replace(tzinfo=IST)
        except ValueError:
            pass
    return None


def parse_day(v) -> date | None:
    if not v:
        return None
    s = str(v).strip()[:11]
    for f in _DAY_FORMATS:
        try:
            return datetime.strptime(s, f).date()
        except ValueError:
            pass
    return None


def parse_announcements(payload, symbol: str) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    for r in _records(payload):
        ts = parse_ts(_pick(r, "sort_date", "an_dt", "dt", "exchdisstime"))
        if ts is None:
            skipped += 1
            continue
        subject = str(_pick(r, "desc", "subject", "smIndustry") or "").strip()
        body = _pick(r, "attchmntText", "body")
        body = str(body).strip() if body else None
        sym = str(_pick(r, "symbol") or symbol).upper()
        ext = _pick(r, "seq_id", "an_id", "id")
        ext_id = str(ext) if ext else hashlib.sha1(f"{sym}|{ts.isoformat()}|{subject}".encode()).hexdigest()[:20]
        rows.append({"source": "nse", "ext_id": ext_id, "symbol": sym, "public_at": ts, "subject": subject[:500],
                     "body": body[:4000] if body else None, "attachment": _pick(r, "attchmntFile"),
                     "category": classify.classify(subject, body)})
    return rows, skipped


def parse_board_meetings(payload, symbol: str) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    for r in _records(payload):
        day = parse_day(_pick(r, "bm_date", "meetingDate", "date"))
        if day is None:
            skipped += 1
            continue
        purpose = str(_pick(r, "bm_purpose", "purpose") or "").strip()
        desc = str(_pick(r, "bm_desc", "desc") or "")
        known = parse_ts(_pick(r, "bm_timestamp", "an_dt", "sort_date"))
        estimated = known is None
        if estimated:
            # rules require intimation >= 2 working days before a results meeting; assuming it became
            # known only 2 calendar days before (midnight) is never earlier than reality
            known = datetime.combine(day - timedelta(days=2), datetime.min.time(), tzinfo=IST)
        rows.append({"source": "nse", "symbol": str(_pick(r, "bm_symbol", "symbol") or symbol).upper(),
                     "meeting_date": day, "purpose": (purpose or desc)[:300],
                     "is_results": classify.is_results_meeting(purpose + " " + desc),
                     "known_at": known, "known_at_estimated": estimated})
    return rows, skipped


# ------------------------------------------------------------------ requests
def windows(start: date, end: date, days: int) -> list[tuple[date, date]]:
    out, cur = [], start
    while cur <= end:
        last = min(cur + timedelta(days=days - 1), end)
        out.append((cur, last))
        cur = last + timedelta(days=1)
    return out


def _q(d: date) -> str:
    return d.strftime("%d-%m-%Y")


def announcements_url(symbol: str, start: date, end: date) -> str:
    return (f"{BASE}/api/corporate-announcements?index=equities&symbol={urllib.parse.quote(symbol)}"
            f"&from_date={_q(start)}&to_date={_q(end)}")


def board_meetings_url(symbol: str, start: date, end: date) -> str:
    return (f"{BASE}/api/corporate-board-meetings?index=equities&symbol={urllib.parse.quote(symbol)}"
            f"&from_date={_q(start)}&to_date={_q(end)}")
