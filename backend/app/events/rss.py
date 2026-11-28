"""NSE's official RSS feeds (nsearchives.nseindia.com/content/RSS/*.xml), polled continuously.

The announcements feed only holds the latest handful of filings, so it is polled about once a
minute and every poll is logged. If the oldest item in a poll is newer than the newest item
we already had, filings may have been missed in between: that poll is flagged as a gap, so
coverage is measured rather than assumed.

Item format (Nov 2026):
  title       Company name
  link        https://nsearchives.nseindia.com/corporate/<SYMBOL>_<ddmmyyyyHHMMSS>_<name>.pdf
  description <text> |SUBJECT: <NSE subject>
  pubDate     28-Nov-2026 07:11:16   (IST, not RFC 822)
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET  # noqa: S405 - official NSE feed; expat >= 2.4 resists entity bombs
from datetime import datetime

from app.core.market_clock import IST

from . import classify

log = logging.getLogger(__name__)

FEEDS = {"announcements": "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml"}
UA = {"User-Agent": "Mozilla/5.0 (personal trading research; polite RSS reader)"}

_LINK_SYMBOL = re.compile(r"/corporate/(?:xbrl/)?([^/]+?)_\d{8,14}_", re.IGNORECASE)


def poll_interval(now: datetime) -> int:
    """Seconds to the next poll: every minute from 07:00 to 20:00 IST (filings cluster then), else every 5."""
    return 60 if 7 <= now.astimezone(IST).hour < 20 else 300


def fetch(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 - fixed https host
        return r.read()


def symbol_from_link(link: str | None) -> str | None:
    if not link:
        return None
    m = _LINK_SYMBOL.search(urllib.parse.unquote(link))
    return m.group(1).upper() if m else None


def _norm_name(name: str) -> str:
    n = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    n = re.sub(r"\b(limited|ltd|the|india|co|company|corporation|corp)\b", " ", n)
    return " ".join(n.split())


def name_index(names_to_symbols: dict[str, str]) -> dict[str, str]:
    return {_norm_name(k): v for k, v in names_to_symbols.items()}


def parse_announcements(xml_bytes: bytes, by_name: dict[str, str] | None = None) -> tuple[list[dict], int]:
    """Returns (rows, unparsed). Symbol from the link, else from the company name."""
    root = ET.fromstring(xml_bytes)  # noqa: S314
    rows, bad = [], 0
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip() or None
        desc = (item.findtext("description") or "").strip()
        try:
            ts = datetime.strptime((item.findtext("pubDate") or "").strip(), "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST)
        except ValueError:
            bad += 1
            continue
        body, _, subject = desc.rpartition("|SUBJECT:")
        if not _:
            body, subject = desc, ""
        sym = symbol_from_link(link) or (by_name or {}).get(_norm_name(title))
        if not sym:
            bad += 1
            continue
        subject, body = subject.strip(), body.strip() or None
        rows.append({"source": "nse", "ext_id": ext_id(link, sym, ts, subject), "symbol": sym, "public_at": ts,
                     "subject": subject[:500] or title[:500], "body": body[:4000] if body else None,
                     "attachment": link, "category": classify.classify(subject, body)})
    return rows, bad


def ext_id(link: str | None, symbol: str, ts: datetime, subject: str) -> str:
    """Same id whichever way a filing reaches us (RSS now, CSV export for history): the attachment file name."""
    if link:
        return urllib.parse.unquote(link.rsplit("/", 1)[-1])[:200]
    import hashlib

    return hashlib.sha1(f"{symbol}|{ts.isoformat()}|{subject}".encode()).hexdigest()[:20]


def poll_once(engine, get_bytes, universe: set[str], by_name: dict[str, str]) -> dict:
    """One poll: parse, keep universe stocks, store, and log coverage."""
    from sqlalchemy import text

    from .importer import meeting_rows
    from .store import save_announcements, save_board_meetings

    rows, bad = parse_announcements(get_bytes(FEEDS["announcements"]), by_name)
    with engine.connect() as conn:
        newest_before = conn.execute(text("SELECT max(newest) FROM rss_polls WHERE feed = 'announcements'")).scalar()
    oldest = min((r["public_at"] for r in rows), default=None)
    newest = max((r["public_at"] for r in rows), default=None)
    gap = bool(newest_before and oldest and oldest > newest_before)
    keep = [r for r in rows if r["symbol"] in universe]
    with engine.begin() as conn:
        save_announcements(conn, keep)
        save_board_meetings(conn, meeting_rows(keep))
        conn.execute(text("INSERT INTO rss_polls (feed, items, kept, unparsed, oldest, newest, gap)"
                          " VALUES ('announcements', :i, :k, :u, :o, :n, :g)"),
                     {"i": len(rows), "k": len(keep), "u": bad, "o": oldest, "n": newest or newest_before, "g": gap})
    if gap:
        log.warning("possible gap: feed starts at %s but last poll ended at %s", oldest, newest_before)
    return {"items": len(rows), "kept": len(keep), "unparsed": bad, "gap": gap}
