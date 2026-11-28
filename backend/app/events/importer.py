"""History from NSE's own CSV export (corporate-filings-announcements page, one company and up to a year per file).

File: CF-AN-equities-<SYMBOL>-<dd-mm-yyyy>-to-<dd-mm-yyyy>.csv
Columns: SYMBOL, COMPANY NAME, SUBJECT, DETAILS, BROADCAST DATE/TIME, RECEIPT, DISSEMINATION, DIFFERENCE, ATTACHMENT

The user downloads these in a normal browser and drops them into object storage under
imports/announcements/; the events-import job reads every file there, stores the rows and moves
the file to imports/announcements/done/. Re-importing a file is harmless (rows are keyed on the
attachment file name, the same key the RSS poller uses).

Board-meeting notices ("...meeting of the Board ... scheduled to be held on Friday, April 24, 2026 ...
to consider ... financial results") become results-calendar rows, known from the moment the
notice was disseminated.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime

from app.core.market_clock import IST

from . import classify, rss

PREFIX = "imports/announcements/"
DONE = PREFIX + "done/"

_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|" \
          "september|october|november|december"
_DATE = (rf"(?P<d1>(?:{_MONTHS})\.?\s+\d{{1,2}},?\s+\d{{4}})"                    # October 30, 2026
         rf"|(?P<d2>\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?,?\s+\d{{4}})"     # 30th October, 2026
         rf"|(?P<d3>\d{{1,2}}-(?:{_MONTHS})-\d{{4}})"                             # 30-Oct-2026
         rf"|(?P<d4>\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{4}})")                        # 30-10-2026, 30/10/2026
_MEETING = re.compile(r"board(?:\s+of\s+directors)?\b.{0,120}?\b(?:scheduled|to\s+be\s+held|will\s+be\s+held|"
                      r"proposed\s+to\s+be\s+held|meeting\s+on)\b\D{0,40}?(?:" + _DATE + ")",
                      re.IGNORECASE | re.DOTALL)


def _parse_date(text: str) -> date | None:
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text.strip(), flags=re.IGNORECASE).replace(",", " ").replace(".", " ")
    t = " ".join(t.split()).replace("Sept ", "Sep ").replace("sept ", "sep ")
    for f in ("%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y", "%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y", "%d/%m/%Y",
              "%d %m %Y"):
        try:
            return datetime.strptime(t, f).date()
        except ValueError:
            pass
    return None


def results_meeting(subject: str, body: str | None, public_at: datetime) -> date | None:
    """The date of an upcoming results board meeting announced in this filing, if any."""
    text = f"{subject or ''}. {body or ''}"
    low = text.lower()
    if "result" not in low or "outcome of board meeting" in (subject or "").lower():
        return None
    m = _MEETING.search(text)
    if not m:
        return None
    d = _parse_date(next(g for g in m.groups() if g))
    if d is None or not (public_at.date() <= d <= date.fromordinal(public_at.date().toordinal() + 60)):
        return None           # a past or far-off date is a reference to something else
    return d


def meeting_rows(announcements: list[dict]) -> list[dict]:
    out = []
    for a in announcements:
        d = results_meeting(a["subject"], a["body"], a["public_at"])
        if d:
            out.append({"source": "nse", "symbol": a["symbol"], "meeting_date": d, "purpose": "Financial Results",
                        "is_results": True, "known_at": a["public_at"], "known_at_estimated": False})
    return out


def parse_csv(raw: bytes) -> tuple[list[dict], int]:
    """Rows of one NSE export file, and how many lines could not be used."""
    text = raw.decode("utf-8-sig", errors="replace")
    rows, bad = [], 0
    for r in csv.DictReader(io.StringIO(text)):
        r = {(k or "").strip().upper(): (v or "").strip() for k, v in r.items()}
        sym = r.get("SYMBOL", "").upper()
        ts = None
        for col in ("DISSEMINATION", "BROADCAST DATE/TIME"):
            try:
                ts = datetime.strptime(r.get(col, ""), "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST)
                break
            except ValueError:
                continue
        if not sym or ts is None:
            bad += 1
            continue
        subject, body = r.get("SUBJECT", ""), r.get("DETAILS") or None
        link = r.get("ATTACHMENT")
        link = link if link and link != "-" else None
        rows.append({"source": "nse", "ext_id": rss.ext_id(link, sym, ts, subject), "symbol": sym, "public_at": ts,
                     "subject": subject[:500], "body": body[:4000] if body else None, "attachment": link,
                     "category": classify.classify(subject, body)})
    # the same filing can appear twice in one export (two attachments of one notice keep both)
    seen, unique = set(), []
    for row in rows:
        if row["ext_id"] not in seen:
            seen.add(row["ext_id"])
            unique.append(row)
    return unique, bad


_NAME = re.compile(r"CF-AN-equities-(?P<sym>.+?)-(?P<a>\d{2}-\d{2}-\d{4})-to-(?P<b>\d{2}-\d{2}-\d{4})", re.IGNORECASE)


def coverage(key: str, rows: list[dict]) -> tuple[str, date, date] | None:
    """(symbol, from, to) the file covers: from its NSE file name, else from its own rows."""
    m = _NAME.search(key.rsplit("/", 1)[-1])
    if m:
        a, b = (datetime.strptime(m.group(x), "%d-%m-%Y").date() for x in ("a", "b"))
        return m.group("sym").upper(), a, b
    if rows and len({r["symbol"] for r in rows}) == 1:
        days = [r["public_at"].date() for r in rows]
        return rows[0]["symbol"], min(days), max(days)
    return None


def import_all(engine, list_keys, get_bytes, move) -> dict:
    """Import every CSV under PREFIX (not done/). move(src, dst) archives a file after it is stored."""
    from .store import _log_window, save_announcements, save_board_meetings

    stats = {"files": 0, "announcements": 0, "results_meetings": 0, "bad_lines": 0, "symbols": set(),
             "first": None, "last": None, "skipped_files": []}
    for key in sorted(k for k in list_keys(PREFIX) if not k.startswith(DONE)):
        if not key.lower().endswith(".csv"):
            stats["skipped_files"].append(key)
            continue
        rows, bad = parse_csv(get_bytes(key))
        meetings = meeting_rows(rows)
        cov = coverage(key, rows)
        with engine.begin() as conn:
            save_announcements(conn, rows)
            save_board_meetings(conn, meetings)
            if cov:   # which (stock, days) we really have news for: training uses only these
                _log_window(conn, "announcements", cov[0], cov[1], cov[2], len(rows))
        if not cov:
            stats["skipped_files"].append(key + " (coverage unknown: rows stored but not used for training)")
        move(key, DONE + key[len(PREFIX):])
        stats["files"] += 1
        stats["announcements"] += len(rows)
        stats["results_meetings"] += len(meetings)
        stats["bad_lines"] += bad
        stats["symbols"].update(r["symbol"] for r in rows)
        for r in rows:
            d = r["public_at"].date()
            stats["first"] = min(stats["first"] or d, d)
            stats["last"] = max(stats["last"] or d, d)
    stats["symbols"] = len(stats["symbols"])
    return stats
