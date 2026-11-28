"""Event storage and the incremental fetch loop.

A (symbol, window) is fetched once and logged. Re-runs skip logged windows except the most
recent few days, which are always re-asked so late filings are picked up. Events older than
the 5-minute candle retention are deleted with the candles (no point keeping news we have no
prices for).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from . import nse

log = logging.getLogger(__name__)

REFRESH_RECENT_DAYS = 3


def universe_symbols(engine, index_name: str, since: date) -> list[str]:
    from sqlalchemy import text

    with engine.connect() as conn:
        return sorted(r[0] for r in conn.execute(text(
            "SELECT DISTINCT symbol FROM universe_members WHERE index_name = :i"
            " AND (valid_to IS NULL OR valid_to >= :d)"), {"i": index_name, "d": since}))


def fetched_windows(engine, kind: str) -> set[tuple[str, date, date]]:
    from sqlalchemy import text

    with engine.connect() as conn:
        return {(r[0], r[1], r[2]) for r in conn.execute(text(
            "SELECT symbol, from_day, to_day FROM event_fetch_log WHERE source = 'nse' AND kind = :k"),
            {"k": kind})}


def save_announcements(conn, rows: list[dict]) -> None:
    from sqlalchemy import text

    if rows:
        conn.execute(text(
            "INSERT INTO corporate_announcements"
            " (source, ext_id, symbol, public_at, subject, body, attachment, category)"
            " VALUES (:source, :ext_id, :symbol, :public_at, :subject, :body, :attachment, :category)"
            " ON CONFLICT (source, ext_id) DO UPDATE SET subject = EXCLUDED.subject, body = EXCLUDED.body,"
            " category = EXCLUDED.category, attachment = EXCLUDED.attachment"), rows)


def save_board_meetings(conn, rows: list[dict]) -> None:
    from sqlalchemy import text

    if rows:
        conn.execute(text(
            "INSERT INTO board_meetings (source, symbol, meeting_date, purpose, is_results, known_at,"
            " known_at_estimated) VALUES (:source, :symbol, :meeting_date, :purpose, :is_results, :known_at,"
            " :known_at_estimated) ON CONFLICT (source, symbol, meeting_date, purpose) DO UPDATE SET"
            " is_results = EXCLUDED.is_results,"
            " known_at = LEAST(board_meetings.known_at, EXCLUDED.known_at)"), rows)


def _log_window(conn, kind: str, symbol: str, a: date, b: date, n: int) -> None:
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO event_fetch_log (source, kind, symbol, from_day, to_day, rows) VALUES ('nse', :k, :s, :a, :b, :n)"
        " ON CONFLICT (source, kind, symbol, from_day, to_day) DO UPDATE SET rows = EXCLUDED.rows,"
        " fetched_at = now()"), {"k": kind, "s": symbol, "a": a, "b": b, "n": n})


def plan(symbols: list[str], start: date, end: date, done: set, chunk_days: int) -> list[tuple[str, date, date]]:
    """Windows still to fetch. Windows touching the last few days are always re-fetched."""
    fresh_from = end - timedelta(days=REFRESH_RECENT_DAYS)
    todo = []
    for s in symbols:
        for a, b in nse.windows(start, end, chunk_days):
            if (s, a, b) in done and b < fresh_from:
                continue
            todo.append((s, a, b))
    return todo


def fetch_events(engine, get_json, symbols: list[str], start: date, end: date,
                 kinds: tuple[str, ...] = ("announcements", "board_meetings")) -> dict:
    """get_json(url) -> payload (NseSession.get_json in production). Stops on NseBlocked."""
    stats = {"requests": 0, "announcements": 0, "board_meetings": 0, "skipped_records": 0, "failures": []}
    specs = {"announcements": (nse.ANN_CHUNK_DAYS, nse.announcements_url, nse.parse_announcements,
                               save_announcements),
             "board_meetings": (nse.BM_CHUNK_DAYS, nse.board_meetings_url, nse.parse_board_meetings,
                                save_board_meetings)}
    for kind in kinds:
        chunk, url_fn, parse_fn, save_fn = specs[kind]
        todo = plan(symbols, start, end, fetched_windows(engine, kind), chunk)
        log.info("%s: %d windows to fetch for %d symbols", kind, len(todo), len(symbols))
        for i, (sym, a, b) in enumerate(todo, 1):
            try:
                payload = get_json(url_fn(sym, a, b))
            except nse.NseBlocked:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad window must not stop the backfill
                stats["failures"].append(f"{kind} {sym} {a}..{b}: {exc.__class__.__name__}: {exc}")
                continue
            stats["requests"] += 1
            rows, skipped = parse_fn(payload, sym)
            stats["skipped_records"] += skipped
            with engine.begin() as conn:
                save_fn(conn, rows)
                _log_window(conn, kind, sym, a, b, len(rows))
            stats[kind] += len(rows)
            if i % 100 == 0:
                log.info("%s: %d/%d windows, %d rows so far", kind, i, len(todo), stats[kind])
    return stats


def purge(engine, keep_days: int) -> dict:
    from sqlalchemy import text

    with engine.begin() as conn:
        a = conn.execute(text("DELETE FROM corporate_announcements"
                              " WHERE public_at < now() - make_interval(days => :d)"), {"d": keep_days}).rowcount
        b = conn.execute(text("DELETE FROM board_meetings WHERE meeting_date < current_date - :d"),
                         {"d": keep_days}).rowcount
        conn.execute(text("DELETE FROM event_fetch_log WHERE to_day < current_date - :d"), {"d": keep_days})
    return {"announcements": a, "board_meetings": b}


def load(engine, since: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text

    with engine.connect() as conn:
        ann = pd.DataFrame(conn.execute(text(
            "SELECT symbol, public_at, category FROM corporate_announcements WHERE public_at >= :d"),
            {"d": since}).mappings().all(), columns=["symbol", "public_at", "category"])
        bm = pd.DataFrame(conn.execute(text(
            "SELECT symbol, meeting_date, is_results, known_at FROM board_meetings WHERE meeting_date >= :d"),
            {"d": since - timedelta(days=120)}).mappings().all(),
            columns=["symbol", "meeting_date", "is_results", "known_at"])
    return ann, bm


def covered(engine, df: pd.DataFrame) -> pd.Series:
    """True for rows whose (symbol, day) falls in an imported or fetched announcements window."""
    from sqlalchemy import text

    with engine.connect() as conn:
        win = conn.execute(text("SELECT symbol, from_day, to_day FROM event_fetch_log WHERE kind = 'announcements'"
                                " AND rows >= 0")).all()
    return covered_by(df, win)


def covered_by(df: pd.DataFrame, windows) -> pd.Series:
    out = pd.Series(False, index=df.index)
    if not windows:
        return out
    days = pd.to_datetime(df["day"])
    for sym, a, b in windows:
        m = (df["symbol"] == sym) & (days >= pd.Timestamp(a)) & (days <= pd.Timestamp(b))
        out |= m
    return out
