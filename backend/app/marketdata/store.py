"""Candle storage in Postgres: newest timestamp, upsert, retention purge."""

from __future__ import annotations

from datetime import datetime


def last_ts(engine, token: int, interval: str) -> datetime | None:
    from sqlalchemy import text

    with engine.connect() as conn:
        return conn.execute(
            text("SELECT max(ts) FROM candles WHERE instrument_token=:t AND interval=:i"),
            {"t": token, "i": interval},
        ).scalar()


def upsert_candles(engine, token: int, interval: str, rows: list[dict]) -> int:
    """Insert candles; a re-fetched candle overwrites the stored one (corrections, adjustments)."""
    if not rows:
        return 0
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO candles (instrument_token, interval, ts, open, high, low, close, volume)"
                " VALUES (:t, :i, :ts, :open, :high, :low, :close, :volume)"
                " ON CONFLICT (instrument_token, interval, ts) DO UPDATE SET open=EXCLUDED.open,"
                " high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume"
            ),
            [{"t": token, "i": interval, **r} for r in rows],
        )
    return len(rows)


def delete_token_interval(engine, token: int, interval: str) -> int:
    """Used by --refresh: drop a series so it is re-downloaded cleanly."""
    from sqlalchemy import text

    with engine.begin() as conn:
        return conn.execute(
            text("DELETE FROM candles WHERE instrument_token=:t AND interval=:i"), {"t": token, "i": interval}
        ).rowcount


def purge_older_than(engine, retention_days: dict[str, int]) -> dict[str, int]:
    from sqlalchemy import text

    deleted = {}
    with engine.begin() as conn:
        for interval, days in retention_days.items():
            deleted[interval] = conn.execute(
                text("DELETE FROM candles WHERE interval=:i AND ts < now() - make_interval(days => :d)"),
                {"i": interval, "d": days},
            ).rowcount
    return deleted


def load_candles(engine, token: int, interval: str, since: datetime) -> list[dict]:
    from sqlalchemy import text

    with engine.connect() as conn:
        return [
            dict(r) for r in conn.execute(
                text("SELECT ts, open, high, low, close, volume FROM candles WHERE instrument_token=:t"
                     " AND interval=:i AND ts >= :s ORDER BY ts"),
                {"t": token, "i": interval, "s": since},
            ).mappings()
        ]
