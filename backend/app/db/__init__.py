"""Database helpers: migrations, job-run bookkeeping and advisory locks."""

from __future__ import annotations

import json
import logging
import zlib
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_MIGRATION_LOCK = 7_272_001


def split_sql(script: str) -> list[str]:
    """Split a migration file into single statements (driver-safe). No ';' inside bodies allowed."""
    lines = [ln for ln in script.splitlines() if not ln.strip().startswith("--")]
    return [stmt.strip() for stmt in "\n".join(lines).split(";") if stmt.strip()]


def migrate(engine) -> list[str]:
    """Apply any .sql migration not yet recorded. Safe to call from many pods at once."""
    from sqlalchemy import text

    applied_now: list[str] = []
    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _MIGRATION_LOCK})
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        ))
        done = {r[0] for r in conn.execute(text("SELECT name FROM schema_migrations"))}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in done:
                continue
            log.info("applying migration %s", path.name)
            for statement in split_sql(path.read_text(encoding="utf-8")):
                conn.exec_driver_sql(statement)
            conn.execute(text("INSERT INTO schema_migrations (name) VALUES (:n)"), {"n": path.name})
            applied_now.append(path.name)
    return applied_now


def _lock_key(name: str) -> int:
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


@contextmanager
def advisory_lock(engine, name: str):
    """Session-level lock so two copies of the same job never run at once.

    Yields True if we got the lock, False if another run holds it.
    """
    from sqlalchemy import text

    conn = engine.connect()
    try:
        got = conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": _lock_key(name)}).scalar()
        try:
            yield bool(got)
        finally:
            if got:
                conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _lock_key(name)})
    finally:
        conn.close()


class JobRun:
    """Records one job execution in job_runs."""

    def __init__(self, engine, job: str) -> None:
        from sqlalchemy import text

        self.engine, self.job, self.stats = engine, job, {}
        with engine.begin() as conn:
            self.id = conn.execute(
                text("INSERT INTO job_runs (job) VALUES (:j) RETURNING id"), {"j": job}
            ).scalar_one()

    def finish(self, status: str, error: str | None = None) -> None:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE job_runs SET finished_at = now(), status = :s, stats = CAST(:st AS jsonb),"
                    " error = :e WHERE id = :id"
                ),
                {"s": status, "st": json.dumps(self.stats, default=str), "e": error, "id": self.id},
            )
