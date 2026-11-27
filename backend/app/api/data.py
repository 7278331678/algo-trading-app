"""Read-only data endpoints: what is stored, how fresh it is, last job runs, quality."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query

from app.config import get_settings
from app.marketdata import universe

from .deps import get_clock, get_engine

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/status")
def status() -> dict:
    from sqlalchemy import text

    engine = get_engine()
    with engine.connect() as conn:
        coverage = [
            dict(r) for r in conn.execute(text(
                "SELECT interval, count(DISTINCT instrument_token) AS instruments, count(*) AS candles,"
                " min(ts) AS oldest, max(ts) AS newest FROM candles GROUP BY interval ORDER BY interval"
            )).mappings()
        ]
        jobs = [
            dict(r) for r in conn.execute(text(
                "SELECT DISTINCT ON (job) job, status, started_at, finished_at, error, stats"
                " FROM job_runs ORDER BY job, started_at DESC"
            )).mappings()
        ]
        last_qc = conn.execute(text(
            "SELECT run_id, count(*) FILTER (WHERE severity='critical') AS critical,"
            " count(*) FILTER (WHERE severity='warning') AS warning FROM data_quality_issues"
            " WHERE run_id = (SELECT max(run_id) FROM data_quality_issues) GROUP BY run_id"
        )).mappings().first()
        blocks = [
            dict(r) for r in conn.execute(text(
                "SELECT b.day, i.tradingsymbol, b.reason FROM trading_blocks b"
                " LEFT JOIN instruments i USING (instrument_token) WHERE b.day >= current_date"
                " ORDER BY b.day, i.tradingsymbol"
            )).mappings()
        ]
    for j in jobs:  # keep the response small
        j["stats"] = {k: v for k, v in (j["stats"] or {}).items() if k not in {"failures", "by_interval"}}
    return {
        "universe_members_today": len(universe.members_on(engine, get_settings().universe_index,
                                                          get_clock().now().date())),
        "coverage": coverage,
        "last_jobs": jobs,
        "last_quality_check": dict(last_qc) if last_qc else None,
        "upcoming_trading_blocks": blocks,
    }


@router.get("/universe")
def members(on: date | None = Query(None, description="point-in-time date, default today")) -> dict:
    day = on or get_clock().now().date()
    rows = universe.members_on(get_engine(), get_settings().universe_index, day)
    return {"index": get_settings().universe_index, "on": day, "count": len(rows), "members": rows}


@router.get("/quality")
def quality_issues(limit: int = Query(100, le=1000), severity: str | None = None) -> dict:
    from sqlalchemy import text

    with get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT q.day, i.tradingsymbol, q.interval, q.severity, q.issue, q.detail"
            " FROM data_quality_issues q LEFT JOIN instruments i USING (instrument_token)"
            " WHERE q.run_id = (SELECT max(run_id) FROM data_quality_issues)"
            " AND (CAST(:sev AS text) IS NULL OR q.severity = :sev)"
            " ORDER BY q.severity, q.day DESC NULLS FIRST LIMIT :lim"
        ), {"sev": severity, "lim": limit}).mappings().all()
    return {"count": len(rows), "issues": [dict(r) for r in rows]}
