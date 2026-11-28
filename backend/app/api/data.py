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


@router.get("/features")
def features_manifest() -> dict:
    """Summary of the last feature build (from the manifest in object storage)."""
    import json

    from app.features.build import FEATURE_VERSION
    from app.providers.storage import ObjectStore

    try:
        raw = ObjectStore(get_settings()).get_bytes(f"features/{FEATURE_VERSION}/manifest.json")
    except Exception as exc:  # noqa: BLE001
        return {"built": False, "detail": f"no feature build yet ({exc.__class__.__name__})"}
    m = json.loads(raw)
    m["stats"] = {k: v for k, v in m["stats"].items() if k != "per_symbol"}
    return {"built": True, **m}


@router.get("/backtests")
def backtests(limit: int = Query(20, le=200)) -> dict:
    from sqlalchemy import text

    with get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT id, run_id, created_at, name, summary->>'trades' AS trades, summary->>'net_pnl' AS net_pnl,"
            " summary->>'profit_factor' AS profit_factor, summary->>'win_rate' AS win_rate,"
            " summary->>'max_drawdown_pct' AS max_drawdown_pct, summary->>'profitable_quarters' AS profitable_quarters,"
            " summary->>'gross_before_costs' AS gross_before_costs, summary->>'cost_per_trade' AS cost_per_trade,"
            " summary->>'would_halt_on' AS would_halt_on"
            " FROM backtest_runs ORDER BY created_at DESC, id LIMIT :l"), {"l": limit}).mappings().all()
    return {"count": len(rows), "runs": [dict(r) for r in rows]}


@router.get("/backtests/{run_pk}")
def backtest_detail(run_pk: int) -> dict:
    from fastapi import HTTPException
    from sqlalchemy import text

    with get_engine().connect() as conn:
        row = conn.execute(text("SELECT * FROM backtest_runs WHERE id = :i"), {"i": run_pk}).mappings().first()
    if not row:
        raise HTTPException(404, "backtest not found")
    return dict(row)


@router.get("/models")
def model_runs(limit: int = Query(10, le=100)) -> dict:
    from sqlalchemy import text

    with get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT id, run_id, created_at, barrier, verdict, status, status_reason, trained_until, spec,"
            " mlflow_run_id, chosen, chosen_on_holdout,"
            " summary->'base_win_rate_after_costs' AS base_win_rate_after_costs,"
            " summary->'holdout_quarters' AS holdout_quarters"
            " FROM model_runs ORDER BY created_at DESC LIMIT :l"), {"l": limit}).mappings().all()
    return {"count": len(rows), "runs": [dict(r) for r in rows]}


@router.get("/models/{run_pk}")
def model_run_detail(run_pk: int) -> dict:
    from fastapi import HTTPException
    from sqlalchemy import text

    with get_engine().connect() as conn:
        row = conn.execute(text("SELECT * FROM model_runs WHERE id = :i"), {"i": run_pk}).mappings().first()
    if not row:
        raise HTTPException(404, "model run not found")
    return dict(row)


@router.get("/models/{run_pk}/evals")
def model_run_evals(run_pk: int, limit: int = Query(20, le=200)) -> dict:
    """Forward (unseen-day) results recorded by the weekly retrain for a shadow or champion model."""
    from sqlalchemy import text

    with get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT evaluated_at, from_day, to_day, trading_days, summary FROM model_evals"
            " WHERE model_run_id = :i ORDER BY evaluated_at DESC LIMIT :l"), {"i": run_pk, "l": limit}).mappings().all()
    return {"count": len(rows), "evals": [dict(r) for r in rows]}


@router.get("/champion")
def champion() -> dict:
    """The only model a later phase may trade with. None is a valid, safe answer."""
    from sqlalchemy import text

    with get_engine().connect() as conn:
        row = conn.execute(text(
            "SELECT id, run_id, barrier, spec, chosen, trained_until, status_reason, status_changed_at"
            " FROM model_runs WHERE status = 'champion' ORDER BY status_changed_at DESC LIMIT 1")).mappings().first()
        shadows = conn.execute(text("SELECT count(*) FROM model_runs WHERE status = 'shadow'")).scalar()
    return {"champion": dict(row) if row else None, "shadows": shadows,
            "note": None if row else "no champion: models are not allowed to trade"}
