"""Health endpoints: liveness, readiness and a system summary for the dashboard."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response, status

from app.config import get_settings
from app.core.market_clock import HolidayCalendarMissing

from .deps import get_clock, get_engine, get_redis, get_token_store

log = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
def live() -> dict:
    return {"status": "ok"}


@router.get("/ready")
def ready(response: Response) -> dict:
    checks: dict[str, str] = {}
    try:
        get_redis().ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {exc.__class__.__name__}: {str(exc)[:200]}"
    try:
        from sqlalchemy import text

        with get_engine().connect() as conn:
            conn.execute(text("select 1"))
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["postgres"] = f"error: {exc.__class__.__name__}: {str(exc)[:200]}"
    ok = all(v == "ok" for v in checks.values())
    if not ok:
        log.warning("readiness failed", extra={"extra_fields": checks})
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if ok else "degraded", "checks": checks}


@router.get("/system")
def system() -> dict:
    s, clock = get_settings(), get_clock()
    now = clock.now()
    try:
        phase = clock.phase(now).value
        calendar = "ok"
    except HolidayCalendarMissing as exc:
        phase, calendar = "unknown", str(exc)
    try:
        session = get_token_store().load()
        kite = {"logged_in": bool(session), **(session.public() if session else {})}
    except Exception as exc:  # noqa: BLE001
        kite = {"logged_in": False, "error": exc.__class__.__name__}
    return {
        "app_mode": s.app_mode.value,
        "trading_mode": s.trading_mode.value,
        "now_ist": now.isoformat(),
        "market_phase": phase,
        "holiday_calendar": calendar,
        "kite_configured": s.kite_configured,
        "kite": kite,
    }
