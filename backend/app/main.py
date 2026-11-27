"""FastAPI entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import auth, data, health
from app.config import get_settings
from app.logging_setup import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        from app.api.deps import get_engine
        from app.db import migrate

        applied = migrate(get_engine())
        if applied:
            log.info("migrations applied: %s", applied)
    except Exception as exc:  # noqa: BLE001 - readiness will report the database problem
        log.error("migrations not applied at startup: %s", exc.__class__.__name__)
    yield


app = FastAPI(
    title="Algo Trading App",
    version="0.2.0",
    description="Multi-agent intraday trading system for Zerodha (Phase 1: data layer).",
    lifespan=lifespan,
)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(data.router)


@app.get("/")
def root() -> dict:
    return {
        "name": "algo-trading-app",
        "app_mode": settings.app_mode.value,
        "trading_mode": settings.trading_mode.value,
        "docs": "/docs",
        "kite_login": "/auth/kite/login",
        "data_status": "/data/status",
    }
