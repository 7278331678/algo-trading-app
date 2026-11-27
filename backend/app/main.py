"""FastAPI entry point."""

from __future__ import annotations

from fastapi import FastAPI

from app.api import auth, health
from app.config import get_settings
from app.logging_setup import setup_logging

settings = get_settings()
setup_logging(settings.log_level)

app = FastAPI(
    title="Algo Trading App",
    version="0.1.0",
    description="Multi-agent intraday trading system for Zerodha (Phase 0).",
)
app.include_router(health.router)
app.include_router(auth.router)


@app.get("/")
def root() -> dict:
    return {
        "name": "algo-trading-app",
        "app_mode": settings.app_mode.value,
        "trading_mode": settings.trading_mode.value,
        "docs": "/docs",
        "kite_login": "/auth/kite/login",
    }
