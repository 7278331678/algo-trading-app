"""Shared singletons for request handlers."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.core.market_clock import MarketClock
from app.state.token_store import TokenStore


@lru_cache
def get_clock() -> MarketClock:
    return MarketClock()


@lru_cache
def get_redis():
    import redis

    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


@lru_cache
def get_engine():
    from sqlalchemy import create_engine

    return create_engine(get_settings().db_url, pool_pre_ping=True)


def get_token_store() -> TokenStore:
    return TokenStore(get_redis(), get_clock())
