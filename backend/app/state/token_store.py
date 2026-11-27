"""Stores the daily Kite access token in Redis until it expires (06:00 IST)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime

from app.core.market_clock import MarketClock

_KEY = "kite:session"


@dataclass
class KiteSession:
    user_id: str
    access_token: str
    issued_at: str   # ISO, IST
    expires_at: str  # ISO, IST

    def public(self) -> dict:
        """Safe view for the API: never returns the token itself."""
        return {"user_id": self.user_id, "issued_at": self.issued_at, "expires_at": self.expires_at}


class TokenStore:
    def __init__(self, redis_client, clock: MarketClock) -> None:
        self.r = redis_client
        self.clock = clock

    def save(self, user_id: str, access_token: str, issued_at: datetime | None = None) -> KiteSession:
        issued = self.clock.to_ist(issued_at) if issued_at else self.clock.now()
        expires = self.clock.kite_token_expiry(issued)
        session = KiteSession(user_id, access_token, issued.isoformat(), expires.isoformat())
        ttl = max(1, int((expires - issued).total_seconds()))
        self.r.set(_KEY, json.dumps(asdict(session)), ex=ttl)
        return session

    def load(self) -> KiteSession | None:
        raw = self.r.get(_KEY)
        return KiteSession(**json.loads(raw)) if raw else None

    def clear(self) -> None:
        self.r.delete(_KEY)
