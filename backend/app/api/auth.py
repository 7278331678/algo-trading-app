"""Daily Zerodha login.

Flow: open /auth/kite/login -> log in on Zerodha -> Zerodha redirects to
/auth/kite/callback with a request_token -> we exchange it for the day's access
token and keep it in Redis until 06:00 IST. The token is never returned by the API.
"""

from __future__ import annotations

import logging
import secrets
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config import get_settings

from .deps import get_redis, get_token_store

log = logging.getLogger(__name__)
router = APIRouter(prefix="/auth/kite", tags=["auth"])

_STATE_KEY = "kite:login_state:{}"
_STATE_TTL = 600  # 10 minutes to finish logging in


def _kite():
    from kiteconnect import KiteConnect

    s = get_settings()
    if not s.kite_configured:
        raise HTTPException(503, "KITE_API_KEY / KITE_API_SECRET are not set in .env")
    return KiteConnect(api_key=s.kite_api_key)


@router.get("/login")
def login() -> RedirectResponse:
    kite = _kite()
    state = secrets.token_urlsafe(24)
    get_redis().set(_STATE_KEY.format(state), "1", ex=_STATE_TTL)
    # redirect_params are echoed back by Zerodha on the callback: our CSRF check
    url = f"{kite.login_url()}&{urlencode({'redirect_params': urlencode({'state': state})})}"
    return RedirectResponse(url)


@router.get("/callback", response_class=HTMLResponse)
def callback(
    request_token: str | None = Query(None),
    status: str | None = Query(None),
    state: str | None = Query(None),
) -> HTMLResponse:
    if status != "success" or not request_token:
        raise HTTPException(400, "Zerodha login was not successful")
    r = get_redis()
    if not state or not r.delete(_STATE_KEY.format(state)):
        raise HTTPException(400, "login state missing or expired; start again from /auth/kite/login")

    kite = _kite()
    secret = get_settings().kite_api_secret.get_secret_value()
    try:
        data = kite.generate_session(request_token, api_secret=secret)
    except Exception as exc:  # noqa: BLE001
        log.error("kite session exchange failed: %s", exc.__class__.__name__)
        raise HTTPException(502, "could not exchange the request token with Zerodha") from exc

    session = get_token_store().save(user_id=data["user_id"], access_token=data["access_token"])
    log.info("kite login ok", extra={"extra_fields": {"user_id": session.user_id}})
    return HTMLResponse(
        "<html><body style='font-family:sans-serif;padding:2rem'>"
        f"<h2>Zerodha login successful</h2><p>User {session.user_id}. "
        f"Token valid until {session.expires_at}.</p>"
        "<p>You can close this tab.</p></body></html>"
    )


@router.get("/status")
def status_() -> dict:
    session = get_token_store().load()
    return {"logged_in": bool(session), **(session.public() if session else {})}


@router.post("/logout")
def logout() -> dict:
    get_token_store().clear()
    return {"logged_in": False}
