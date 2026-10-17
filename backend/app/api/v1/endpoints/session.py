"""Exchange a Supabase user token for a short-lived HttpOnly API cookie."""

from __future__ import annotations

import os
from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.dependencies import AuthenticatedUser, require_current_user, verify_supabase_access_token
from app.api.v1.endpoints._utils import envelope

router = APIRouter()
_bearer = HTTPBearer(auto_error=True)
_SESSION_COOKIE = "moatcheck_session"


def _cookie_is_secure() -> bool:
    """Match cookie Secure behavior to the deployed HTTPS environment."""
    secure_setting = os.getenv("COOKIE_SECURE")
    if os.getenv("APP_ENV", "development").lower() == "production":
        return True
    if secure_setting is not None:
        return secure_setting.lower() in {"1", "true", "yes"}
    return os.getenv("APP_ENV", "development").lower() == "production"


@router.get("/session")
async def current_browser_session(user: AuthenticatedUser = Depends(require_current_user)) -> dict:
    """Return the authenticated browser identity without exposing its token."""
    return {"user_id": str(user.id), "email": user.email}


@router.post("/session")
async def create_browser_session(
    response: Response,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """Validate the submitted user token and store it in a secure HttpOnly cookie.

    The access token is accepted only in the HTTPS request Authorization header;
    JavaScript cannot read it after this exchange. The cookie expires after one
    hour and remains scoped to the API path.
    """
    started = perf_counter()
    try:
        user = await verify_supabase_access_token(credentials.credentials)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired access token") from exc
    response.set_cookie(
        key=_SESSION_COOKIE,
        value=credentials.credentials,
        max_age=3600,
        httponly=True,
        secure=_cookie_is_secure(),
        samesite="lax",
        path="/api",
    )
    elapsed = (perf_counter() - started) * 1000.0
    return envelope({"user_id": str(user.id), "session_expires_in": 3600}, elapsed, {})


@router.delete("/session")
async def clear_browser_session(response: Response) -> dict:
    """Clear the browser's API session cookie without revealing auth state."""
    response.delete_cookie(
        key=_SESSION_COOKIE,
        httponly=True,
        secure=_cookie_is_secure(),
        samesite="lax",
        path="/api",
    )
    return envelope({"session_cleared": True}, 0.0, {})
