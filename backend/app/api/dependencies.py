"""HTTP authentication context for owner-scoped API operations."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.db.supabase import get_async_supabase_client

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    """Identity verified by Supabase Auth, never supplied as an owner ID."""

    id: UUID
    email: str | None = None


async def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session_token: str | None = Cookie(default=None, alias="moatcheck_session"),
) -> AuthenticatedUser | None:
    """Validate a bearer token or HttpOnly browser session when present.

    Invalid bearer credentials fail closed with HTTP 401. A backend-only
    elevated Supabase API key asks Supabase Auth to verify the presented user
    token; the resulting UUID is then used in explicit owner-scoped queries.
    """
    token = credentials.credentials if credentials is not None else session_token
    if token is None:
        return None
    try:
        return await verify_supabase_access_token(token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def verify_supabase_access_token(access_token: str) -> AuthenticatedUser:
    """Ask Supabase Auth to validate a user access token and return its identity.

    The token is never treated as an owner identifier without successful
    server-side validation. This helper is shared by API dependencies and the
    browser-session exchange endpoint.
    """
    client = await get_async_supabase_client()
    response = await client.auth.get_user(access_token)
    user = getattr(response, "user", None)
    if user is None:
        raise ValueError("Supabase Auth returned no user")
    return AuthenticatedUser(id=UUID(str(user.id)), email=getattr(user, "email", None))


async def require_current_user(
    user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> AuthenticatedUser:
    """Require verified Supabase identity for a protected API operation."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
