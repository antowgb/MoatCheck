"""Privacy controls for authenticated MoatCheck data."""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import AuthenticatedUser, require_current_user
from app.db.supabase import get_async_supabase_client
from app.repositories._base import RepositoryError
from app.repositories.privacy import PrivacyRepository
from app.api.v1.endpoints._utils import envelope

router = APIRouter()


@router.delete("/privacy/data")
async def delete_my_persisted_data(
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Delete saved portfolios, linked backtests, stress runs and recommendations."""
    started = perf_counter()
    try:
        client = await get_async_supabase_client()
        deleted = await PrivacyRepository(client).delete_owner_data(user.id)
    except (RepositoryError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="Could not delete saved data") from exc
    elapsed_ms = (perf_counter() - started) * 1000.0
    return envelope(
        {"deleted": deleted, "message": "Persisted portfolio data was deleted."},
        elapsed_ms,
        {"owner_scope": 1},
    )
