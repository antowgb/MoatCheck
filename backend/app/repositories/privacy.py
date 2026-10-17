"""Server-side deletion of persisted quant data for an authenticated owner."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from supabase import AsyncClient

from app.repositories._base import RepositoryError


class PrivacyRepository:
    """Invoke the restricted, atomic owner-data deletion database function."""

    def __init__(self, client: AsyncClient) -> None:
        self.client = client

    async def delete_owner_data(self, owner_id: UUID) -> dict[str, int]:
        """Delete all persisted quant records linked to ``owner_id``.

        The caller must pass a UUID obtained from verified Supabase Auth, never
        a request-body value. The database function is executable only by the
        backend service role.
        """
        try:
            response: Any = await self.client.rpc(
                "delete_owner_quant_data", {"p_owner_id": str(owner_id)}
            ).execute()
        except Exception as exc:
            raise RepositoryError(f"Supabase owner-data deletion failed: {exc}") from exc

        error = getattr(response, "error", None)
        if error:
            raise RepositoryError(f"Supabase owner-data deletion failed: {error}")
        data = getattr(response, "data", None)
        if not isinstance(data, dict):
            raise RepositoryError("Supabase owner-data deletion returned an invalid response")
        expected = ("portfolios_deleted", "backtest_runs_deleted", "stress_tests_deleted")
        try:
            result = {key: int(data[key]) for key in expected}
        except (KeyError, TypeError, ValueError) as exc:
            raise RepositoryError("Supabase owner-data deletion returned invalid counts") from exc
        if any(value < 0 for value in result.values()):
            raise RepositoryError("Supabase owner-data deletion returned invalid counts")
        return result
