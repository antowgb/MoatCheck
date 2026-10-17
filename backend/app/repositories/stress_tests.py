"""Stress-test result persistence operations."""

from __future__ import annotations

from uuid import UUID

from supabase import AsyncClient

from app.models.stress_test import StressTest
from app.repositories._base import SupabaseRepository


class StressTestRepository(SupabaseRepository[StressTest]):
    """Save and retrieve historical, macro, and reverse-stress outputs."""

    table_name = "stress_tests"
    model_type = StressTest

    def __init__(self, client: AsyncClient) -> None:
        super().__init__(client)

    async def save(self, result: StressTest) -> StressTest:
        """Insert a stress scenario result as an immutable record."""
        rows = await self._execute(
            "insert", self.client.table(self.table_name).insert(result.to_record())
        )
        return self._model(rows[0]) if rows else result

    async def get(self, stress_test_id: UUID | str) -> StressTest | None:
        """Retrieve one saved stress result by UUID."""
        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("id", self._uuid(stress_test_id))
            .limit(1)
        )
        rows = await self._execute("select", query)
        return self._model(rows[0]) if rows else None

    async def delete(self, stress_test_id: UUID | str) -> bool:
        """Delete a stress result and report whether a row was removed."""
        query = (
            self.client.table(self.table_name)
            .delete()
            .eq("id", self._uuid(stress_test_id))
            .select("id")
        )
        return bool(await self._execute("delete", query))

    async def list_for_portfolio(self, portfolio_id: UUID | str, *, limit: int = 100) -> list[StressTest]:
        """List stress scenarios associated with a portfolio."""
        if isinstance(limit, bool) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("portfolio_id", self._uuid(portfolio_id))
            .order("created_at", desc=True)
            .limit(limit)
        )
        return [self._model(row) for row in await self._execute("list", query)]
