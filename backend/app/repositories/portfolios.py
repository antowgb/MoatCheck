"""Portfolio configuration persistence operations."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from supabase import AsyncClient

from app.models.portfolio import PortfolioConfig
from app.repositories._base import SupabaseRepository


class PortfolioRepository(SupabaseRepository[PortfolioConfig]):
    """Create, update, retrieve, list, and delete saved portfolio configs."""

    table_name = "portfolios"
    model_type = PortfolioConfig

    def __init__(self, client: AsyncClient) -> None:
        super().__init__(client)

    async def save(self, portfolio: PortfolioConfig) -> PortfolioConfig:
        """Insert or replace a portfolio configuration by its stable UUID."""
        portfolio = portfolio.model_copy(update={"updated_at": datetime.now(timezone.utc)})
        query = self.client.table(self.table_name).upsert(
            portfolio.to_record(), on_conflict="id"
        )
        rows = await self._execute("upsert", query)
        return self._model(rows[0]) if rows else portfolio

    async def get(self, portfolio_id: UUID | str) -> PortfolioConfig | None:
        """Retrieve one saved configuration, or ``None`` when it does not exist."""
        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("id", self._uuid(portfolio_id))
            .limit(1)
        )
        rows = await self._execute("select", query)
        return self._model(rows[0]) if rows else None

    async def get_for_owner(
        self, portfolio_id: UUID | str, owner_id: UUID | str
    ) -> PortfolioConfig | None:
        """Retrieve a portfolio only when it belongs to the verified owner."""
        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("id", self._uuid(portfolio_id))
            .eq("owner_id", self._uuid(owner_id))
            .limit(1)
        )
        rows = await self._execute("owner-scoped select", query)
        return self._model(rows[0]) if rows else None

    async def list_for_owner(self, owner_id: UUID | str, *, limit: int = 100) -> list[PortfolioConfig]:
        """List configurations belonging to one owner, newest first."""
        if isinstance(limit, bool) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("owner_id", self._uuid(owner_id))
            .order("updated_at", desc=True)
            .limit(limit)
        )
        return [self._model(row) for row in await self._execute("list", query)]

    async def delete_for_owner(self, portfolio_id: UUID | str, owner_id: UUID | str) -> bool:
        """Delete only a portfolio belonging to the verified owner."""
        query = (
            self.client.table(self.table_name)
            .delete()
            .eq("id", self._uuid(portfolio_id))
            .eq("owner_id", self._uuid(owner_id))
            .select("id")
        )
        return bool(await self._execute("owner-scoped delete", query))

    async def delete(self, portfolio_id: UUID | str) -> bool:
        """Delete a saved configuration and return whether a row was removed."""
        query = (
            self.client.table(self.table_name)
            .delete()
            .eq("id", self._uuid(portfolio_id))
            .select("id")
        )
        return bool(await self._execute("delete", query))
