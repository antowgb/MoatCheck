"""Owner-scoped persistence for recommendation previews and review state."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from supabase import AsyncClient

from app.models.advisory_recommendation import AdvisoryRecommendation
from app.repositories._base import SupabaseRepository


class AdvisoryRecommendationRepository(SupabaseRepository[AdvisoryRecommendation]):
    """Persist recommendations with explicit owner filters on every lookup."""

    table_name = "advisor_recommendations"
    model_type = AdvisoryRecommendation

    def __init__(self, client: AsyncClient) -> None:
        super().__init__(client)

    async def save(self, item: AdvisoryRecommendation) -> AdvisoryRecommendation:
        rows = await self._execute("insert", self.client.table(self.table_name).insert(item.to_record()))
        return self._model(rows[0]) if rows else item

    async def latest_for_portfolio(
        self, portfolio_id: UUID | str, owner_id: UUID | str
    ) -> AdvisoryRecommendation | None:
        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("portfolio_id", self._uuid(portfolio_id))
            .eq("owner_id", self._uuid(owner_id))
            .order("created_at", desc=True)
            .limit(1)
        )
        rows = await self._execute("owner-scoped latest recommendation", query)
        return self._model(rows[0]) if rows else None

    async def get_for_owner(
        self, recommendation_id: UUID | str, owner_id: UUID | str
    ) -> AdvisoryRecommendation | None:
        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("id", self._uuid(recommendation_id))
            .eq("owner_id", self._uuid(owner_id))
            .limit(1)
        )
        rows = await self._execute("owner-scoped recommendation", query)
        return self._model(rows[0]) if rows else None

    async def acknowledge(
        self, recommendation_id: UUID | str, owner_id: UUID | str
    ) -> AdvisoryRecommendation | None:
        now = datetime.now(timezone.utc)
        query = (
            self.client.table(self.table_name)
            .update({"status": "acknowledged", "acknowledged_at": now.isoformat()})
            .eq("id", self._uuid(recommendation_id))
            .eq("owner_id", self._uuid(owner_id))
            .select("*")
        )
        rows = await self._execute("owner-scoped acknowledgement", query)
        return self._model(rows[0]) if rows else None
