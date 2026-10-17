"""Owner-authorized storage operations for portfolio tracking records."""

from __future__ import annotations

from uuid import UUID

from supabase import AsyncClient

from app.models.portfolio_tracking import PortfolioHolding, PortfolioSettings
from app.repositories._base import SupabaseRepository


class PortfolioTrackingRepository(SupabaseRepository[PortfolioHolding]):
    """Read and replace holdings/settings after the API verifies portfolio ownership."""

    table_name = "portfolio_holdings"
    model_type = PortfolioHolding

    def __init__(self, client: AsyncClient) -> None:
        super().__init__(client)

    async def list_holdings(self, portfolio_id: UUID | str) -> list[PortfolioHolding]:
        query = self.client.table("portfolio_holdings").select("*").eq("portfolio_id", self._uuid(portfolio_id)).order("ticker")
        return [self._model(row) for row in await self._execute("list holdings", query)]

    async def replace_holdings(self, portfolio_id: UUID | str, holdings: list[PortfolioHolding]) -> list[PortfolioHolding]:
        """Atomically replace the owner-authorized portfolio's complete holdings set."""
        portfolio_key = self._uuid(portfolio_id)
        payload = [
            {
                "ticker": item.ticker,
                "quantity": item.quantity,
                "average_buy_price": item.average_buy_price,
                "currency": item.currency,
                "acquired_at": item.acquired_at.isoformat(),
            }
            for item in holdings
        ]
        query = self.client.rpc(
            "replace_portfolio_holdings",
            {"p_portfolio_id": portfolio_key, "p_holdings": payload},
        )
        saved = await self._execute("replace holdings", query)
        return [self._model(row) for row in saved]

    async def get_settings(self, portfolio_id: UUID | str) -> PortfolioSettings:
        query = self.client.table("portfolio_settings").select("*").eq("portfolio_id", self._uuid(portfolio_id)).limit(1)
        rows = await self._execute("get settings", query)
        return PortfolioSettings.model_validate(rows[0]) if rows else PortfolioSettings(portfolio_id=self._uuid(portfolio_id))

    async def save_settings(self, settings: PortfolioSettings) -> PortfolioSettings:
        query = self.client.table("portfolio_settings").upsert(
            settings.to_record(), on_conflict="portfolio_id"
        )
        rows = await self._execute("save settings", query)
        return PortfolioSettings.model_validate(rows[0]) if rows else settings
