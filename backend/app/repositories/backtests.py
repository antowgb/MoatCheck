"""Backtest result and tear-sheet persistence operations."""

from __future__ import annotations

from uuid import UUID

from supabase import AsyncClient

from app.models.backtest_run import BacktestRun
from app.repositories._base import SupabaseRepository


class BacktestRepository(SupabaseRepository[BacktestRun]):
    """Save immutable runs and retrieve full results or tear-sheet summaries."""

    table_name = "backtest_runs"
    model_type = BacktestRun

    def __init__(self, client: AsyncClient) -> None:
        super().__init__(client)

    async def save_run(self, run: BacktestRun) -> BacktestRun:
        """Insert one completed run; run records are not silently overwritten."""
        rows = await self._execute(
            "insert", self.client.table(self.table_name).insert(run.to_record())
        )
        return self._model(rows[0]) if rows else run

    async def update_run(self, run: BacktestRun) -> BacktestRun:
        """Update an existing run record by ID, preserving its creation time."""
        payload = run.to_record()
        payload.pop("id", None)
        payload.pop("created_at", None)
        query = (
            self.client.table(self.table_name)
            .update(payload)
            .eq("id", str(run.id))
            .select("*")
        )
        rows = await self._execute("update", query)
        return self._model(rows[0]) if rows else run

    async def get_run(self, run_id: UUID | str) -> BacktestRun | None:
        """Retrieve one complete backtest result by UUID."""
        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("id", self._uuid(run_id))
            .limit(1)
        )
        rows = await self._execute("select", query)
        return self._model(rows[0]) if rows else None

    async def get_tear_sheet(self, run_id: UUID | str) -> dict | None:
        """Return compact headline metrics and execution telemetry for a run."""
        run = await self.get_run(run_id)
        if run is None:
            return None
        return {
            "run_id": str(run.id),
            "portfolio_id": str(run.portfolio_id) if run.portfolio_id else None,
            "strategy_name": run.strategy_name,
            "start_date": run.start_date,
            "end_date": run.end_date,
            "metrics": run.metrics,
            "total_turnover": run.total_turnover,
            "total_transaction_cost": run.total_transaction_cost,
            "total_transaction_drag_bps": run.total_transaction_drag_bps,
            "execution_cost_bps_of_traded_notional": run.execution_cost_bps_of_traded_notional,
            "telemetry": run.telemetry,
        }

    async def list_for_portfolio(self, portfolio_id: UUID | str, *, limit: int = 100) -> list[BacktestRun]:
        """List the most recent backtests for a portfolio."""
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

    async def delete_run(self, run_id: UUID | str) -> bool:
        """Delete a stored run and report whether a row was returned."""
        query = (
            self.client.table(self.table_name)
            .delete()
            .eq("id", self._uuid(run_id))
            .select("id")
        )
        return bool(await self._execute("delete", query))
