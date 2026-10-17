"""Persisted backtest result and tear-sheet model."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any
from uuid import UUID

from pydantic import Field, model_validator

from app.models._base import PersistenceModel, require_finite_json


class BacktestRun(PersistenceModel):
    """Historical backtest summary, net-PnL observations, and cost telemetry."""

    portfolio_id: UUID | None = None
    strategy_name: str = Field(min_length=1, max_length=120)
    start_date: date
    end_date: date
    tickers: list[str] = Field(min_length=1)
    net_pnl_series: list[float]
    observation_dates: list[date] | None = None
    turnover_series: list[float] = Field(default_factory=list)
    total_turnover: float = Field(ge=0)
    average_annual_turnover: float = Field(default=0, ge=0)
    total_transaction_cost: float = Field(default=0, ge=0)
    total_transaction_drag_bps: float = Field(default=0, ge=0)
    execution_cost_bps_of_traded_notional: float = Field(default=0, ge=0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_series(self) -> "BacktestRun":
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        if self.observation_dates is not None and len(self.observation_dates) != len(self.net_pnl_series):
            raise ValueError("observation_dates must align with net_pnl_series")
        if self.turnover_series and len(self.turnover_series) != len(self.net_pnl_series):
            raise ValueError("turnover_series must align with net_pnl_series")
        if any(not math.isfinite(x) for x in self.net_pnl_series + self.turnover_series):
            raise ValueError("PnL and turnover series must contain finite values")
        if any(x < 0 for x in self.turnover_series):
            raise ValueError("turnover_series cannot contain negative turnover")
        scalars = (
            self.total_turnover,
            self.average_annual_turnover,
            self.total_transaction_cost,
            self.total_transaction_drag_bps,
            self.execution_cost_bps_of_traded_notional,
        )
        if any(not math.isfinite(x) for x in scalars):
            raise ValueError("backtest scalar metrics must be finite")
        require_finite_json(self.metrics, path="metrics")
        require_finite_json(self.telemetry, path="telemetry")
        return self
