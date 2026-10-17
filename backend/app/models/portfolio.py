"""Portfolio configuration persistence model."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from app.models._base import PersistenceModel, require_finite_json


class PortfolioConfig(PersistenceModel):
    """Saved target allocation, risk limits, and optimizer configuration."""

    owner_id: UUID | None = None
    name: str = Field(min_length=1, max_length=120)
    tickers: list[str] = Field(min_length=1)
    target_weights: dict[str, float]
    risk_constraints: dict[str, Any] = Field(default_factory=dict)
    optimization_parameters: dict[str, Any] = Field(default_factory=dict)
    description: str | None = Field(default=None, max_length=2000)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("tickers")
    @classmethod
    def unique_nonempty_tickers(cls, tickers: list[str]) -> list[str]:
        if any(not item.strip() for item in tickers) or len(set(tickers)) != len(tickers):
            raise ValueError("tickers must be non-empty and unique")
        return tickers

    @field_validator("target_weights")
    @classmethod
    def finite_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        if not weights or any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("target_weights must be non-empty, finite, and non-negative")
        return weights

    @model_validator(mode="after")
    def validate_weights_cover_assets(self) -> "PortfolioConfig":
        if set(self.target_weights) != set(self.tickers):
            raise ValueError("target_weights must have exactly one entry for each ticker")
        if not math.isclose(sum(self.target_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("target_weights must sum to 1")
        require_finite_json(self.risk_constraints, path="risk_constraints")
        require_finite_json(self.optimization_parameters, path="optimization_parameters")
        return self
