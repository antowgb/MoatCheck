"""Historical and reverse-stress result persistence model."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.models._base import PersistenceModel, require_finite_json


class StressTest(PersistenceModel):
    """Stored crisis replay or reverse-stress request and quantitative output."""

    portfolio_id: UUID | None = None
    stress_type: Literal["historical_crisis", "reverse_stress", "macro_scenario"]
    scenario_name: str = Field(min_length=1, max_length=160)
    scenario_start: date | None = None
    scenario_end: date | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)
    portfolio_return: float | None = None
    worst_drawdown: float | None = None
    minimum_shock_magnitude: float | None = Field(default=None, ge=0)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_dates_and_json(self) -> "StressTest":
        if (self.scenario_start is None) != (self.scenario_end is None):
            raise ValueError("scenario_start and scenario_end must be supplied together")
        if self.scenario_start and self.scenario_end and self.scenario_start > self.scenario_end:
            raise ValueError("scenario_start must be on or before scenario_end")
        scalar_values = (self.portfolio_return, self.worst_drawdown, self.minimum_shock_magnitude)
        if any(value is not None and not math.isfinite(value) for value in scalar_values):
            raise ValueError("stress metrics must be finite")
        for field_name in ("inputs", "results", "telemetry"):
            require_finite_json(getattr(self, field_name), path=field_name)
        return self
