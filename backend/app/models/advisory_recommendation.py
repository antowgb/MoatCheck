"""Immutable advisory result and human review state."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.models._base import PersistenceModel, require_finite_json


class AdvisoryRecommendation(PersistenceModel):
    """Persist an owner-scoped, reproducible recommendation snapshot."""

    owner_id: UUID
    portfolio_id: UUID
    horizon_days: int = Field(ge=30, le=365)
    minimum_holding_period_days: int = Field(ge=30, le=365)
    snapshot_timestamp: datetime
    status: Literal["pending_review", "acknowledged"] = "pending_review"
    acknowledged_at: datetime | None = None
    recommendation: dict[str, Any]
    rationale: list[dict[str, Any]] = Field(default_factory=list)
    telemetry: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_record(self) -> "AdvisoryRecommendation":
        if self.minimum_holding_period_days > self.horizon_days:
            raise ValueError("minimum_holding_period_days cannot exceed horizon_days")
        if (self.status == "acknowledged") != (self.acknowledged_at is not None):
            raise ValueError("acknowledged_at must be set exactly when status is acknowledged")
        for name, value in (
            ("recommendation", self.recommendation),
            ("rationale", self.rationale),
            ("telemetry", self.telemetry),
        ):
            require_finite_json(value, path=name)
        return self
