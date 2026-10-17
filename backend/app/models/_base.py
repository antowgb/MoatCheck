"""Shared JSON-safe persistence model primitives."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from numbers import Real
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class PersistenceModel(BaseModel):
    """Base for strict database row models with explicit JSON conversion."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_record(self, *, exclude_none: bool = True) -> dict[str, Any]:
        """Return a JSON-compatible mapping suitable for PostgREST inserts."""
        return self.model_dump(mode="json", exclude_none=exclude_none)


def require_finite_json(value: Any, *, path: str = "value") -> None:
    """Reject non-finite numeric values recursively before JSON persistence."""
    if isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(value):
        raise ValueError(f"{path} must contain only finite numbers")
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_json(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_json(item, path=f"{path}[{index}]")
