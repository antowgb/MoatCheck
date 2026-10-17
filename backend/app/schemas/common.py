"""Shared strict API response contracts."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class StrictModel(BaseModel):
    """Base model that rejects unknown request fields and coercion surprises."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ResponseMeta(BaseModel):
    """Execution metadata attached to every quant API response."""

    execution_time_ms: float = Field(ge=0)
    input_shape: dict[str, list[int] | int]
    snapshot_timestamp: datetime | None = None


class APIEnvelope(BaseModel):
    """Stable transport envelope; ``data`` contains endpoint-specific results."""

    data: dict[str, JsonValue]
    meta: ResponseMeta
