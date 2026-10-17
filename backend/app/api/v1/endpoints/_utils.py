"""Shared serialization and timing helpers for v1 endpoints."""

from __future__ import annotations

from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException


def json_safe(value: Any) -> Any:
    """Convert numerical Python/Pandas values into JSON-native structures."""
    if isinstance(value, pd.DataFrame):
        return [{str(k): json_safe(v) for k, v in row.items()} for row in value.reset_index().to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def run_core(operation: Any) -> tuple[Any, float]:
    """Run a core calculation and translate input/math failures to HTTP 422."""
    started = perf_counter()
    try:
        result = operation()
    except (ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_quant_input", "message": "Invalid input or calculation parameters."}) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail={"code": "quant_computation_failed", "message": "The calculation could not be completed."}) from exc
    return result, (perf_counter() - started) * 1000.0


def envelope(
    data: Any,
    elapsed_ms: float,
    input_shape: dict[str, list[int] | int],
    snapshot_timestamp: Any | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe response envelope with server-side latency and snapshot time."""
    metadata = {"execution_time_ms": float(elapsed_ms), "input_shape": input_shape}
    if snapshot_timestamp is not None:
        metadata["snapshot_timestamp"] = json_safe(snapshot_timestamp)
    return {"data": json_safe(data), "meta": metadata}
