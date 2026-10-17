"""Memory layout and vectorization helpers for the pure quant compute engine.

This module performs in-memory numerical work only. It does not access files,
networks, databases, or web frameworks. Its helpers make array storage order
visible because cache locality can materially affect large portfolio jobs.
"""

from __future__ import annotations

from time import perf_counter_ns
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg.blas import dsyrk


def _validate_array(arr: np.ndarray, name: str = "arr") -> np.ndarray:
    """Validate a real numeric NumPy array without modifying it.

    Validation is linear in the number of elements, :math:`O(N)`, because
    finiteness is checked elementwise; extra space is :math:`O(1)` aside from
    NumPy's temporary boolean result. The pure numerical contract excludes
    object, complex, and non-finite data from downstream BLAS operations.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise TypeError(f"{name} must have a real numeric dtype")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def ensure_c_contiguous(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` in row-major C-contiguous layout, without in-place edits.

    In C order, the last index changes fastest. Iterating across each row
    therefore accesses nearby values and uses cache lines efficiently. A
    typical cache line is 64 bytes (eight float64 values); for a 1000x1000
    float64 matrix, row neighbors are 8 bytes apart, whereas stepping down a
    column jumps 8000 bytes. C conversion costs :math:`O(N)` time and up to
    :math:`O(N)` additional space when a copy is needed; already contiguous
    input is returned as-is. Conversion never mutates the original array.
    """
    _validate_array(arr)
    return np.ascontiguousarray(arr)


def ensure_f_contiguous(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` in column-major Fortran-contiguous layout.

    In Fortran order, the first index changes fastest, favoring column-wise
    cross-sectional operations and BLAS routines configured for that layout.
    The same 64-byte cache-line reasoning applies: contiguous neighboring
    values can be consumed together and often vectorized with SIMD. Conversion
    costs :math:`O(N)` time and up to :math:`O(N)` extra space if needed; a
    suitable input is returned as-is, and the source is never mutated.
    """
    _validate_array(arr)
    return np.asfortranarray(arr)


def memory_layout_telemetry(arr: np.ndarray) -> dict[str, Any]:
    """Describe array layout, strides, size, and telemetry collection latency.

    The report itself is constant-time, :math:`O(1)`, and allocates only a
    small metadata dictionary, :math:`O(1)`. For example, on a 1000x1000
    float64 matrix, C order has strides ``(8000, 8)`` and F order ``(8, 8000)``;
    choosing the matching traversal keeps consecutive reads within cache lines.
    """
    started = perf_counter_ns()
    _validate_array(arr)
    flags = arr.flags
    report = {
        "shape": tuple(arr.shape),
        "ndim": arr.ndim,
        "dtype": str(arr.dtype),
        "nbytes": int(arr.nbytes),
        "strides_bytes": tuple(int(x) for x in arr.strides),
        "c_contiguous": bool(flags.c_contiguous),
        "f_contiguous": bool(flags.f_contiguous),
        "aligned": bool(flags.aligned),
    }
    report["latency_ms"] = (perf_counter_ns() - started) / 1_000_000
    return report


def _timed_sum(matrix: np.ndarray, axis: int, iterations: int) -> tuple[float, float]:
    r"""Time repeated NumPy reductions and return average milliseconds/checksum.

    Each reduction is :math:`O(RC)` in time and :math:`O(\max(R,C))` in output
    space; the loop performs ``iterations`` reductions. The checksum keeps the
    result observable while NumPy executes its vectorized, SIMD-capable kernel.
    """
    start = perf_counter_ns()
    checksum = 0.0
    for _ in range(iterations):
        checksum += float(np.sum(matrix, axis=axis, dtype=np.float64).sum())
    elapsed_ms = (perf_counter_ns() - start) / 1_000_000
    return elapsed_ms / iterations, checksum


def benchmark_memory_layout_impact(
    matrix_shape: tuple[int, int], iterations: int = 100
) -> dict[str, Any]:
    r"""Compare row and column reductions over C- and F-contiguous matrices.

    Returns per-operation latency, throughput, layout flags/strides, checksums,
    and speedup ratios. Runtime is :math:`O(I\,R\,C)` for ``I`` iterations and
    :math:`O(RC)` memory. The 1000x1000 float64 intuition is concrete: a row
    traversal reads 8-byte neighbors, while column traversal on C-order jumps
    8000 bytes between values, often wasting most of each 64-byte cache line.
    A C-order layout usually favors row reductions; F order usually favors
    columns, though actual results depend on hardware and NumPy/BLAS builds.
    """
    if (not isinstance(matrix_shape, tuple) or len(matrix_shape) != 2
            or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v <= 0
                   for v in matrix_shape)):
        raise ValueError("matrix_shape must be a pair of positive integers")
    if not isinstance(iterations, (int, np.integer)) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")

    rng = np.random.default_rng(0)
    base = rng.standard_normal(matrix_shape)
    c_matrix = np.ascontiguousarray(base)
    f_matrix = np.asfortranarray(base)
    measurements: dict[str, dict[str, float]] = {}
    checksums: dict[str, float] = {}
    for layout, matrix in (("c", c_matrix), ("f", f_matrix)):
        for axis, direction in ((1, "row_wise"), (0, "column_wise")):
            latency, checksum = _timed_sum(matrix, axis, int(iterations))
            key = f"{direction}_{layout}"
            elements = int(np.prod(matrix_shape))
            measurements[key] = {
                "latency_ms": latency,
                "throughput_elements_per_second": elements / max(latency / 1000, np.finfo(float).tiny),
            }
            checksums[key] = checksum

    row_speedup = measurements["row_wise_f"]["latency_ms"] / max(
        measurements["row_wise_c"]["latency_ms"], np.finfo(float).tiny
    )
    col_speedup = measurements["column_wise_c"]["latency_ms"] / max(
        measurements["column_wise_f"]["latency_ms"], np.finfo(float).tiny
    )
    return {
        "matrix_shape": tuple(int(x) for x in matrix_shape),
        "iterations": int(iterations),
        "measurements": measurements,
        "speedup_ratio": {"row_wise_c_over_f": row_speedup, "column_wise_f_over_c": col_speedup},
        "layouts": {"c": memory_layout_telemetry(c_matrix), "f": memory_layout_telemetry(f_matrix)},
        "checksums": checksums,
        "peak_latency_ms": max(item["latency_ms"] for item in measurements.values()),
    }


def vectorized_matrix_operations(
    returns_matrix: np.ndarray, rolling_window: int | None = None
) -> dict[str, Any]:
    """Compute sample covariance and per-asset rolling means with vectorized kernels.

    Covariance is calculated as :math:`X_c^T X_c/(T-1)` using SciPy's symmetric
    rank-k BLAS kernel; rolling means use a ``sliding_window_view`` and NumPy
    reduction, so the window tensor is a zero-copy view when supported. Runtime
    is :math:`O(TA^2 + TAw)` and outputs require :math:`O(A^2 + TA)` space, with
    ``T`` observations, ``A`` assets, and window ``w``. For a 1000x1000 input,
    a layout aligned with the reduction axis improves spatial locality; SIMD
    lets compiled kernels process adjacent floating point values efficiently.
    """
    started = perf_counter_ns()
    _validate_array(returns_matrix, "returns_matrix")
    if returns_matrix.ndim != 2:
        raise ValueError("returns_matrix must be two-dimensional (observations, assets)")
    n_obs, n_assets = returns_matrix.shape
    if n_obs < 2 or n_assets < 1:
        raise ValueError("returns_matrix must contain at least two observations and one asset")
    window = min(20, n_obs) if rolling_window is None else rolling_window
    if not isinstance(window, (int, np.integer)) or isinstance(window, bool) or not 1 <= window <= n_obs:
        raise ValueError("rolling_window must be an integer between 1 and number of observations")

    values = np.asarray(returns_matrix, dtype=np.float64)
    centered_f = np.asfortranarray(values - values.mean(axis=0))
    covariance = dsyrk(1.0 / (n_obs - 1), centered_f, trans=1, lower=0)
    covariance = np.triu(covariance) + np.triu(covariance, 1).T
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=int(window), axis=0)
    rolling_mean = windows.mean(axis=-1)

    frame = pd.DataFrame(values)
    latency = (perf_counter_ns() - started) / 1_000_000
    return {
        "covariance_matrix": covariance,
        "rolling_mean": rolling_mean,
        "rolling_window": int(window),
        "input_shape": tuple(returns_matrix.shape),
        "output_shapes": {"covariance_matrix": covariance.shape, "rolling_mean": rolling_mean.shape},
        "memory_strides": {
            "input_bytes": tuple(int(x) for x in returns_matrix.strides),
            "rolling_window_bytes": tuple(int(x) for x in windows.strides),
        },
        "flags": {
            "input_c_contiguous": bool(returns_matrix.flags.c_contiguous),
            "input_f_contiguous": bool(returns_matrix.flags.f_contiguous),
            "working_frame_shape": frame.shape,
        },
        "rolling_windows_zero_copy": bool(np.shares_memory(windows, values)),
        "latency_ms": latency,
    }
