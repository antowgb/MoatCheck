"""Optional native kernels with safe NumPy fallbacks for quant workloads.

Compiled bindings can be supplied as ``app.core.profiling._cpp_extensions`` or
``app.core.profiling.cpp_extensions_native``. Import failure is expected and
does not prevent the pure NumPy implementations from working. This module does
not perform network, database, or application I/O.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import math
from time import perf_counter_ns
from typing import Any, Callable

import numpy as np


def _validate_real_array(arr: np.ndarray, name: str) -> np.ndarray:
    r"""Validate finite real numeric input in :math:`O(N)` time, :math:`O(1)` auxiliary space.

    Compiled kernels accept raw floating-point memory, so object and complex
    arrays are rejected before pointer-level access. A million-element vector
    scan is linear in its element count; no array is mutated.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise TypeError(f"{name} must have a real numeric dtype")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_positive_float(value: float, name: str, *, allow_zero: bool = False) -> float:
    r"""Return a finite positive (or optionally non-negative) scalar in :math:`O(1)`.

    Measurement noise variance must be positive for a finite Kalman gain
    denominator. Process noise may be zero for a constant latent state.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result < 0.0 or (result == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _numpy_moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    r"""Compute trailing means while preserving shape and initial NaNs.

    For each time index :math:`t\ge w-1`,
    :math:`MA_t=w^{-1}\sum_{j=0}^{w-1}x_{t-j}`. A cumulative-sum difference
    computes all rows in :math:`O(NC)` time and output space for ``C`` columns;
    the first ``window - 1`` values are NaN because a complete trailing window
    is not available. One-dimensional input returns one-dimensional output.
    """
    values = np.asarray(arr, dtype=np.float64)
    was_1d = values.ndim == 1
    matrix = values[:, None] if was_1d else values
    cumulative = np.vstack((np.zeros((1, matrix.shape[1])), np.cumsum(matrix, axis=0)))
    result = np.full(matrix.shape, np.nan, dtype=np.float64)
    result[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    return result[:, 0] if was_1d else result


def _numpy_kalman_loop(measurements: np.ndarray, R: float, Q: float) -> np.ndarray:
    r"""Run a scalar random-walk Kalman filter with fixed process/measurement noise.

    The recursion is :math:`P_t^-=P_{t-1}+Q`,
    :math:`K_t=P_t^-/(P_t^-+R)`, :math:`x_t=x_{t-1}+K_t(y_t-x_{t-1})`, and
    :math:`P_t=(1-K_t)P_t^-`. Initialization uses :math:`x_0=y_0` and
    :math:`P_0=R`; output length equals input length. Time is :math:`O(N)` and
    output space is :math:`O(N)`. A native loop can run pointer arithmetic
    without allocating Python scalar objects on every iteration.
    """
    values = np.asarray(measurements, dtype=np.float64)
    result = np.empty(values.shape, dtype=np.float64)
    estimate = float(values[0])
    variance = float(R)
    result[0] = estimate
    for index in range(1, values.size):
        predicted_variance = variance + Q
        gain = predicted_variance / (predicted_variance + R)
        estimate += gain * (float(values[index]) - estimate)
        variance = (1.0 - gain) * predicted_variance
        result[index] = estimate
    return result


class CppExtensionWrapper:
    r"""Discover optional native functions and dispatch contiguous buffers safely.

    Pybind11's ``py::array_t`` can expose NumPy memory through the buffer
    protocol without a copy when dtype, alignment, and contiguity match. Native
    loops can use raw ``double*`` pointer arithmetic rather than constructing
    Python objects per element; a binding may release the GIL with
    ``py::gil_scoped_release`` while it computes. A million-step Kalman loop
    illustrates the impact: Python performs a million interpreter-level loop
    iterations, while C++ performs the recurrence in a tight native loop, often
    several times faster depending on CPU and compiler. Discovery and dispatch
    metadata are :math:`O(1)`; fallback kernel work is :math:`O(N)`.

    ``native_module`` allows dependency injection for tests or custom bindings.
    By default the wrapper tries the two documented module names; absent
    binaries are a normal condition and select NumPy fallbacks.
    """

    _NATIVE_MODULES = (
        "app.core.profiling._cpp_extensions",
        "app.core.profiling.cpp_extensions_native",
    )

    def __init__(self, native_module: Any | None = None) -> None:
        r"""Initialize extension discovery and empty telemetry in :math:`O(1)` time/space.

        Importing a built ``.so`` or ``.pyd`` extension is the extension
        interface boundary. No binaries are required: if both imports fail,
        each public operation is handled by its equivalent NumPy recurrence.
        """
        self.native_module = native_module if native_module is not None else self._discover_native_module()
        self.last_telemetry: dict[str, Any] | None = None

    @classmethod
    def _discover_native_module(cls) -> Any | None:
        r"""Try known compiled-module names and return the first successful import.

        The bounded list requires :math:`O(K)` import attempts for ``K`` names;
        no generated files are inspected or created by this helper.
        """
        for module_name in cls._NATIVE_MODULES:
            try:
                module = importlib.import_module(module_name)
            except (ImportError, OSError):
                continue
            origin = getattr(module, "__file__", "") or ""
            if any(origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
                return module
        return None

    @property
    def extension_status(self) -> str:
        r"""Return ``cpp_native`` or ``numpy_fallback`` in :math:`O(1)` time/space."""
        return "cpp_native" if self.native_module is not None else "numpy_fallback"

    def _record(self, operation: str, started_ns: int, shape: tuple[int, ...], status: str) -> None:
        r"""Store latest operation telemetry in :math:`O(1)` time and space.

        Telemetry is in-memory only and includes latency, input shape, and the
        selected native/fallback implementation status.
        """
        self.last_telemetry = {
            "operation": operation,
            "latency_ms": (perf_counter_ns() - started_ns) / 1_000_000.0,
            "input_shape": shape,
            "extension_status": status,
        }

    def moving_average(self, arr: np.ndarray, window: int) -> np.ndarray:
        r"""Dispatch shape-preserving trailing mean using a C-contiguous native buffer.

        If the input is a sliced view such as ``values[::2]``, an explicit
        contiguous copy is made before passing a pointer to native code; this
        avoids interpreting a strided NumPy view as densely packed memory. The
        copy costs :math:`O(NC)` time/space. Native or fallback computation is
        :math:`O(NC)` time and output storage is :math:`O(NC)`.
        """
        source = _validate_real_array(arr, "arr")
        if source.ndim not in (1, 2) or source.shape[0] == 0:
            raise ValueError("arr must be a non-empty 1D or 2D array")
        if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)):
            raise ValueError("window must be a positive integer")
        window = int(window)
        if window < 1 or window > source.shape[0]:
            raise ValueError("window must be between 1 and the number of rows")
        contiguous = np.ascontiguousarray(source, dtype=np.float64)
        started = perf_counter_ns()
        status = "numpy_fallback"
        native_func = getattr(self.native_module, "c_fast_moving_average", None)
        if callable(native_func):
            output = np.asarray(native_func(contiguous, window), dtype=np.float64)
            status = "cpp_native"
        else:
            output = _numpy_moving_average(contiguous, window)
        if output.shape != source.shape:
            raise RuntimeError("moving-average implementation returned an unexpected shape")
        self._record("moving_average", started, tuple(source.shape), status)
        return output

    def kalman_filter(self, measurements: np.ndarray, R: float, Q: float) -> np.ndarray:
        r"""Dispatch a scalar Kalman loop with a contiguous float64 measurement buffer.

        A non-contiguous view is copied before native pointer access. The
        iterative recurrence is :math:`O(N)` time and the output is
        :math:`O(N)` memory; a native implementation can avoid Python
        interpreter overhead and release the GIL for parallel native work.
        """
        source = _validate_real_array(measurements, "measurements")
        if source.ndim != 1 or source.size == 0:
            raise ValueError("measurements must be a non-empty one-dimensional array")
        measurement_noise = _validate_positive_float(R, "R")
        process_noise = _validate_positive_float(Q, "Q", allow_zero=True)
        contiguous = np.ascontiguousarray(source, dtype=np.float64)
        started = perf_counter_ns()
        native_func = getattr(self.native_module, "c_fast_kalman_loop", None)
        if callable(native_func):
            output = np.asarray(native_func(contiguous, measurement_noise, process_noise), dtype=np.float64)
            status = "cpp_native"
        else:
            output = _numpy_kalman_loop(contiguous, measurement_noise, process_noise)
            status = "numpy_fallback"
        if output.shape != source.shape:
            raise RuntimeError("Kalman implementation returned an unexpected shape")
        self._record("kalman_filter", started, tuple(source.shape), status)
        return output

    def benchmark_speedup(
        self, operation: str, values: np.ndarray, *parameters: Any
    ) -> dict[str, Any]:
        r"""Compare native and NumPy fallback timings and return parity/speedup telemetry.

        Both implementations are timed on equivalent contiguous inputs. The
        ratio is :math:`T_{fallback}/T_{native}` (larger than one favors native).
        Benchmarking adds a second full pass, so total runtime is twice the
        kernel cost, :math:`O(N)` for the moving mean or Kalman recurrence; the
        fallback output and native output temporarily require :math:`O(N)`
        memory each. If no native implementation exists, the result reports a
        speedup of ``1.0`` and status ``numpy_fallback``.
        """
        if operation == "moving_average":
            if len(parameters) != 1:
                raise ValueError("moving_average benchmark requires a window parameter")
            window = int(parameters[0])
            fallback = lambda: _numpy_moving_average(np.ascontiguousarray(values, dtype=np.float64), window)
            actual = lambda: self.moving_average(values, window)
        elif operation == "kalman_filter":
            if len(parameters) != 2:
                raise ValueError("kalman_filter benchmark requires R and Q parameters")
            R, Q = map(float, parameters)
            fallback = lambda: _numpy_kalman_loop(np.ascontiguousarray(values, dtype=np.float64), R, Q)
            actual = lambda: self.kalman_filter(values, R, Q)
        else:
            raise ValueError("operation must be 'moving_average' or 'kalman_filter'")
        start = perf_counter_ns()
        fallback_result = fallback()
        fallback_ms = (perf_counter_ns() - start) / 1_000_000.0
        start = perf_counter_ns()
        actual_result = actual()
        actual_ms = (perf_counter_ns() - start) / 1_000_000.0
        parity = bool(np.allclose(fallback_result, actual_result, equal_nan=True, rtol=1e-12, atol=1e-12))
        selected_status = (
            self.last_telemetry["extension_status"]
            if self.last_telemetry is not None
            else self.extension_status
        )
        speedup = fallback_ms / max(actual_ms, np.finfo(float).tiny) if selected_status == "cpp_native" else 1.0
        return {
            "operation": operation,
            "extension_status": selected_status,
            "fallback_latency_ms": float(fallback_ms),
            "selected_latency_ms": float(actual_ms),
            "speedup_ratio": float(speedup),
            "numerical_parity": parity,
            "input_shape": tuple(values.shape),
        }


_DEFAULT_WRAPPER = CppExtensionWrapper()


def c_fast_moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    r"""Run the native or NumPy trailing moving average with shape-preserving output.

    The equation is :math:`MA_t=w^{-1}\sum_{j=0}^{w-1}x_{t-j}` with NaNs for
    unavailable initial windows. Pybind11's ``py::array_t`` buffer protocol
    enables zero-copy passing for aligned C-contiguous float64 inputs; a sliced
    input is first copied. One million observations require about 8 MB input
    and an additional 8 MB output. Runtime and output memory are :math:`O(NC)`.
    """
    return _DEFAULT_WRAPPER.moving_average(arr, window)


def c_fast_kalman_loop(measurements: np.ndarray, R: float, Q: float) -> np.ndarray:
    r"""Run a native or NumPy 1D Kalman filter loop over measurements.

    The filter uses :math:`P_t^-=P_{t-1}+Q`,
    :math:`K_t=P_t^-/(P_t^-+R)`, and :math:`x_t=x_{t-1}+K_t(y_t-x_{t-1})`.
    On a million measurements, a Python loop constructs roughly a million sets
    of Python scalar operations; native C++ uses contiguous ``double*`` loads
    and stores and may release the GIL, while retaining only the output vector.
    Pybind11 can pass the input without copying if it is already contiguous.
    Complexity is :math:`O(N)` time and :math:`O(N)` output memory.
    """
    return _DEFAULT_WRAPPER.kalman_filter(measurements, R, Q)


def cpp_extension_telemetry() -> dict[str, Any]:
    r"""Return latest default-wrapper telemetry and extension availability in :math:`O(1)`.

    The dictionary includes the last latency in milliseconds, input shape, and
    ``cpp_native`` versus ``numpy_fallback`` status. It creates no copies of
    numerical buffers and performs no external I/O.
    """
    return {
        "extension_status": _DEFAULT_WRAPPER.extension_status,
        "native_available": _DEFAULT_WRAPPER.native_module is not None,
        "last_operation": _DEFAULT_WRAPPER.last_telemetry,
    }


def benchmark_cpp_extension(
    operation: str, values: np.ndarray, *parameters: Any
) -> dict[str, Any]:
    r"""Measure one default-wrapper kernel against fallback and report speedup.

    See :meth:`CppExtensionWrapper.benchmark_speedup` for the timing ratio
    definition :math:`T_{fallback}/T_{native}`, numerical parity, and
    :math:`O(N)` runtime. This public helper is useful for profiling a
    million-step recursive filter before deciding whether a compiled extension
    is worthwhile.
    """
    return _DEFAULT_WRAPPER.benchmark_speedup(operation, values, *parameters)
