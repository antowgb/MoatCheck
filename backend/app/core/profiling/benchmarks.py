"""In-memory latency and empirical complexity tools for quant pipelines.

The module records measurements in Python objects only; it performs no file,
network, database, or web-framework operations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import wraps
from time import perf_counter_ns
from typing import Any, Callable, Iterable, Mapping, ParamSpec, TypeVar

import numpy as np
import pandas as pd

P = ParamSpec("P")
R = TypeVar("R")
_TELEMETRY_HISTORY_LIMIT = 1_000


@dataclass(frozen=True)
class BenchmarkInput:
    r"""Describe positional and keyword inputs generated for one benchmark size.

    The object separates input construction from measured execution, so the
    recorded runtime :math:`T(N)` covers the function under test rather than
    data setup. For example, an optimizer benchmark can supply a covariance
    DataFrame as its single positional argument. Construction is :math:`O(1)`
    excluding the referenced inputs; storage is :math:`O(A+K)` for positional
    arguments :math:`A` and keyword arguments :math:`K`.
    """

    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] | None = None


def _shape_summary(value: Any) -> dict[str, Any]:
    r"""Summarize a value's dimensions and storage without copying it in :math:`O(1)`.

    NumPy arrays and Pandas objects report their exact shape; a Python list
    reports its outer length. For example, a ``(100, 4)`` DataFrame is logged
    as that shape. Complexity: :math:`O(C)` for a DataFrame with :math:`C`
    columns and :math:`O(1)` additional space.
    """
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "nbytes": int(value.nbytes),
            "kind": "numpy.ndarray",
        }
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype) if isinstance(value, pd.Series) else None,
            "nbytes": int(value.memory_usage(index=True, deep=False).sum()
                           if isinstance(value, pd.DataFrame)
                           else value.memory_usage(index=True, deep=False)),
            "kind": type(value).__name__,
        }
    if isinstance(value, Mapping):
        return {
            "shape": None,
            "kind": type(value).__name__,
            "items": {str(key): _shape_summary(item) for key, item in value.items()},
        }
    if isinstance(value, (list, tuple)):
        return {"shape": [len(value)], "kind": type(value).__name__}
    return {"shape": [], "kind": type(value).__name__}


def _element_count(summary: Mapping[str, Any]) -> int:
    r"""Count represented input elements from a shape summary in :math:`O(A)`.

    A vector of length :math:`N` contributes :math:`N` elements and an
    :math:`N\times N` matrix contributes :math:`N^2`; scalars count as one.
    Complexity: :math:`O(A)` for :math:`A` nested argument summaries and
    :math:`O(1)` additional space.
    """
    if isinstance(summary.get("items"), Mapping):
        return sum(_element_count(item) for item in summary["items"].values())
    shape = summary.get("shape")
    if shape is None:
        return 0
    if not shape:
        return 1
    return int(np.prod(shape, dtype=np.int64))


def _record_decorator_telemetry(
    wrapper: Callable[..., Any],
    started_ns: int,
    input_shapes: dict[str, Any],
    *,
    result: Any = None,
    error: BaseException | None = None,
) -> None:
    r"""Append one call record and update latest latency in :math:`O(A+K)`.

    Each record contains the wall-clock duration and input/output shape
    summaries; e.g. a vector-to-scalar factor call records ``[N]`` and ``[]``.
    Complexity: :math:`O(A+K)` for argument summaries and :math:`O(1)` append
    time, with bounded history storage.
    """
    elapsed_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
    telemetry = {
        "latency_ms": float(elapsed_ms),
        "input_shapes": input_shapes,
        "output_shape": _shape_summary(result) if error is None else None,
        "success": error is None,
        "exception_type": type(error).__name__ if error is not None else None,
    }
    history = wrapper.telemetry_history
    history.append(telemetry)
    telemetry["peak_latency_ms"] = float(
        max((item["latency_ms"] for item in history), default=elapsed_ms)
    )
    wrapper.last_telemetry = telemetry


def profile_execution_time(func: Callable[P, R]) -> Callable[P, R]:
    r"""Decorate a function with latency and shape telemetry while preserving output.

    For each invocation it appends a bounded in-memory record with
    :math:`T_{ms}=10^{-6}(t_{end,ns}-t_{start,ns})`, argument shapes, output
    shape, success state, and peak observed latency. A numerical example is a
    function mapping a ``(252, 4)`` returns matrix to a ``(4,)`` statistic;
    the value returned by that function is returned unchanged. No logging I/O
    is performed: records are available on ``wrapped.last_telemetry`` and
    ``wrapped.telemetry_history``. Complexity overhead is :math:`O(A+K)` for
    shape summaries plus the wrapped function's cost, where :math:`A,K` count
    arguments and keyword arguments.
    """

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        input_shapes = {
            "args": [_shape_summary(value) for value in args],
            "kwargs": {key: _shape_summary(value) for key, value in kwargs.items()},
        }
        started_ns = perf_counter_ns()
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            _record_decorator_telemetry(wrapped, started_ns, input_shapes, error=exc)
            raise
        _record_decorator_telemetry(wrapped, started_ns, input_shapes, result=result)
        return result

    wrapped.telemetry_history = deque(maxlen=_TELEMETRY_HISTORY_LIMIT)  # type: ignore[attr-defined]
    wrapped.last_telemetry = None  # type: ignore[attr-defined]
    return wrapped


class ComplexityAnalyzer:
    r"""Estimate empirical Big-O growth by timing a function at increasing sizes.

    For measured sizes :math:`N_i` and median runtimes :math:`T_i`, this class
    fits :math:`\ln(T_i)=\alpha+\gamma\ln(N_i)+\epsilon_i` by ordinary least
    squares. The slope :math:`\gamma` estimates polynomial growth: matrix
    inversion is expected near :math:`O(N^3)` while a vector dot product is
    near :math:`O(N)`. An ``input_factory`` can provide workload data; for a
    simple numeric matrix example, an :math:`N\times N` matrix product grows
    roughly fourfold in work when :math:`N` doubles, while a vector pass
    roughly doubles. Benchmarking :math:`S` sizes with :math:`R` repeats costs
    the sum of the wrapped workloads plus :math:`O(S)` regression work; result
    storage is :math:`O(S)`.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        input_factory: Callable[[int], Any] | None = None,
        *,
        repeats: int = 3,
        warmups: int = 1,
    ) -> None:
        r"""Set the workload function, data factory, and repeat counts in :math:`O(1)`.

        If no factory is supplied, each size :math:`N` generates a float vector
        of length :math:`N` and passes it as the sole argument. A factory may
        return :class:`BenchmarkInput` for custom arguments. For example, a
        factor transform can receive a Pandas Series of length 1,000.
        Complexity: :math:`O(1)` time and configuration space.
        """
        if not callable(func):
            raise TypeError("func must be callable")
        if input_factory is not None and not callable(input_factory):
            raise TypeError("input_factory must be callable")
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
            raise ValueError("warmups must be a non-negative integer")
        self.func = func
        self.input_factory = input_factory
        self.repeats = repeats
        self.warmups = warmups

    def _make_input(self, size: int) -> BenchmarkInput:
        r"""Create one workload without including setup in timings, in :math:`O(N)`.

        A custom factory may return ``BenchmarkInput`` or any object to pass as
        one positional argument. With no factory, the default is ``linspace``
        of length :math:`N`. Complexity: :math:`O(N)` time and memory for the
        default vector, or the factory's cost for custom data.
        """
        if self.input_factory is None:
            return BenchmarkInput((np.linspace(0.0, 1.0, size, dtype=float),), {})
        generated = self.input_factory(size)
        if isinstance(generated, BenchmarkInput):
            return BenchmarkInput(generated.args, dict(generated.kwargs or {}))
        if (
            isinstance(generated, tuple)
            and len(generated) == 2
            and isinstance(generated[1], Mapping)
        ):
            raw_args = generated[0]
            args = tuple(raw_args) if isinstance(raw_args, (tuple, list)) else (raw_args,)
            return BenchmarkInput(args, dict(generated[1]))
        return BenchmarkInput((generated,), {})

    def analyze(self, sizes: Iterable[int]) -> dict[str, Any]:
        r"""Benchmark sizes and fit the log-time/log-size exponent :math:`\gamma`.

        The OLS estimate is
        :math:`\hat\gamma=\sum_i(x_i-\bar x)(y_i-\bar y)/
        \sum_i(x_i-\bar x)^2`, with :math:`x_i=\ln N_i` and
        :math:`y_i=\ln T_i`. For example, timings increasing by about eight
        times when matrix dimension doubles imply :math:`\gamma\approx3`.
        A single positive size is benchmarked but returns ``gamma=None`` and
        status ``insufficient_sizes`` because a slope cannot be identified.
        Complexity is the measured workload total plus :math:`O(S)` regression
        for :math:`S` sizes; telemetry storage is :math:`O(S)`.
        """
        try:
            requested = list(sizes)
        except TypeError as exc:
            raise TypeError("sizes must be an iterable of positive integers") from exc
        if not requested:
            raise ValueError("sizes must contain at least one positive integer")
        if any(
            isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1
            for size in requested
        ):
            raise ValueError("every benchmark size must be a positive integer")
        if len(set(int(size) for size in requested)) != len(requested):
            raise ValueError("benchmark sizes must be unique")
        ordered_sizes = sorted(int(size) for size in requested)

        observations: list[dict[str, Any]] = []
        for size in ordered_sizes:
            workload = self._make_input(size)
            for _ in range(self.warmups):
                self.func(*workload.args, **dict(workload.kwargs or {}))
            elapsed_ms: list[float] = []
            output_shape: dict[str, Any] | None = None
            input_shapes = {
                "args": [_shape_summary(value) for value in workload.args],
                "kwargs": {
                    key: _shape_summary(value)
                    for key, value in dict(workload.kwargs or {}).items()
                },
            }
            input_element_count = sum(
                _element_count(item) for item in input_shapes["args"]
            ) + sum(
                _element_count(item) for item in input_shapes["kwargs"].values()
            )
            for _ in range(self.repeats):
                started = perf_counter_ns()
                output = self.func(*workload.args, **dict(workload.kwargs or {}))
                duration_ms = max((perf_counter_ns() - started) / 1_000_000.0, 1e-9)
                elapsed_ms.append(float(duration_ms))
                output_shape = _shape_summary(output)
            median_ms = float(np.median(np.asarray(elapsed_ms, dtype=float)))
            observations.append(
                {
                    "size": size,
                    "median_latency_ms": median_ms,
                    "min_latency_ms": float(min(elapsed_ms)),
                    "max_latency_ms": float(max(elapsed_ms)),
                    "throughput_per_second": float(size / (median_ms / 1000.0)),
                    "input_element_count": int(input_element_count),
                    "input_elements_per_second": float(
                        input_element_count / (median_ms / 1000.0)
                    ),
                    "input_shapes": input_shapes,
                    "output_shape": output_shape,
                    "repeats": self.repeats,
                }
            )

        gamma: float | None = None
        r_squared: float | None = None
        if len(observations) >= 2:
            log_sizes = np.log(
                np.asarray([item["size"] for item in observations], dtype=float)
            )
            log_times = np.log(
                np.asarray([item["median_latency_ms"] for item in observations], dtype=float)
            )
            design = np.column_stack((np.ones(log_sizes.size), log_sizes))
            intercept, gamma_value = np.linalg.lstsq(design, log_times, rcond=None)[0]
            gamma = float(gamma_value)
            fitted = intercept + gamma * log_sizes
            residual_sum_squares = float(np.sum((log_times - fitted) ** 2))
            total_sum_squares = float(np.sum((log_times - log_times.mean()) ** 2))
            r_squared = (
                float(1.0 - residual_sum_squares / total_sum_squares)
                if total_sum_squares > 0.0
                else 1.0
            )
        return {
            "gamma": gamma,
            "empirical_exponent": gamma,
            "r_squared": r_squared,
            "status": "estimated" if gamma is not None else "insufficient_sizes",
            "measurements": observations,
            "peak_latency_ms": float(max(item["max_latency_ms"] for item in observations)),
            "throughput_per_second": observations[-1]["throughput_per_second"],
            "input_elements_per_second": observations[-1]["input_elements_per_second"],
            "input_sizes": [item["size"] for item in observations],
        }

    def benchmark(self, sizes: Iterable[int]) -> dict[str, Any]:
        r"""Alias for :meth:`analyze` with the same empirical OLS complexity estimate.

        This spelling reads naturally at call sites, for example
        ``ComplexityAnalyzer(fn).benchmark([50, 100, 200])``. Runtime remains
        the workload cost plus :math:`O(S)` regression for :math:`S` sizes.
        """
        return self.analyze(sizes)


def _series_input(size: int) -> BenchmarkInput:
    r"""Build a deterministic length-:math:`N` Pandas workload in :math:`O(N)`.

    The sinusoidal sample provides finite nonconstant data for factor or risk
    functions; for example, ``size=100`` produces a 100-row return Series.
    Complexity: :math:`O(N)` time and memory.
    """
    values = np.sin(np.linspace(0.0, 20.0, size, dtype=float)) * 0.01
    return BenchmarkInput((pd.Series(values, name="synthetic_returns"),), {})


def _covariance_input(size: int) -> BenchmarkInput:
    r"""Build a positive-definite :math:`N\times N` covariance DataFrame in :math:`O(N^2)`.

    A diagonal covariance is deterministic, positive definite, and suitable
    for an optimizer callback that accepts one covariance matrix. Complexity:
    :math:`O(N^2)` time and memory for matrix construction.
    """
    diagonal = np.linspace(0.01, 0.03, size, dtype=float)
    matrix = np.diag(diagonal)
    labels = [f"Asset_{index}" for index in range(size)]
    return BenchmarkInput((pd.DataFrame(matrix, index=labels, columns=labels),), {})


def benchmark_pipeline_phases(
    factor_func: Callable[..., Any],
    opt_func: Callable[..., Any],
    risk_func: Callable[..., Any],
    sizes: list[int],
) -> dict[str, Any]:
    r"""Compare empirical scaling of factor, optimizer, and risk pipeline phases.

    Each phase is measured using the same requested sizes and fitted by
    :math:`\ln(T)=\alpha+\gamma\ln(N)`. Factor and risk callbacks receive a
    deterministic Pandas Series of length :math:`N`; the optimizer callback
    receives an :math:`N\times N` positive-definite covariance DataFrame. A
    vector dot-product phase should trend toward :math:`O(N)`, while a matrix
    inversion callback should trend toward :math:`O(N^3)`. For a small example,
    doubling a matrix dimension can multiply inversion work by roughly eight,
    while doubling a one-pass factor operation roughly doubles its work.
    Setup data is generated outside each timed call; results include per-size
    throughput and peak latency. Complexity equals the summed benchmark
    workloads plus :math:`O(S)` regressions for :math:`S` sizes; input storage
    peaks at :math:`O(N_{max}^2)` for covariance workloads.
    """
    for name, func in (("factor", factor_func), ("optimization", opt_func), ("risk", risk_func)):
        if not callable(func):
            raise TypeError(f"{name} callback must be callable")
    if not isinstance(sizes, list) or not sizes:
        raise ValueError("sizes must be a non-empty list of positive integers")

    started = perf_counter_ns()
    phase_specs = {
        "factor": (factor_func, _series_input),
        "optimization": (opt_func, _covariance_input),
        "risk": (risk_func, _series_input),
    }
    phases: dict[str, dict[str, Any]] = {}
    for phase_name, (func, factory) in phase_specs.items():
        result = ComplexityAnalyzer(
            func, input_factory=factory, repeats=1, warmups=0
        ).analyze(sizes)
        phases[phase_name] = result
    total_ms = (perf_counter_ns() - started) / 1_000_000.0
    return {
        "phases": phases,
        "input_sizes": sorted(int(size) for size in sizes),
        "peak_latency_ms": float(
            max(result["peak_latency_ms"] for result in phases.values())
        ),
        "total_benchmark_latency_ms": float(total_ms),
        "throughput_per_second": {
            name: float(result["input_elements_per_second"])
            for name, result in phases.items()
        },
        "scale_units_per_second": {
            name: float(result["throughput_per_second"])
            for name, result in phases.items()
        },
        "input_array_shapes": {
            name: [measurement["input_shapes"] for measurement in result["measurements"]]
            for name, result in phases.items()
        },
    }


__all__ = [
    "BenchmarkInput",
    "ComplexityAnalyzer",
    "benchmark_pipeline_phases",
    "profile_execution_time",
]
