"""In-memory concurrency helpers for quantitative workloads.

The module has no external I/O and uses only Python executors plus NumPy and
Pandas. It makes worker failures explicit in returned values so one failed
simulation does not discard successful results from the same batch.
"""

from __future__ import annotations

import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from time import perf_counter_ns
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


def _invoke_task(payload: tuple[Callable[..., Any], tuple[Any, ...]]) -> Any:
    r"""Execute one task and convert ordinary exceptions to data in :math:`O(C)`.

    This top-level function is pickleable, which allows process workers to
    receive it through IPC. A task failure is returned as a mapping with
    ``task_error=True`` and exception details rather than terminating the
    entire executor batch. Error representation takes :math:`O(1)` extra space.
    """
    func, args = payload
    try:
        return func(*args)
    except Exception as exc:  # Return task failures without swallowing pool/system errors.
        return {
            "task_error": True,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }


def _validate_tasks(
    func: Callable[..., Any], tasks_args: list[tuple[Any, ...]], max_workers: int | None
) -> int | None:
    r"""Validate executor inputs and return normalized worker count in :math:`O(N)`.

    The task list is inspected for tuple argument bundles; no task executes
    here. Validation is :math:`O(K)` in task count and requires :math:`O(1)`
    additional memory. ``None`` preserves the executor's platform default.
    """
    if not callable(func):
        raise TypeError("func must be callable")
    if not isinstance(tasks_args, list):
        raise TypeError("tasks_args must be a list of argument tuples")
    if any(not isinstance(args, tuple) for args in tasks_args):
        raise TypeError("each tasks_args item must be a tuple")
    if max_workers is None:
        return None
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer or None")
    return max_workers


def _serial_map(func: Callable[..., Any], tasks_args: list[tuple[Any, ...]]) -> list[Any]:
    r"""Run a batch serially with the shared exception contract in :math:`O(K)` calls.

    The returned list preserves input order. Each successful task contributes
    its original result, and each raised exception contributes a failure
    mapping. Total time is the sum of task times; result storage is :math:`O(K)`.
    """
    return [_invoke_task((func, args)) for args in tasks_args]


def _safe_process_context() -> multiprocessing.context.BaseContext:
    r"""Select a process start method that avoids unsafe POSIX forks when possible.

    ``spawn`` starts a clean interpreter and avoids inheriting potentially
    locked BLAS/MKL thread state; ``forkserver`` is the next choice. Plain
    ``fork`` is retained only on systems that offer neither safe method.
    Selecting a context is :math:`O(1)` time and space.
    """
    methods = multiprocessing.get_all_start_methods()
    if "spawn" in methods:
        method = "spawn"
    elif "forkserver" in methods:
        method = "forkserver"
    else:
        method = methods[0]
    return multiprocessing.get_context(method)


def run_parallel_monte_carlo_multiprocessing(
    func: Callable[..., Any],
    tasks_args: list[tuple[Any, ...]],
    max_workers: int | None = None,
) -> list[Any]:
    r"""Execute CPU-bound simulation calls in separate processes to bypass the GIL.

    Each task is sent through process IPC and normally pickled; for payload
    size :math:`B`, serialization and transfer add :math:`O(B)` work per task.
    Pure Python CPU loops hold the CPython Global Interpreter Lock (GIL), so
    threads cannot execute those bytecodes concurrently, while processes own
    independent interpreters and can use multiple cores. A 100-portfolio
    Monte Carlo batch can therefore scale across cores when each portfolio
    simulation is substantial; tiny jobs may run slower because startup and
    pickle overhead dominate. Ideal speedup is bounded by Amdahl's law
    :math:`S(P)=1/((1-f)+f/P)`, further reduced by IPC. Complexity is workload
    time plus executor and serialization costs; output memory is :math:`O(K)`.

    Results remain in task order. If ``max_workers`` is one, execution falls
    back to serial calls and matches serial results exactly. Process workers
    require ``func`` and its arguments to be pickleable. Exceptions raised by
    an individual task are returned as ``task_error`` mappings.
    """
    workers = _validate_tasks(func, tasks_args, max_workers)
    if not tasks_args:
        return []
    if workers == 1:
        return _serial_map(func, tasks_args)
    payloads = [(func, args) for args in tasks_args]
    context = _safe_process_context()
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        return list(executor.map(_invoke_task, payloads))


def run_parallel_tasks_threading(
    func: Callable[..., Any],
    tasks_args: list[tuple[Any, ...]],
    max_workers: int | None = None,
) -> list[Any]:
    r"""Run independent tasks with a low-overhead thread pool.

    Threads share memory, so they avoid process pickle/IPC copies. They are
    useful when the task spends time in C-level NumPy kernels that release the
    GIL (many BLAS and ufunc operations do), or while waiting on non-CPU work.
    Pure Python CPU loops remain serialized by the GIL. For a 100-portfolio
    batch whose work is a large NumPy matrix product, threads may approach
    multi-core throughput without copying each matrix; for Python loops a
    process pool is usually more effective. Amdahl's bound is
    :math:`S(P)=1/((1-f)+f/P)`; measured speedup also reflects contention and
    memory bandwidth. Scheduling is :math:`O(K)` beyond task costs and result
    storage is :math:`O(K)`.

    The result list preserves task order, and per-task exceptions become
    ``task_error`` mappings. A one-worker call executes serially and returns
    the same values as the serial contract.
    """
    workers = _validate_tasks(func, tasks_args, max_workers)
    if not tasks_args:
        return []
    if workers == 1:
        return _serial_map(func, tasks_args)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_invoke_task, (func, args)) for args in tasks_args]
        return [future.result() for future in futures]


def _python_cpu_work(size: int) -> float:
    r"""Compute a deterministic Python loop workload in :math:`O(N)` time and :math:`O(1)` space."""
    total = 0.0
    for i in range(size):
        total += math.sqrt(i + 1.0)
    return total


def _numpy_vector_work(size: int) -> float:
    r"""Run a deterministic NumPy vector workload, :math:`O(N)` time/space.

    The ufunc and reduction execute in compiled C code; NumPy kernels commonly
    release the GIL during those operations, which makes this suitable for
    demonstrating thread-pool behavior.
    """
    values = np.arange(size, dtype=np.float64)
    return float(np.sqrt(values + 1.0).sum())


def _measure(
    func: Callable[..., Any], tasks_args: list[tuple[Any, ...]], max_workers: int
) -> tuple[list[Any], float]:
    r"""Measure one execution mode and return its results and elapsed milliseconds.

    Timing is :math:`O(1)` overhead around the workload and result storage is
    :math:`O(K)`. Input construction occurs outside the timed interval.
    """
    started = perf_counter_ns()
    results = func(tasks_args=tasks_args, max_workers=max_workers)
    return results, (perf_counter_ns() - started) / 1_000_000.0


def benchmark_concurrency_overhead(
    task_func: Callable[..., Any],
    data_sizes: list[int],
    max_workers: int = 4,
) -> dict[str, Any]:
    r"""Compare serial, thread-pool, and process-pool timings for two workload types.

    For each size :math:`N`, this function benchmarks the supplied callable in
    serial, threaded, and process modes. Call it once with a pure Python CPU
    function and once with a NumPy vector function to compare both workload
    types. It records elapsed milliseconds, speedup over serial, throughput
    (tasks/second), worker count, and input shape. A useful
    intuition is 100 portfolio simulations: heavy Python loops tend to benefit
    from processes, but large NumPy matrix operations often benefit from
    threads because OpenBLAS/NumPy runs in C and can release the GIL without
    transferring large input buffers. Process IPC pickle overhead can erase
    gains for small jobs. Amdahl's law gives the ideal bound
    :math:`S(P)=1/((1-f)+f/P)` for parallel fraction :math:`f`; this benchmark
    measures actual rather than ideal speedup. For ``D`` data sizes and ``K``
    tasks per size, runtime is the sum of measured workloads plus process
    startup/serialization; report storage is :math:`O(D)`.

    A task count of 100 is used per size to make the example comparable to a
    100-portfolio batch while keeping each workload tunable. ``task_func`` and
    its arguments must be pickleable for process measurements.
    """
    if not callable(task_func):
        raise TypeError("task_func must be callable")
    if not isinstance(data_sizes, list) or any(
        isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1
        for size in data_sizes
    ):
        raise ValueError("data_sizes must be a list of positive integers")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")

    breakdown: dict[str, Any] = {}
    for size_value in data_sizes:
        size = int(size_value)
        tasks_args = [(size,) for _ in range(100)]
        serial_started = perf_counter_ns()
        serial_results = _serial_map(task_func, tasks_args)
        serial_ms = (perf_counter_ns() - serial_started) / 1_000_000.0
        thread_started = perf_counter_ns()
        thread_results = run_parallel_tasks_threading(task_func, tasks_args, max_workers)
        thread_ms = (perf_counter_ns() - thread_started) / 1_000_000.0
        process_started = perf_counter_ns()
        process_results = run_parallel_monte_carlo_multiprocessing(task_func, tasks_args, max_workers)
        process_ms = (perf_counter_ns() - process_started) / 1_000_000.0
        # Ensure benchmark modes performed equivalent work, including ordering.
        if not (serial_results == thread_results == process_results):
            raise RuntimeError("concurrency benchmark modes returned inconsistent results")
        timings = {"serial": serial_ms, "thread_pool": thread_ms, "process_pool": process_ms}
        breakdown[str(size)] = {
            "task_func": getattr(task_func, "__qualname__", repr(task_func)),
            "tasks": len(tasks_args),
            "data_size": size,
            "worker_count": max_workers,
            "latency_ms": timings,
            "speedup_vs_serial": {
                "thread_pool": serial_ms / max(thread_ms, np.finfo(float).tiny),
                "process_pool": serial_ms / max(process_ms, np.finfo(float).tiny),
            },
            "throughput_tasks_per_second": {
                mode: len(tasks_args) / max(latency / 1000.0, np.finfo(float).tiny)
                for mode, latency in timings.items()
            },
            "input_shape": [len(tasks_args), size],
        }
    return {
        "task_func": getattr(task_func, "__qualname__", repr(task_func)),
        "worker_count": max_workers,
        "results_by_size": breakdown,
        "execution_time_ms": float(sum(max(item["latency_ms"].values()) for item in breakdown.values())),
        "python_version_gil_note": "CPU-bound Python bytecode is serialized by the CPython GIL; NumPy C kernels may release it.",
        "process_start_method": _safe_process_context().get_start_method(),
        "cpu_count": os.cpu_count(),
        "timing_table": pd.DataFrame(
            [
                {"data_size": int(size), "mode": mode, "latency_ms": latency}
                for size, values in breakdown.items()
                for mode, latency in values["latency_ms"].items()
            ]
        ).to_dict(orient="records"),
    }
