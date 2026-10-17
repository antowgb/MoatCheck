"""Tests for serial, threaded, and process-based quantitative task execution."""

import numpy as np

from app.core.profiling.concurrency import (
    benchmark_concurrency_overhead,
    run_parallel_monte_carlo_multiprocessing,
    run_parallel_tasks_threading,
)


def _add(a, b):
    return a + b


def _maybe_fail(value):
    if value == 2:
        raise ValueError("simulated task error")
    return value * 3


def _numpy_sum(size):
    return float(np.arange(size, dtype=np.float64).sum())


def test_one_worker_falls_back_to_serial_results():
    tasks = [(1, 4), (10, -3), (0, 0)]
    assert run_parallel_monte_carlo_multiprocessing(_add, tasks, max_workers=1) == [5, 7, 0]
    assert run_parallel_tasks_threading(_add, tasks, max_workers=1) == [5, 7, 0]


def test_empty_task_lists_return_empty_without_creating_pool():
    assert run_parallel_monte_carlo_multiprocessing(_add, [], max_workers=2) == []
    assert run_parallel_tasks_threading(_add, [], max_workers=2) == []


def test_process_pool_returns_task_exceptions_as_results():
    results = run_parallel_monte_carlo_multiprocessing(
        _maybe_fail, [(1,), (2,), (3,)], max_workers=2
    )
    assert results[0] == 3
    assert results[1] == {
        "task_error": True,
        "exception_type": "ValueError",
        "message": "simulated task error",
    }
    assert results[2] == 9


def test_thread_pool_preserves_order_and_reports_failures():
    results = run_parallel_tasks_threading(_maybe_fail, [(3,), (2,), (1,)], max_workers=3)
    assert results[0] == 9
    assert results[1]["task_error"] is True
    assert results[2] == 3


def test_benchmark_reports_latency_speedup_throughput_and_shapes():
    result = benchmark_concurrency_overhead(_numpy_sum, [20, 40], max_workers=2)
    assert result["worker_count"] == 2
    assert set(result["results_by_size"]) == {"20", "40"}
    for item in result["results_by_size"].values():
        assert item["tasks"] == 100
        assert item["input_shape"] == [100, item["data_size"]]
        assert set(item["latency_ms"]) == {"serial", "thread_pool", "process_pool"}
        assert item["latency_ms"]["serial"] >= 0
        assert item["speedup_vs_serial"]["thread_pool"] > 0
        assert item["throughput_tasks_per_second"]["process_pool"] > 0
    assert len(result["timing_table"]) == 6


def test_benchmark_validates_callable_sizes_and_worker_count():
    import pytest

    with pytest.raises(TypeError):
        benchmark_concurrency_overhead("not callable", [1], max_workers=1)
    with pytest.raises(ValueError):
        benchmark_concurrency_overhead(_numpy_sum, [0], max_workers=1)
    with pytest.raises(ValueError):
        benchmark_concurrency_overhead(_numpy_sum, [1], max_workers=0)
