"""Tests for array layout conversion and vectorized profiling helpers."""

import numpy as np
import pytest

from app.core.profiling.memory_layout import (
    benchmark_memory_layout_impact,
    ensure_c_contiguous,
    ensure_f_contiguous,
    memory_layout_telemetry,
    vectorized_matrix_operations,
)


def test_one_dimensional_arrays_are_both_c_and_f_contiguous():
    values = np.arange(8, dtype=np.float64)
    assert values.flags.c_contiguous and values.flags.f_contiguous
    assert ensure_c_contiguous(values) is values
    assert ensure_f_contiguous(values) is values


def test_non_contiguous_slice_conversions_copy_without_mutating_source():
    base = np.arange(48, dtype=np.float64).reshape(6, 8)
    view = base[:, ::2]
    original = view.copy()
    assert not view.flags.c_contiguous

    c_array = ensure_c_contiguous(view)
    f_array = ensure_f_contiguous(view)
    assert c_array.flags.c_contiguous
    assert f_array.flags.f_contiguous
    np.testing.assert_array_equal(c_array, original)
    np.testing.assert_array_equal(f_array, original)
    np.testing.assert_array_equal(view, original)
    assert not np.shares_memory(c_array, view)
    assert not np.shares_memory(f_array, view)


def test_benchmark_compares_both_orders_and_reports_telemetry():
    result = benchmark_memory_layout_impact((24, 16), iterations=2)
    assert result["matrix_shape"] == (24, 16)
    assert set(result["measurements"]) == {
        "row_wise_c", "row_wise_f", "column_wise_c", "column_wise_f"
    }
    assert result["layouts"]["c"]["c_contiguous"]
    assert result["layouts"]["f"]["f_contiguous"]
    assert result["speedup_ratio"]["row_wise_c_over_f"] > 0
    assert result["speedup_ratio"]["column_wise_f_over_c"] > 0
    assert result["peak_latency_ms"] >= 0


@pytest.mark.parametrize("shape", [(0, 2), (2,), (2, -1)])
def test_benchmark_rejects_invalid_shape(shape):
    with pytest.raises(ValueError):
        benchmark_memory_layout_impact(shape)


def test_benchmark_rejects_nonpositive_iterations():
    with pytest.raises(ValueError):
        benchmark_memory_layout_impact((2, 2), iterations=0)


def test_vectorized_covariance_and_rolling_mean_match_manual_results():
    data = np.array(
        [[0.01, 0.02], [0.03, -0.01], [-0.02, 0.04], [0.05, 0.03]],
        dtype=np.float64,
    )
    result = vectorized_matrix_operations(data, rolling_window=2)
    np.testing.assert_allclose(result["covariance_matrix"], np.cov(data, rowvar=False))
    expected_rolling = np.stack([data[i : i + 2].mean(axis=0) for i in range(3)])
    np.testing.assert_allclose(result["rolling_mean"], expected_rolling)
    assert result["rolling_windows_zero_copy"] is True
    assert result["output_shapes"]["rolling_mean"] == (3, 2)
    assert result["latency_ms"] >= 0


def test_vectorized_operation_accepts_noncontiguous_float_input():
    base = np.arange(60, dtype=np.float64).reshape(10, 6)
    view = base[:, ::2]
    result = vectorized_matrix_operations(view, rolling_window=3)
    np.testing.assert_allclose(result["covariance_matrix"], np.cov(view, rowvar=False))
    assert result["rolling_windows_zero_copy"] is True
    assert result["flags"]["input_c_contiguous"] is False


@pytest.mark.parametrize(
    "data,kwargs,error",
    [
        (np.ones(5), {}, ValueError),
        (np.ones((1, 2)), {}, ValueError),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), {}, ValueError),
        (np.ones((3, 2)), {"rolling_window": 4}, ValueError),
    ],
)
def test_vectorized_operation_validates_inputs(data, kwargs, error):
    with pytest.raises(error):
        vectorized_matrix_operations(data, **kwargs)


def test_layout_telemetry_reports_strides_and_flags():
    matrix = np.asfortranarray(np.ones((3, 4)))
    report = memory_layout_telemetry(matrix)
    assert report["shape"] == (3, 4)
    assert report["f_contiguous"] is True
    assert report["strides_bytes"] == matrix.strides
    assert report["latency_ms"] >= 0


def test_conversion_rejects_nonfinite_and_non_array_inputs():
    with pytest.raises(ValueError):
        ensure_c_contiguous(np.array([1.0, np.inf]))
    with pytest.raises(TypeError):
        ensure_f_contiguous([1.0, 2.0])
