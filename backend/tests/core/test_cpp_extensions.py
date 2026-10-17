"""Tests for native extension dispatch and transparent NumPy fallbacks."""

import numpy as np
import pytest

from app.core.profiling.cpp_extensions import (
    CppExtensionWrapper,
    _numpy_kalman_loop,
    _numpy_moving_average,
    c_fast_kalman_loop,
    c_fast_moving_average,
)


class FakeNativeExtension:
    """A Python stand-in exposing the compiled binding contract."""

    def __init__(self):
        self.received_contiguous = []

    def c_fast_moving_average(self, arr, window):
        self.received_contiguous.append(bool(arr.flags.c_contiguous))
        return _numpy_moving_average(arr, window)

    def c_fast_kalman_loop(self, measurements, R, Q):
        self.received_contiguous.append(bool(measurements.flags.c_contiguous))
        return _numpy_kalman_loop(measurements, R, Q)


def test_numpy_fallback_moving_average_matches_native_binding_output():
    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    # An injected module without the target symbol forces the NumPy fallback
    # even on machines where a real compiled extension happens to be installed.
    fallback = CppExtensionWrapper(native_module=object())
    native = CppExtensionWrapper(native_module=FakeNativeExtension())
    expected = fallback.moving_average(values, 3)
    actual = native.moving_average(values, 3)
    np.testing.assert_array_equal(actual, expected)
    assert native.last_telemetry["extension_status"] == "cpp_native"


def test_moving_average_preserves_1d_and_2d_shapes():
    one_dimensional = c_fast_moving_average(np.arange(6, dtype=float), 2)
    two_dimensional = c_fast_moving_average(np.arange(12, dtype=float).reshape(6, 2), 2)
    assert one_dimensional.shape == (6,)
    assert two_dimensional.shape == (6, 2)
    np.testing.assert_allclose(two_dimensional[-1], [9.0, 10.0])
    assert np.isnan(one_dimensional[0])


def test_kalman_fallback_matches_native_binding_numerically():
    measurements = np.array([1.0, 1.4, 0.8, 2.0, 2.5])
    wrapper = CppExtensionWrapper(native_module=FakeNativeExtension())
    expected = _numpy_kalman_loop(measurements, 0.3, 0.05)
    actual = wrapper.kalman_filter(measurements, 0.3, 0.05)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    np.testing.assert_allclose(c_fast_kalman_loop(measurements, 0.3, 0.05), expected)


def test_non_contiguous_views_are_copied_before_native_calls():
    native_module = FakeNativeExtension()
    wrapper = CppExtensionWrapper(native_module=native_module)
    base = np.arange(40, dtype=np.float64).reshape(10, 4)
    view2d = base[::2, ::2]
    view1d = np.arange(20, dtype=np.float64)[::2]
    assert not view2d.flags.c_contiguous
    assert not view1d.flags.c_contiguous

    result = wrapper.moving_average(view2d, 2)
    kalman = wrapper.kalman_filter(view1d, 1.0, 0.0)
    assert result.shape == view2d.shape
    assert kalman.shape == view1d.shape
    assert native_module.received_contiguous == [True, True]


def test_speedup_telemetry_and_status_fields_are_present():
    wrapper = CppExtensionWrapper(native_module=FakeNativeExtension())
    result = wrapper.benchmark_speedup("moving_average", np.arange(100.0), 5)
    assert result["extension_status"] == "cpp_native"
    assert result["numerical_parity"] is True
    assert result["speedup_ratio"] > 0
    assert result["input_shape"] == (100,)


@pytest.mark.parametrize("array,window", [(np.ones(4), 0), (np.ones(4), 5), (np.ones((2, 2, 2)), 1)])
def test_moving_average_rejects_invalid_window_or_dimensions(array, window):
    with pytest.raises(ValueError):
        c_fast_moving_average(array, window)


@pytest.mark.parametrize("r,q", [(0, 1), (1, -1), (np.nan, 1), (1, np.inf)])
def test_kalman_rejects_invalid_noise_variances(r, q):
    with pytest.raises(ValueError):
        c_fast_kalman_loop(np.array([1.0, 2.0]), r, q)


def test_kalman_rejects_empty_or_multidimensional_measurements():
    with pytest.raises(ValueError):
        c_fast_kalman_loop(np.array([]), 1, 1)
    with pytest.raises(ValueError):
        c_fast_kalman_loop(np.ones((3, 1)), 1, 1)
