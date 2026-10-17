"""Tests for finite-state Markov chains and martingale diagnostics."""

import numpy as np
import pandas as pd
import pytest

from app.core.profiling.stochastic_processes import (
    calculate_stationary_distribution,
    simulate_markov_chain_path,
    stochastic_process_telemetry,
    test_martingale_property as run_martingale_diagnostic,
)


THREE_REGIME = np.array(
    [
        [0.80, 0.05, 0.15],  # Bull -> Bull, Bear, Sideways
        [0.10, 0.75, 0.15],
        [0.20, 0.10, 0.70],
    ]
)


def test_transition_matrix_validation_and_stationary_invariance():
    stationary = calculate_stationary_distribution(THREE_REGIME)
    assert stationary.shape == (3,)
    assert np.all(stationary >= 0)
    assert stationary.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(stationary @ THREE_REGIME, stationary, atol=1e-12)
    telemetry = stochastic_process_telemetry(THREE_REGIME, stationary, latency_ms=1.25)
    assert telemetry["input_shape"] == (3, 3)
    assert telemetry["stationary_convergence_error"] < 1e-12
    assert telemetry["latency_ms"] == 1.25


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[0.8, 0.3], [0.2, 0.8]]),  # Row sums exceed one.
        np.array([[1.1, -0.1], [0.1, 0.9]]),  # Invalid probability entries.
        np.array([[1.0, 0.0], [0.2, np.nan]]),
        np.array([0.5, 0.5]),
    ],
)
def test_invalid_transition_matrices_are_rejected(matrix):
    with pytest.raises(ValueError):
        calculate_stationary_distribution(matrix)


def test_markov_path_is_reproducible_and_uses_valid_state_ids():
    first = simulate_markov_chain_path(THREE_REGIME, initial_state=0, n_steps=100, seed=44)
    second = simulate_markov_chain_path(THREE_REGIME, initial_state=0, n_steps=100, seed=44)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (101,)
    assert set(np.unique(first)) <= {0, 1, 2}
    assert first[0] == 0


def test_zero_step_chain_returns_initial_state_only():
    path = simulate_markov_chain_path(np.eye(3), initial_state=2, n_steps=0, seed=2)
    np.testing.assert_array_equal(path, [2])


def test_reducible_chain_returns_valid_stationary_mixture():
    transition = np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    stationary = calculate_stationary_distribution(transition)
    np.testing.assert_allclose(stationary @ transition, stationary, atol=1e-12)
    assert np.all(stationary >= 0)
    assert stationary.sum() == pytest.approx(1.0)


def test_random_walk_log_returns_are_compatible_with_martingale_difference_null():
    rng = np.random.default_rng(1234)
    log_returns = rng.normal(loc=0.0, scale=0.01, size=20_000)
    prices = pd.Series(100.0 * np.exp(np.r_[0.0, np.cumsum(log_returns)]))
    result = run_martingale_diagnostic(prices, max_lags=5)
    assert result["mean_zero_test"]["p_value"] > 0.05
    assert result["martingale_difference_test"]["p_value"] > 0.05
    assert result["variance_ratio_tests"]
    assert result["n_log_returns"] == 20_000
    assert result["latency_ms"] >= 0


def test_martingale_diagnostic_rejects_invalid_price_inputs():
    with pytest.raises(ValueError):
        run_martingale_diagnostic(pd.Series([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        run_martingale_diagnostic(pd.Series([1.0, 2.0]), max_lags=0)
