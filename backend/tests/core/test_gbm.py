import numpy as np
import pytest

from app.core.risk.gbm import (
    calculate_breach_probability,
    calculate_path_terminal_stats,
    gbm_telemetry,
    simulate_gbm_paths,
)


def test_zero_volatility_produces_deterministic_compound_growth():
    s0, mu, dt, steps = 100.0, 0.08, 1 / 252, 252
    paths = simulate_gbm_paths(s0, mu, 0.0, dt=dt, n_steps=steps, n_simulations=5, seed=7)
    assert paths.shape == (steps + 1, 5)
    assert np.allclose(paths[-1], s0 * np.exp(mu * dt * steps))
    assert np.all(paths[:, 1:] == paths[:, [0]])


def test_seed_makes_simulations_reproducible():
    first = simulate_gbm_paths(100.0, 0.05, 0.2, n_steps=40, n_simulations=20, seed=123)
    second = simulate_gbm_paths(100.0, 0.05, 0.2, n_steps=40, n_simulations=20, seed=123)
    assert np.array_equal(first, second)


def test_paths_are_nonnegative_and_include_initial_price():
    paths = simulate_gbm_paths(100.0, 0.03, 0.4, n_steps=100, n_simulations=100, seed=4)
    assert np.all(paths >= 0.0)
    assert np.all(paths[0] == 100.0)
    assert np.isfinite(paths).all()


def test_large_sample_terminal_mean_matches_theoretical_expectation():
    s0, mu, sigma, horizon = 100.0, 0.07, 0.25, 1.0
    paths = simulate_gbm_paths(
        s0, mu, sigma, dt=1 / 252, n_steps=252, n_simulations=50_000, seed=99
    )
    expected = s0 * np.exp(mu * horizon)
    assert np.mean(paths[-1]) == pytest.approx(expected, rel=0.005)


def test_terminal_statistics_report_distribution_quantiles():
    paths = simulate_gbm_paths(100, 0.05, 0.2, n_steps=20, n_simulations=1000, seed=8)
    stats = calculate_path_terminal_stats(paths)
    assert stats["n_simulations"] == 1000
    assert stats["terminal_mean"] > 0
    assert stats["terminal_std"] > 0
    assert stats["terminal_5th_percentile"] < stats["terminal_median"]
    assert stats["terminal_median"] < stats["terminal_95th_percentile"]


def test_barrier_breach_probability_uses_all_path_steps():
    paths = np.array([[100, 100, 100], [95, 85, 110], [90, 70, 120]], dtype=float)
    assert calculate_breach_probability(paths, 80, "down_and_out") == pytest.approx(1 / 3)
    assert calculate_breach_probability(paths, 115, "up_and_out") == pytest.approx(1 / 3)
    assert calculate_breach_probability(paths, 100, "down_and_out") == 1.0


def test_gbm_telemetry_includes_shape_parameters_and_memory_footprint():
    result = gbm_telemetry(100, 0.04, 0.2, n_steps=10, n_simulations=25, seed=1)
    assert result["input_shape"] == [11, 25]
    assert result["memory_bytes"] == 11 * 25 * np.dtype(float).itemsize
    assert result["terminal_mean"] > 0
    assert result["latency_ms"] >= 0


@pytest.mark.parametrize("kwargs", [
    {"s0": 0, "mu": 0.1, "sigma": 0.2},
    {"s0": 100, "mu": 0.1, "sigma": -0.2},
    {"s0": 100, "mu": 0.1, "sigma": 0.2, "n_steps": 0},
    {"s0": 100, "mu": 0.1, "sigma": 0.2, "n_simulations": 0},
])
def test_invalid_simulation_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        simulate_gbm_paths(**kwargs)


def test_barrier_and_path_validation_errors():
    with pytest.raises(ValueError):
        calculate_breach_probability(np.ones((2, 2)), 1.0, "unknown")
    with pytest.raises(ValueError):
        calculate_path_terminal_stats(np.array([[1.0, np.nan]]))
