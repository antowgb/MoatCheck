"""Tests for vectorized Monte Carlo simulation and portfolio risk summaries."""

import numpy as np
import pytest

from app.core.profiling.monte_carlo import (
    MonteCarloEngine,
    calculate_monte_carlo_var_cvar,
    simulate_geometric_brownian_motion,
)


def test_simulation_reproducible_with_fixed_seed():
    first = simulate_geometric_brownian_motion(100, 0.08, 0.2, 12, 5, 1 / 252, seed=17)
    second = simulate_geometric_brownian_motion(100, 0.08, 0.2, 12, 5, 1 / 252, seed=17)
    np.testing.assert_array_equal(first, second)


def test_antithetic_variates_reduce_terminal_mean_estimator_variance():
    standard_estimates = []
    antithetic_estimates = []
    for seed in range(60):
        ordinary = simulate_geometric_brownian_motion(
            100, 0.03, 0.8, 100, 1, 1 / 252, use_antithetic=False, seed=seed
        )
        paired = simulate_geometric_brownian_motion(
            100, 0.03, 0.8, 100, 1, 1 / 252, use_antithetic=True, seed=seed
        )
        standard_estimates.append(ordinary[-1].mean())
        antithetic_estimates.append(paired[-1].mean())
    assert np.var(antithetic_estimates, ddof=1) < np.var(standard_estimates, ddof=1)


def test_single_path_single_step_bounds_and_shape():
    paths = simulate_geometric_brownian_motion(
        50, 0.04, 0.1, n_sims=1, n_steps=1, dt=1 / 252, seed=1
    )
    assert paths.shape == (2, 1)
    assert paths[0, 0] == 50
    assert np.isfinite(paths).all()


def test_zero_volatility_matches_exact_deterministic_gbm_path():
    paths = simulate_geometric_brownian_motion(
        100, 0.1, 0.0, n_sims=3, n_steps=4, dt=0.01, seed=4
    )
    expected = 100 * np.exp(0.1 * 0.01 * np.arange(5))
    np.testing.assert_allclose(paths, expected[:, None] * np.ones((1, 3)))


def test_high_volatility_and_negative_drift_never_create_negative_prices():
    paths = simulate_geometric_brownian_motion(
        s0=100.0,
        mu=-2.0,
        sigma=5.0,
        n_sims=500,
        n_steps=25,
        dt=1.0,
        use_antithetic=True,
        seed=101,
    )
    assert np.isfinite(paths).all()
    assert np.all(paths >= 0.0)
    assert np.all(paths[0] == 100.0)


def test_var_cvar_and_confidence_interval_from_known_terminal_outcomes():
    paths = np.array(
        [[100.0] * 4, [110.0, 100.0, 90.0, 80.0]],
    )
    result = calculate_monte_carlo_var_cvar(paths, confidence_level=0.75)
    assert result["var"] == pytest.approx(0.125)
    assert result["cvar"] == pytest.approx(0.2)
    assert result["var_confidence_interval"][0] <= result["var"]
    assert result["var_confidence_interval"][1] >= result["var"]
    assert result["input_shape"] == (2, 4)


def test_var_cvar_handles_single_simulation_and_single_time_row():
    result = calculate_monte_carlo_var_cvar(np.array([[100.0], [95.0]]))
    assert result["var"] == pytest.approx(0.05)
    assert result["cvar"] == pytest.approx(0.05)
    assert result["terminal_std"] == 0


@pytest.mark.parametrize(
    "args",
    [
        (0, 0.1, 0.2, 2, 2, 0.1),
        (100, 0.1, -0.2, 2, 2, 0.1),
        (100, 0.1, 0.2, 0, 2, 0.1),
        (100, 0.1, 0.2, 2, 0, 0.1),
        (100, 0.1, 0.2, 2, 2, 0),
    ],
)
def test_simulator_rejects_invalid_parameters(args):
    with pytest.raises(ValueError):
        simulate_geometric_brownian_motion(*args)


def test_var_cvar_rejects_invalid_confidence_and_path_values():
    with pytest.raises(ValueError):
        calculate_monte_carlo_var_cvar(np.ones((2, 3)), confidence_level=1.0)
    with pytest.raises(ValueError):
        calculate_monte_carlo_var_cvar(np.array([[100.0, 100.0], [np.nan, 90.0]]))


def test_portfolio_engine_correlated_paths_and_telemetry():
    engine = MonteCarloEngine(
        initial_values=np.array([60.0, 40.0]),
        expected_returns=np.array([0.08, 0.04]),
        covariance_matrix=np.array([[0.04, 0.018], [0.018, 0.09]]),
    )
    result = engine.simulate(n_sims=2000, n_steps=8, dt=1 / 252, seed=9)
    assert result["asset_paths"].shape == (9, 2000, 2)
    assert result["portfolio_paths"].shape == (9, 2000)
    assert result["latency_ms"] >= 0
    assert result["simulation_throughput_paths_per_second"] > 0
    assert result["memory_bytes"] == result["asset_paths"].nbytes + result["portfolio_paths"].nbytes
    assert np.all(result["asset_paths"] > 0)
    assert result["covariance_factorization"] == "cholesky"

    # Correlation of simulated log returns should reflect positive input covariance.
    log_returns = np.diff(np.log(result["asset_paths"]), axis=0).reshape(-1, 2)
    assert np.corrcoef(log_returns.T)[0, 1] > 0.2


def test_portfolio_engine_rejects_invalid_covariance():
    with pytest.raises(ValueError):
        MonteCarloEngine([100, 100], [0.1, 0.1], [[1.0, 2.0], [2.0, 1.0]])


def test_singular_covariance_uses_spectral_fallback_and_reports_it():
    engine = MonteCarloEngine(
        initial_values=np.array([100.0, 100.0]),
        expected_returns=np.array([0.05, 0.05]),
        covariance_matrix=np.array([[0.04, 0.04], [0.04, 0.04]]),
    )
    result = engine.simulate(n_sims=50, n_steps=3, seed=7)
    assert result["covariance_factorization"] == "spectral_fallback"
    assert result["spectral_clipped_eigenvalues"] == 0
    assert np.isfinite(result["asset_paths"]).all()
    assert np.all(result["asset_paths"] > 0)


def test_zero_covariance_uses_spectral_fallback():
    engine = MonteCarloEngine(
        initial_values=np.array([25.0]),
        expected_returns=np.array([0.02]),
        covariance_matrix=np.zeros((1, 1)),
    )
    result = engine.simulate(n_sims=3, n_steps=2, seed=5)
    assert result["covariance_factorization"] == "spectral_fallback"
    np.testing.assert_allclose(
        result["asset_paths"][:, :, 0],
        25.0 * np.exp(0.02 * (1 / 252) * np.arange(3))[:, None] * np.ones((1, 3)),
    )
