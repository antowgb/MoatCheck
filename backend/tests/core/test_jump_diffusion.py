import numpy as np
import pytest

from app.core.risk.gbm import simulate_gbm_paths
from app.core.risk.jump_diffusion import (
    calculate_jump_impact_diagnostics,
    jump_diffusion_telemetry,
    simulate_merton_jump_diffusion,
    simulate_poisson_jumps,
)


def test_zero_jump_intensity_matches_pure_gbm_exactly_with_same_seed():
    params = dict(s0=100.0, mu=0.06, sigma=0.2, dt=1 / 252, n_steps=80, n_simulations=40, seed=44)
    merton = simulate_merton_jump_diffusion(**params, lambda_param=0.0, mu_j=-0.2, sigma_j=0.3)
    gbm = simulate_gbm_paths(**params)
    assert np.array_equal(merton, gbm)


def test_merton_paths_are_nonnegative_finite_and_have_expected_shape():
    paths = simulate_merton_jump_diffusion(
        100, 0.05, 0.2, 0.4, -0.2, 0.25, n_steps=100, n_simulations=200, seed=10
    )
    assert paths.shape == (101, 200)
    assert np.isfinite(paths).all()
    assert np.all(paths >= 0.0)
    assert np.all(paths[0] == 100.0)


def test_jump_intensity_scales_with_horizon():
    short = simulate_poisson_jumps(2.0, n_steps=126, n_simulations=10000, dt=1 / 252, seed=5)
    long = simulate_poisson_jumps(2.0, n_steps=252, n_simulations=10000, dt=1 / 252, seed=5)
    assert long.sum() == pytest.approx(2 * short.sum(), rel=0.08)


def test_poisson_jump_count_converges_to_lambda_times_horizon():
    intensity, dt, steps, simulations = 1.7, 1 / 252, 252, 20000
    arrivals = simulate_poisson_jumps(intensity, steps, simulations, dt=dt, seed=22)
    expected_mean = intensity * dt * steps
    assert arrivals.shape == (steps, simulations)
    assert arrivals.sum() / simulations == pytest.approx(expected_mean, abs=0.04)


def test_jump_diagnostics_isolate_tail_and_daily_return_differences():
    params = dict(s0=100.0, mu=0.04, sigma=0.15, dt=1 / 252, n_steps=252, n_simulations=2000, seed=71)
    gbm = simulate_gbm_paths(**params)
    jumps = simulate_merton_jump_diffusion(
        **params, lambda_param=3.0, mu_j=-0.25, sigma_j=0.15
    )
    result = calculate_jump_impact_diagnostics(jumps, gbm)
    assert result["input_shape"] == [253, 2000]
    assert np.isfinite(result["excess_kurtosis"])
    assert result["max_negative_daily_return_gap"] < 0
    assert result["tail_loss_divergence"] > 0


def test_zero_intensity_telemetry_has_no_realized_jumps():
    result = jump_diffusion_telemetry(
        100, 0.03, 0.2, 0.0, -0.2, 0.3, n_steps=30, n_simulations=40, seed=8
    )
    assert result["input_shape"] == [31, 40]
    assert result["total_jumps_realized"] == 0
    assert result["mean_jump_magnitude"] == 0.0
    assert result["latency_ms"] >= 0


@pytest.mark.parametrize("kwargs", [
    {"s0": 0, "mu": 0.0, "sigma": 0.1, "lambda_param": 0, "mu_j": 0, "sigma_j": 0},
    {"s0": 100, "mu": 0.0, "sigma": -0.1, "lambda_param": 0, "mu_j": 0, "sigma_j": 0},
    {"s0": 100, "mu": 0.0, "sigma": 0.1, "lambda_param": -1, "mu_j": 0, "sigma_j": 0},
])
def test_invalid_jump_diffusion_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        simulate_merton_jump_diffusion(**kwargs)


def test_diagnostics_reject_different_path_shapes():
    with pytest.raises(ValueError):
        calculate_jump_impact_diagnostics(np.ones((5, 2)), np.ones((4, 2)))
