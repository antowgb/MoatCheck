import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from app.core.risk.copulas import (
    copula_telemetry,
    fit_clayton_copula,
    fit_gaussian_copula,
    simulate_copula_joint_returns,
)


def test_independent_variables_have_near_zero_gaussian_copula_coupling():
    rng = np.random.default_rng(71)
    returns = pd.DataFrame(rng.normal(size=(6000, 2)), columns=["A", "B"])
    fitted = fit_gaussian_copula(returns)
    assert abs(fitted["correlation_matrix"][0, 1]) < 0.05
    assert np.allclose(np.diag(fitted["correlation_matrix"]), 1.0)


def test_gaussian_copula_latent_correlation_matches_bivariate_normal_input():
    rng = np.random.default_rng(12)
    correlation = np.array([[1.0, 0.68], [0.68, 1.0]])
    returns = pd.DataFrame(rng.multivariate_normal([0.0, 0.0], correlation, size=8000))
    fitted = fit_gaussian_copula(returns)
    assert fitted["correlation_matrix"][0, 1] == pytest.approx(0.68, abs=0.04)


def test_empirical_uniform_transform_is_bounded_in_unit_interval():
    rng = np.random.default_rng(2)
    returns = pd.DataFrame({"a": rng.standard_t(4, 200), "b": rng.normal(size=200)})
    fitted = fit_gaussian_copula(returns)
    uniforms = fitted["uniform_marginals"]
    assert uniforms.shape == returns.shape
    assert np.all(uniforms >= 0.0)
    assert np.all(uniforms <= 1.0)
    assert np.all(uniforms > 0.0)
    assert np.all(uniforms < 1.0)


def test_clayton_fit_reports_positive_lower_tail_dependence_for_comonotonic_data():
    rng = np.random.default_rng(10)
    common = rng.standard_normal(3000)
    returns = pd.DataFrame({"asset_a": common, "asset_b": common + rng.normal(scale=0.2, size=common.size)})
    fitted = fit_clayton_copula(returns)
    assert fitted["theta"] > 0
    assert fitted["tail_dependence_lower"] == pytest.approx(2 ** (-1 / fitted["theta"]))
    assert fitted["tail_dependence_upper"] == 0.0


def test_simulation_applies_arbitrary_marginals_and_reproduces_seed():
    params = {"correlation_matrix": [[1.0, 0.7], [0.7, 1.0]], "seed": 33}
    marginals = [
        {"distribution": "norm", "params": {"loc": 0.0, "scale": 0.02}},
        {"distribution": "t", "args": [5], "params": {"loc": 0.0, "scale": 0.03}},
    ]
    first = simulate_copula_joint_returns(params, marginals, n_samples=5000)
    second = simulate_copula_joint_returns(params, marginals, n_samples=5000)
    assert first.shape == (5000, 2)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert first[:, 0].std() == pytest.approx(0.02, rel=0.1)


def test_clayton_joint_sampler_has_empirical_lower_tail_dependence():
    params = {"theta": 2.0, "seed": 19}
    marginals = [
        {"distribution": "norm", "params": {"loc": 0.0, "scale": 1.0}},
        {"distribution": "norm", "params": {"loc": 0.0, "scale": 1.0}},
    ]
    samples = simulate_copula_joint_returns(params, marginals, 30000, copula_type="clayton")
    u = norm.cdf(samples)
    lower_tail = np.mean((u[:, 0] < 0.05) & (u[:, 1] < 0.05)) / 0.05
    assert lower_tail > 0.35


def test_copula_telemetry_includes_tail_coefficients_and_shape():
    rng = np.random.default_rng(4)
    returns = pd.DataFrame(rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], size=500))
    telemetry = copula_telemetry(returns, copula_type="gaussian")
    assert telemetry["input_shape"] == [500, 2]
    assert telemetry["tail_dependence_lower"] == 0.0
    assert telemetry["tail_dependence_upper"] == 0.0
    assert telemetry["latency_ms"] >= 0


def test_clayton_requires_exactly_two_assets_and_invalid_marginal_is_rejected():
    with pytest.raises(ValueError):
        fit_clayton_copula(pd.DataFrame(np.random.default_rng(1).normal(size=(50, 3))))
    with pytest.raises(ValueError):
        simulate_copula_joint_returns(
            {"correlation_matrix": np.eye(1)}, [{"distribution": "not_a_distribution"}]
        )
