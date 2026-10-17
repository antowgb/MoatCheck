import numpy as np
import pandas as pd
import pytest

from app.core.trading.position_sizing import (
    calculate_growth_rate_projection,
    calculate_multivariate_kelly,
    calculate_univariate_kelly,
    position_sizing_telemetry,
)


def test_univariate_kelly_and_half_kelly_example():
    full = calculate_univariate_kelly(0.60, 1.0, fraction=1.0)
    half = calculate_univariate_kelly(0.60, 1.0, fraction=0.5)
    assert full == pytest.approx(0.20)
    assert half == pytest.approx(0.10)


def test_singular_covariance_uses_pseudoinverse_without_failure():
    covariance = np.array([[0.04, 0.04], [0.04, 0.04]])
    weights = calculate_multivariate_kelly(
        np.array([0.08, 0.08]), covariance, fraction=1.0, max_leverage=10.0
    )
    assert np.isfinite(weights).all()
    assert np.allclose(weights, [1.0, 1.0])


def test_zero_excess_return_yields_zero_kelly_allocation():
    weights = calculate_multivariate_kelly(
        expected_returns=np.array([0.03, 0.03]),
        cov_matrix=np.diag([0.04, 0.09]),
        risk_free_rate=0.03,
    )
    assert np.array_equal(weights, np.zeros(2))


def test_gross_leverage_is_capped_proportionally():
    weights = calculate_multivariate_kelly(
        expected_returns=np.array([0.08, 0.09]),
        cov_matrix=np.diag([0.04, 0.09]),
        fraction=1.0,
        max_leverage=0.75,
    )
    assert np.abs(weights).sum() == pytest.approx(0.75)
    assert np.allclose(weights / weights.sum(), np.array([2.0, 1.0]) / 3.0)


def test_half_kelly_strictly_halves_uncapped_full_kelly_weights():
    expected_returns = np.array([0.08, 0.09])
    covariance = np.diag([0.04, 0.09])
    full = calculate_multivariate_kelly(
        expected_returns, covariance, fraction=1.0, max_leverage=10.0
    )
    half = calculate_multivariate_kelly(
        expected_returns, covariance, fraction=0.5, max_leverage=10.0
    )
    assert np.allclose(half, 0.5 * full)


def test_growth_projection_and_telemetry_report_shapes_and_metrics():
    mu = pd.Series([0.08, 0.09], index=["A", "B"])
    covariance = pd.DataFrame(np.diag([0.04, 0.09]), index=mu.index, columns=mu.index)
    weights = np.array([1.0, 0.5])
    projection = calculate_growth_rate_projection(weights, mu.to_numpy(), covariance)
    expected_variance = float(weights @ covariance.to_numpy() @ weights)
    expected_growth = float(weights @ mu.to_numpy()) - 0.5 * expected_variance
    assert projection["expected_log_growth_rate"] == pytest.approx(expected_growth)
    assert projection["projected_log_wealth_variance"] == pytest.approx(expected_variance)

    telemetry = position_sizing_telemetry(mu.to_numpy(), covariance, fraction=0.5)
    assert telemetry["input_shape"] == [2]
    assert telemetry["covariance_shape"] == [2, 2]
    assert telemetry["gross_leverage"] <= telemetry["max_leverage"]
    assert telemetry["fraction"] == 0.5
    assert telemetry["latency_ms"] >= 0.0


@pytest.mark.parametrize(
    ("p", "ratio", "fraction"),
    [(-0.1, 1.0, 0.5), (0.5, 0.0, 0.5), (0.5, 1.0, 1.1)],
)
def test_univariate_kelly_rejects_invalid_parameters(p, ratio, fraction):
    with pytest.raises(ValueError):
        calculate_univariate_kelly(p, ratio, fraction)


def test_multivariate_kelly_rejects_indefinite_covariance_and_bad_dimensions():
    with pytest.raises(ValueError, match="positive semidefinite"):
        calculate_multivariate_kelly(np.array([0.1, 0.1]), np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="shape"):
        calculate_multivariate_kelly(np.array([0.1, 0.1]), np.eye(3))
