import numpy as np
import pandas as pd
import pytest

from app.core.risk.tail_attribution import (
    calculate_component_var,
    calculate_marginal_var_contribution,
    calculate_tail_cvar_attribution,
    tail_attribution_telemetry,
)


def test_identity_covariance_equal_weights_have_equal_var_contributions():
    weights = np.array([1 / 3, 1 / 3, 1 / 3])
    components = calculate_component_var(weights, np.eye(3))
    assert np.allclose(components.to_numpy(), components.iloc[0])
    assert components.sum() == pytest.approx(1.0 / np.sqrt(3) * 1.6448536269514722)


def test_component_var_euler_sum_equals_total_parametric_var():
    weights = np.array([0.5, 0.3, 0.2])
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.01], [0.0, 0.01, 0.01]])
    components = calculate_component_var(weights, cov, confidence_level=0.99)
    total = 2.3263478740408408 * np.sqrt(weights @ cov @ weights)
    assert components.sum() == pytest.approx(total)


def test_marginal_var_matches_gaussian_volatility_gradient():
    weights = np.array([0.6, 0.4])
    cov = np.diag([0.04, 0.01])
    marginal = calculate_marginal_var_contribution(weights, cov)
    expected = 1.6448536269514722 * (cov @ weights) / np.sqrt(weights @ cov @ weights)
    assert np.allclose(marginal, expected)


def test_uncorrelated_and_tail_dependent_assets_have_different_tail_attribution():
    # The second asset experiences large losses on exactly the portfolio's
    # worst dates, so its conditional loss allocation dominates its volatility
    # allocation even though unconditional covariance is modest.
    returns = pd.DataFrame({
        "stable": [0.01] * 18 + [-0.02, -0.03],
        "crash_sensitive": [0.0] * 18 + [-0.25, -0.35],
    })
    weights = np.array([0.8, 0.2])
    tail = calculate_tail_cvar_attribution(returns, weights, confidence_level=0.90)
    assert tail["crash_sensitive"] > tail["stable"]
    portfolio = returns.to_numpy() @ weights
    threshold = np.quantile(portfolio, 0.10)
    assert tail.sum() == pytest.approx(-portfolio[portfolio <= threshold].mean())


def test_empirical_tail_attribution_sums_to_portfolio_tail_expected_loss():
    returns = pd.DataFrame({"a": [-0.10, -0.05, 0.01, 0.02], "b": [-0.20, 0.00, 0.01, 0.03]})
    weights = np.array([0.5, 0.5])
    allocation = calculate_tail_cvar_attribution(returns, weights, confidence_level=0.75)
    portfolio = returns.to_numpy() @ weights
    threshold = np.quantile(portfolio, 0.25)
    expected_loss = -portfolio[portfolio <= threshold].mean()
    assert allocation.sum() == pytest.approx(expected_loss)


def test_telemetry_reports_input_shape_and_exact_euler_reconciliation():
    returns = pd.DataFrame(np.zeros((10, 2)), columns=["A", "B"])
    telemetry = tail_attribution_telemetry(returns, np.array([0.4, 0.6]), np.eye(2))
    assert telemetry["input_shape"] == [10, 2]
    assert telemetry["contribution_sum_matches_total"]
    assert telemetry["percentage_contribution_sum"] == pytest.approx(100.0)
    assert telemetry["latency_ms"] >= 0


@pytest.mark.parametrize("weights,cov", [
    (np.array([1.0]), np.eye(2)),
    (np.array([1.0, 0.0]), np.array([[1.0, 2.0], [0.0, 1.0]])),
])
def test_invalid_covariance_inputs_raise(weights, cov):
    with pytest.raises(ValueError):
        calculate_component_var(weights, cov)


def test_tail_attribution_rejects_mismatched_or_invalid_returns():
    with pytest.raises(ValueError):
        calculate_tail_cvar_attribution(pd.DataFrame([[0.0, 0.1]]), np.array([1.0]))
    with pytest.raises(ValueError):
        calculate_tail_cvar_attribution(pd.DataFrame([[np.nan]]), np.array([1.0]))
