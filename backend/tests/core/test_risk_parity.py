"""Tests for long-only risk-parity portfolio construction."""

import numpy as np
import pandas as pd
import pytest

from app.core.portfolio.risk_parity import (
    calculate_marginal_risk_contribution,
    calculate_risk_contributions,
    optimize_risk_parity,
)


def test_uncorrelated_two_asset_solution_is_exact_inverse_volatility() -> None:
    covariance = pd.DataFrame(
        [[0.20**2, 0.0], [0.0, 0.10**2]],
        index=["Tech", "Staples"],
        columns=["Tech", "Staples"],
    )

    result = optimize_risk_parity(covariance)

    assert result["weights"]["Tech"] == pytest.approx(1 / 3, abs=1e-8)
    assert result["weights"]["Staples"] == pytest.approx(2 / 3, abs=1e-8)
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-15)
    assert list(result["risk_contributions"].values()) == pytest.approx([0.5, 0.5])
    assert result["risk_contribution_dispersion"] < 1e-8
    assert result["success"] is True
    assert result["converged"] is True
    assert result["latency_ms"] >= 0.0
    assert result["input_shape"] == {"cov_matrix": [2, 2]}


def test_identity_covariance_returns_equal_weights() -> None:
    covariance = pd.DataFrame(
        np.eye(4), index=["A", "B", "C", "D"], columns=["A", "B", "C", "D"]
    )

    result = optimize_risk_parity(covariance)

    assert list(result["weights"].values()) == pytest.approx([0.25] * 4)
    assert list(result["risk_contributions"].values()) == pytest.approx([0.25] * 4)
    assert sum(result["risk_contributions"].values()) == pytest.approx(1.0)


def test_risk_contribution_and_marginal_risk_formulas() -> None:
    weights = np.array([1 / 3, 2 / 3])
    covariance = np.diag([0.20**2, 0.10**2])
    volatility = np.sqrt(weights @ covariance @ weights)

    marginal = calculate_marginal_risk_contribution(weights, covariance)
    contributions = calculate_risk_contributions(weights, covariance)

    np.testing.assert_allclose(marginal, (covariance @ weights) / volatility)
    np.testing.assert_allclose(contributions, [0.5, 0.5], atol=1e-12)
    assert contributions.sum() == pytest.approx(1.0)


def test_custom_target_risk_budget_is_normalized_and_matched() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]],
        index=["Tech", "Staples"],
        columns=["Tech", "Staples"],
    )

    result = optimize_risk_parity(covariance, np.array([1.0, 3.0]))

    assert list(result["target_risk_budget"].values()) == pytest.approx([0.25, 0.75])
    assert list(result["risk_contributions"].values()) == pytest.approx(
        [0.25, 0.75], abs=1e-7
    )
    assert result["risk_contribution_dispersion"] < 1e-7


@pytest.mark.parametrize(
    ("weights", "covariance", "message"),
    [
        (np.array([[0.5, 0.5]]), np.eye(2), "one-dimensional"),
        (np.array([0.5, 0.5]), np.eye(3), "length must match"),
        (np.array([0.5, np.nan]), np.eye(2), "finite"),
        (np.array([0.5, 0.5]), np.zeros((2, 3)), "square matrix"),
    ],
)
def test_risk_function_inputs_are_validated(
    weights: np.ndarray, covariance: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        calculate_risk_contributions(weights, covariance)


def test_zero_risk_and_non_psd_covariances_are_rejected() -> None:
    weights = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="volatility must be positive"):
        calculate_marginal_risk_contribution(weights, np.zeros((2, 2)))

    non_psd = pd.DataFrame(
        [[1.0, 2.0], [2.0, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        optimize_risk_parity(non_psd)


@pytest.mark.parametrize(
    "budget",
    [np.array([0.0, 1.0]), np.array([np.nan, 1.0]), np.array([1.0])],
)
def test_invalid_risk_budgets_are_rejected(budget: np.ndarray) -> None:
    covariance = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError, match="target_risk_budget"):
        optimize_risk_parity(covariance, budget)


def test_covariance_labels_must_match_and_be_unique() -> None:
    mismatched = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]], index=["A", "B"], columns=["A", "C"]
    )
    with pytest.raises(ValueError, match="labels must match"):
        optimize_risk_parity(mismatched)

