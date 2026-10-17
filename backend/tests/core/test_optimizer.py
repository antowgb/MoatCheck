"""Tests for long-only Markowitz portfolio optimization."""

import numpy as np
import pandas as pd
import pytest

from app.core.portfolio.optimizer import (
    generate_efficient_frontier,
    optimize_maximum_sharpe,
    optimize_minimum_variance,
)


@pytest.fixture
def two_asset_inputs() -> tuple[pd.Series, pd.DataFrame]:
    assets = ["TECH", "HEALTH"]
    expected_returns = pd.Series([0.12, 0.08], index=assets)
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.0225]], index=assets, columns=assets
    )
    return expected_returns, covariance


def assert_valid_long_only_result(result: dict[str, object]) -> None:
    weights = result["weights"]
    assert isinstance(weights, dict)
    assert sum(weights.values()) == 1.0
    assert all(0.0 <= weight <= 1.0 for weight in weights.values())
    assert result["success"] is True
    assert result["iterations"] > 0
    assert result["latency_ms"] >= 0.0


def test_maximum_sharpe_converges_for_two_assets_and_is_long_only(
    two_asset_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    expected_returns, covariance = two_asset_inputs

    result = optimize_maximum_sharpe(expected_returns, covariance)

    assert_valid_long_only_result(result)
    assert set(result["weights"]) == set(expected_returns.index)
    assert result["expected_return"] > 0.0
    assert result["volatility"] > 0.0
    assert np.isfinite(result["sharpe_ratio"])


def test_minimum_variance_converges_and_returns_exactly_fully_invested_weights(
    two_asset_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    _, covariance = two_asset_inputs

    result = optimize_minimum_variance(covariance)

    assert_valid_long_only_result(result)
    assert result["expected_return"] is None
    assert result["variance"] == pytest.approx(result["volatility"] ** 2)
    assert result["weights"]["TECH"] == pytest.approx((0.0225 - 0.01) / (0.04 + 0.0225 - 2 * 0.01))


def test_efficient_frontier_spans_targets_and_respects_constraints() -> None:
    assets = ["TECH", "HEALTH", "STAPLES"]
    expected_returns = pd.Series([0.14, 0.09, 0.06], index=assets)
    covariance = pd.DataFrame(
        [
            [0.09, 0.018, 0.009],
            [0.018, 0.04, 0.012],
            [0.009, 0.012, 0.0225],
        ],
        index=assets,
        columns=assets,
    )

    frontier = generate_efficient_frontier(expected_returns, covariance, points=5)

    assert len(frontier) == 5
    for point, target in zip(frontier, np.linspace(0.06, 0.14, 5)):
        assert_valid_long_only_result(point)
        assert point["target_return"] == pytest.approx(target)
        assert point["expected_return"] == pytest.approx(target, abs=1e-8)
    volatilities = [point["volatility"] for point in frontier]
    assert all(volatility >= 0.0 for volatility in volatilities)


def test_single_asset_and_equal_return_frontier_degeneracies() -> None:
    single = pd.DataFrame([[0.04]], index=["ONLY"], columns=["ONLY"])
    single_mu = pd.Series([0.10], index=["ONLY"])
    assert optimize_minimum_variance(single)["weights"] == {"ONLY": 1.0}
    assert optimize_maximum_sharpe(single_mu, single)["weights"] == {"ONLY": 1.0}

    assets = ["A", "B"]
    same_mu = pd.Series([0.05, 0.05], index=assets)
    covariance = pd.DataFrame([[0.04, 0.01], [0.01, 0.03]], index=assets, columns=assets)
    frontier = generate_efficient_frontier(same_mu, covariance, points=10)
    assert len(frontier) == 1
    assert_valid_long_only_result(frontier[0])
    assert frontier[0]["target_return"] == pytest.approx(0.05)


@pytest.mark.parametrize("points", [0, 1, 2.5, True])
def test_frontier_requires_at_least_two_integer_points(
    points: int | float | bool,
    two_asset_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    expected_returns, covariance = two_asset_inputs

    with pytest.raises(ValueError, match="at least 2"):
        generate_efficient_frontier(expected_returns, covariance, points=points)  # type: ignore[arg-type]


def test_covariance_and_return_labels_must_match(
    two_asset_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    expected_returns, covariance = two_asset_inputs
    with pytest.raises(ValueError, match="same asset labels"):
        optimize_maximum_sharpe(expected_returns.rename(index={"TECH": "OTHER"}), covariance)


def test_optimizer_rejects_non_positive_semidefinite_covariance() -> None:
    invalid = pd.DataFrame([[0.01, 0.05], [0.05, 0.01]], index=["A", "B"], columns=["A", "B"])

    with pytest.raises(ValueError, match="positive semidefinite"):
        optimize_minimum_variance(invalid)
