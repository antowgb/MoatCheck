"""Tests for the pure Black-Litterman posterior and allocation engine."""

import numpy as np
import pandas as pd
import pytest

from app.core.portfolio.black_litterman import (
    black_litterman_posterior,
    implied_market_returns,
    optimize_black_litterman,
)


def _two_asset_inputs() -> tuple[pd.Series, pd.DataFrame]:
    weights = pd.Series([0.6, 0.4], index=["MSFT", "AAPL"], name="market_weight")
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["MSFT", "AAPL"],
        columns=["MSFT", "AAPL"],
    )
    return weights, covariance


def test_implied_returns_match_reverse_optimization_formula() -> None:
    weights, covariance = _two_asset_inputs()

    result = implied_market_returns(weights, covariance, risk_aversion=2.5)

    expected = 2.5 * covariance.to_numpy() @ weights.to_numpy()
    np.testing.assert_allclose(result.to_numpy(), expected)
    assert list(result.index) == ["MSFT", "AAPL"]


def test_empty_views_preserve_equilibrium_posterior() -> None:
    weights, covariance = _two_asset_inputs()
    implied = implied_market_returns(weights, covariance)
    p = np.empty((0, 2))
    q = np.empty(0)

    posterior, posterior_covariance = black_litterman_posterior(
        implied, covariance, p, q
    )

    pd.testing.assert_series_equal(
        posterior,
        implied.rename("posterior_returns"),
    )
    pd.testing.assert_frame_equal(
        posterior_covariance,
        covariance * 0.05,
        check_names=False,
    )


def test_confident_moat_view_moves_msft_return_above_market_prior() -> None:
    weights, covariance = _two_asset_inputs()
    implied = implied_market_returns(weights, covariance)

    posterior, posterior_uncertainty = black_litterman_posterior(
        implied,
        covariance,
        views_P=np.array([[1.0, -1.0]]),
        views_Q=np.array([0.10]),
        omega=np.array([[1e-5]]),
    )

    assert posterior["MSFT"] - posterior["AAPL"] > implied["MSFT"] - implied["AAPL"]
    assert np.linalg.eigvalsh(posterior_uncertainty.to_numpy()).min() >= -1e-12


def test_high_view_uncertainty_pulls_posterior_back_to_prior() -> None:
    weights, covariance = _two_asset_inputs()
    implied = implied_market_returns(weights, covariance)

    posterior, _ = black_litterman_posterior(
        implied,
        covariance,
        views_P=np.array([[1.0, -1.0]]),
        views_Q=np.array([0.20]),
        omega=np.array([[1e8]]),
    )

    np.testing.assert_allclose(posterior.to_numpy(), implied.to_numpy(), atol=1e-9)


def test_identity_covariance_optimizer_respects_long_only_bounds() -> None:
    assets = ["MSFT", "AAPL", "NVDA"]
    covariance = pd.DataFrame(np.eye(3), index=assets, columns=assets)
    market_weights = pd.Series([0.4, 0.35, 0.25], index=assets)

    result = optimize_black_litterman(
        market_weights,
        covariance,
        views_P=np.array([[1.0, -1.0, 0.0]]),
        views_Q=np.array([2.0]),
        omega=np.array([[0.001]]),
    )

    assert result["success"] is True
    assert result["converged"] is True
    assert result["latency_ms"] >= 0.0
    assert result["input_shape"] == {
        "market_weights": [3],
        "cov_matrix": [3, 3],
        "views_P": [1, 3],
        "views_Q": [1],
    }
    assert result["weights"].sum() == pytest.approx(1.0, abs=1e-12)
    assert (result["weights"] >= 0.0).all()
    assert (result["weights"] <= 1.0).all()
    assert result["view_weight_delta"]["MSFT"] > 0.0


def test_no_view_optimization_recovers_market_weights() -> None:
    weights, covariance = _two_asset_inputs()

    result = optimize_black_litterman(
        weights,
        covariance,
        views_P=np.empty((0, 2)),
        views_Q=np.empty(0),
    )

    pd.testing.assert_series_equal(
        result["weights"], weights.rename("weights"), check_exact=False, atol=1e-8
    )
    assert result["weight_l1_delta"] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize(
    ("views_p", "views_q", "omega", "message"),
    [
        (np.array([1.0, -1.0]), np.array([0.1]), None, "shape"),
        (np.array([[1.0, -1.0]]), np.array([]), None, "one value"),
        (np.array([[1.0, -1.0]]), np.array([0.1]), np.eye(2), "omega must have shape"),
    ],
)
def test_view_dimensions_are_validated(
    views_p: np.ndarray, views_q: np.ndarray, omega: np.ndarray | None, message: str
) -> None:
    weights, covariance = _two_asset_inputs()
    implied = implied_market_returns(weights, covariance)
    with pytest.raises(ValueError, match=message):
        black_litterman_posterior(implied, covariance, views_p, views_q, omega)


def test_invalid_market_inputs_are_rejected() -> None:
    weights, covariance = _two_asset_inputs()
    with pytest.raises(ValueError, match="sum to 1.0"):
        implied_market_returns(weights * 2.0, covariance)
    with pytest.raises(ValueError, match="same asset labels"):
        implied_market_returns(weights.rename(index={"MSFT": "GOOG"}), covariance)
    with pytest.raises(ValueError, match="positive"):
        implied_market_returns(weights, covariance, risk_aversion=0.0)
    with pytest.raises(ValueError, match="positive definite"):
        implied_market_returns(weights, pd.DataFrame(np.zeros((2, 2)), index=weights.index, columns=weights.index))

