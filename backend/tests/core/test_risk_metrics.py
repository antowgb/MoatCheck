"""Tests for scalar VaR, expected shortfall, and risk telemetry."""

import numpy as np
import pandas as pd
import pytest

from app.core.risk.metrics import (
    as_loss_amount,
    as_return_quantile,
    calculate_cornish_fisher_var,
    calculate_cvar_expected_shortfall,
    calculate_historical_var,
    calculate_parametric_var,
    portfolio_var_cvar,
    risk_metrics_telemetry,
)


def test_var_sign_conversion_helpers_are_explicit_and_inverse():
    signed_return_quantile = -0.021
    loss_fraction = as_loss_amount(signed_return_quantile)
    assert loss_fraction == pytest.approx(0.021)
    assert as_return_quantile(loss_fraction) == pytest.approx(signed_return_quantile)

    portfolio = pd.DataFrame({"asset": [-0.02, 0.0, 0.01, 0.02]})
    positive_loss_var = portfolio_var_cvar(portfolio, [1.0], confidence=0.75)["var"]
    assert as_return_quantile(positive_loss_var) < 0.0


@pytest.mark.parametrize("helper", [as_loss_amount, as_return_quantile])
def test_var_sign_conversion_rejects_non_finite_values(helper):
    with pytest.raises(ValueError, match="finite"):
        helper(float("inf"))


def test_gaussian_returns_make_three_var_estimators_close():
    rng = np.random.default_rng(991)
    returns = pd.Series(rng.normal(loc=0.0002, scale=0.02, size=500_000))

    parametric = calculate_parametric_var(returns, 0.95)
    historical = calculate_historical_var(returns, 0.95)
    cornish_fisher = calculate_cornish_fisher_var(returns, 0.95)

    assert parametric == pytest.approx(historical, abs=2.5e-4)
    assert parametric == pytest.approx(cornish_fisher, abs=2.5e-4)
    assert historical == pytest.approx(cornish_fisher, abs=2.5e-4)


def test_heavy_tailed_sample_makes_99_percent_cornish_fisher_var_more_severe():
    rng = np.random.default_rng(212)
    # Student-t with six degrees of freedom has positive excess kurtosis.
    returns = pd.Series(rng.standard_t(df=6, size=250_000) * 0.01)

    gaussian_var = calculate_parametric_var(returns, confidence_level=0.99)
    adjusted_var = calculate_cornish_fisher_var(returns, confidence_level=0.99)

    # These APIs return signed lower-tail returns; a worse estimate is lower.
    assert adjusted_var < gaussian_var
    assert abs(adjusted_var) > abs(gaussian_var)


def test_historical_cvar_is_strictly_more_negative_than_historical_var():
    rng = np.random.default_rng(83)
    returns = pd.Series(rng.normal(loc=0.0005, scale=0.02, size=100_000))

    var = calculate_historical_var(returns, confidence_level=0.95)
    cvar = calculate_cvar_expected_shortfall(
        returns, confidence_level=0.95, method="historical"
    )

    assert cvar < var < 0.0


def test_parametric_expected_shortfall_matches_gaussian_formula():
    returns = np.linspace(-0.04, 0.05, 1001)
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    from scipy.stats import norm

    z = norm.ppf(0.05)
    expected = mean - std * norm.pdf(z) / 0.05

    assert calculate_cvar_expected_shortfall(returns, 0.95, "parametric") == pytest.approx(expected)


def test_telemetry_reports_shape_moments_and_latency():
    returns = pd.Series([-0.04, -0.01, 0.0, 0.01, 0.02, 0.03], name="daily_return")

    telemetry = risk_metrics_telemetry(returns)

    assert telemetry["input_shape"] == [len(returns)]
    assert telemetry["n_observations"] == len(returns)
    assert telemetry["latency_ms"] >= 0.0
    assert np.isfinite(telemetry["skewness"])
    assert np.isfinite(telemetry["excess_kurtosis"])


def test_nan_values_are_ignored_but_infinity_and_bad_shapes_are_rejected():
    assert calculate_historical_var(pd.Series([-0.1, np.nan, 0.0, 0.1]), 0.5) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="finite"):
        calculate_parametric_var(np.array([-0.01, 0.01, np.inf]))
    with pytest.raises(ValueError, match="one-dimensional"):
        calculate_historical_var(np.ones((3, 2)))


@pytest.mark.parametrize("confidence", [0.0, 1.0, float("nan"), -0.1])
def test_var_estimators_reject_invalid_confidence(confidence):
    returns = np.array([-0.01, 0.0, 0.01, 0.02])
    with pytest.raises(ValueError, match="confidence_level"):
        calculate_parametric_var(returns, confidence)


def test_cornish_fisher_requires_enough_observations():
    with pytest.raises(ValueError, match="at least 4"):
        calculate_cornish_fisher_var(np.array([-0.01, 0.01, 0.02]))

