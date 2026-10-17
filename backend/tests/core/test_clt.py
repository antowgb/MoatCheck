"""Tests for bootstrap aggregation and return-distribution diagnostics."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis, skew

from app.core.research.clt import (
    analyze_returns_with_telemetry,
    calculate_higher_moments,
    normal_distribution_reference,
    simulate_clt_aggregation,
    simulate_clt_aggregation_with_telemetry,
    test_normality_shapiro as check_normality_shapiro,
)


def test_standard_normal_input_has_expected_moments_and_shapiro_result() -> None:
    values = pd.Series(np.random.default_rng(23).normal(size=3000))

    moments = calculate_higher_moments(values)
    normality = check_normality_shapiro(values)

    assert moments["mean"] == pytest.approx(0.0, abs=0.06)
    assert moments["variance"] == pytest.approx(1.0, abs=0.08)
    assert moments["skewness"] == pytest.approx(0.0, abs=0.1)
    assert moments["kurtosis"] == pytest.approx(0.0, abs=0.2)
    assert normality["statistic"] > 0.99
    assert normality["p_value"] > 0.05
    assert normality["normality_not_rejected"] is True


def test_skewed_daily_distribution_becomes_less_skewed_after_monthly_aggregation() -> None:
    daily = pd.Series(np.random.default_rng(17).lognormal(mean=0.0, sigma=0.7, size=20_000))

    monthly = simulate_clt_aggregation(
        daily, sample_size=30, n_simulations=20_000, random_state=99
    )

    assert abs(float(skew(monthly, bias=False))) < abs(float(skew(daily, bias=False)))


def test_excess_kurtosis_moves_closer_to_zero_after_aggregation() -> None:
    daily = pd.Series(np.random.default_rng(31).lognormal(mean=0.0, sigma=0.7, size=20_000))
    daily_kurtosis = float(kurtosis(daily, fisher=True, bias=False))

    monthly = simulate_clt_aggregation(
        daily, sample_size=30, n_simulations=20_000, random_state=7
    )
    monthly_kurtosis = float(kurtosis(monthly, fisher=True, bias=False))

    assert abs(monthly_kurtosis) < abs(daily_kurtosis)


def test_bootstrap_sum_matches_manual_draw_and_is_reproducible() -> None:
    daily = pd.Series([1.0, 2.0, 3.0])

    first = simulate_clt_aggregation(daily, sample_size=4, n_simulations=12, random_state=4)
    second = simulate_clt_aggregation(daily, sample_size=4, n_simulations=12, random_state=4)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (12,)
    assert np.all((first >= 4.0) & (first <= 12.0))


def test_simulation_drops_nan_days_and_reports_telemetry() -> None:
    daily = pd.Series([0.01, np.nan, -0.02, 0.03])

    aggregates, telemetry = simulate_clt_aggregation_with_telemetry(
        daily, sample_size=5, n_simulations=17, random_state=11
    )

    assert aggregates.shape == (17,)
    assert telemetry["latency_ms"] >= 0.0
    assert telemetry["input_observations"] == 3
    assert telemetry["sample_size"] == 5
    assert telemetry["n_simulations"] == 17
    assert telemetry["output_shape"] == [17]


def test_higher_moments_return_sample_variance_and_excess_kurtosis() -> None:
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    result = calculate_higher_moments(values)

    assert result["mean"] == pytest.approx(0.0)
    assert result["variance"] == pytest.approx(2.5)
    assert result["skewness"] == pytest.approx(0.0, abs=1e-14)
    assert result["kurtosis"] == pytest.approx(-1.2)
    assert result["observations"] == 5


def test_analysis_telemetry_exposes_normality_flag_and_shape_moments() -> None:
    values = pd.Series(np.random.default_rng(8).normal(size=1000))

    moments, telemetry = analyze_returns_with_telemetry(values)

    assert moments["observations"] == 1000
    assert telemetry["latency_ms"] >= 0.0
    assert telemetry["observations"] == 1000
    assert isinstance(telemetry["normality_not_rejected"], bool)
    assert telemetry["skewness"] == pytest.approx(moments["skewness"])
    assert telemetry["kurtosis"] == pytest.approx(moments["kurtosis"])


def test_normal_reference_curve_uses_requested_mean_and_variance() -> None:
    x, density = normal_distribution_reference(mean=2.0, variance=4.0, points=101)

    assert x.shape == density.shape == (101,)
    assert x[50] == pytest.approx(2.0)
    assert density[50] == pytest.approx(1.0 / np.sqrt(2.0 * np.pi * 4.0))
    assert np.all(density >= 0.0)


def test_small_samples_and_empty_inputs_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="at least three"):
        check_normality_shapiro(pd.Series([0.01, -0.01]))
    with pytest.raises(ValueError, match="at least two"):
        calculate_higher_moments(np.array([0.01]))
    with pytest.raises(ValueError, match="at least one non-missing"):
        simulate_clt_aggregation(pd.Series([np.nan]))
    with pytest.raises(ValueError, match="at least three"):
        check_normality_shapiro(pd.Series([np.nan, 0.01, -0.01]))


@pytest.mark.parametrize(
    ("sample_size", "n_simulations"),
    [(0, 10), (2, 0), (True, 10)],
)
def test_simulation_requires_positive_integer_dimensions(
    sample_size: int, n_simulations: int
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        simulate_clt_aggregation(pd.Series([0.01, -0.01]), sample_size, n_simulations)


@pytest.mark.parametrize("returns", [[0.01, np.inf], [[0.01, 0.02], [0.03, 0.04]]])
def test_invalid_distribution_inputs_are_rejected(returns: list[float]) -> None:
    with pytest.raises(ValueError):
        calculate_higher_moments(np.asarray(returns))


def test_constant_returns_have_undefined_moments_and_shapiro_raises() -> None:
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])

    moments = calculate_higher_moments(returns)

    assert moments["variance"] == 0.0
    assert np.isnan(moments["skewness"])
    assert np.isnan(moments["kurtosis"])
    with pytest.raises(ValueError, match="constant"):
        check_normality_shapiro(returns)
