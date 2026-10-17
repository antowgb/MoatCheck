"""Tests for mean-reversion diagnostics."""

import numpy as np
import pandas as pd
import pytest

from app.core.research.mean_reversion import (
    calculate_half_life,
    calculate_rolling_z_score,
    estimate_ornstein_uhlenbeck_params,
    mean_reversion_telemetry,
)


def _ar1_series(phi: float, *, n: int = 2500, seed: int = 17) -> pd.Series:
    """Build a seeded AR(1) process with intercept anchored at level 25."""
    rng = np.random.default_rng(seed)
    values = np.empty(n, dtype=float)
    values[0] = 25.0
    intercept = (1.0 - phi) * 25.0
    for index in range(1, n):
        values[index] = intercept + phi * values[index - 1] + rng.normal(scale=0.5)
    return pd.Series(values, name="pe_ratio")


def test_estimate_ou_parameters_for_mean_reverting_process():
    series = _ar1_series(0.8)

    params = estimate_ornstein_uhlenbeck_params(series)

    assert params["is_stationary"] is True
    assert params["is_ou_mean_reverting"] is True
    assert params["ar1_coefficient"] == pytest.approx(0.8, abs=0.04)
    assert params["kappa"] == pytest.approx(-np.log(params["ar1_coefficient"]))
    assert params["theta"] == pytest.approx(25.0, abs=1.0)
    assert params["sigma"] > 0
    assert params["half_life_days"] == pytest.approx(np.log(2) / params["kappa"])
    assert params["n_observations"] == len(series)
    assert params["latency_ms"] >= 0


def test_calculate_half_life_matches_fitted_mean_reversion_speed():
    series = _ar1_series(0.7)
    params = estimate_ornstein_uhlenbeck_params(series)

    assert calculate_half_life(series) == pytest.approx(
        np.log(2) / params["kappa"]
    )


def test_trending_nonstationary_process_has_no_finite_half_life():
    # phi > 1 gives a seeded trending/explosive process, whose OLS estimate
    # should not be mislabeled as a stationary mean-reverting OU process.
    series = _ar1_series(1.01, n=1000, seed=8)

    params = estimate_ornstein_uhlenbeck_params(series)

    assert params["ar1_coefficient"] >= 1.0
    assert params["kappa"] <= 0.0
    assert params["is_stationary"] is False
    assert params["is_ou_mean_reverting"] is False
    assert np.isinf(calculate_half_life(series))


def test_rolling_z_score_uses_trailing_sample_window_and_preserves_index():
    index = pd.Index(["a", "b", "c", "d", "e"], name="date")
    series = pd.Series([1, 2, 3, 4, 5], index=index, name="pe_ratio")

    result = calculate_rolling_z_score(series, window=3)

    assert result.index.equals(index)
    assert result.name == series.name
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].to_numpy() == pytest.approx([1.0, 1.0, 1.0])


@pytest.mark.parametrize("window", [0, 1, 6, 2.5, True])
def test_rolling_z_score_rejects_invalid_window(window):
    with pytest.raises(ValueError, match="window"):
        calculate_rolling_z_score(pd.Series([1, 2, 3, 4, 5]), window=window)


def test_rolling_z_score_rejects_constant_series_values_with_nan_result():
    result = calculate_rolling_z_score(pd.Series([3.0] * 4), window=2)

    assert result.isna().all()


def test_estimation_rejects_insufficient_or_constant_observations():
    with pytest.raises(ValueError, match="at least 3"):
        estimate_ornstein_uhlenbeck_params(pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="vary"):
        estimate_ornstein_uhlenbeck_params(pd.Series([2.0, 2.0, 2.0]))


def test_ou_estimation_rejects_missing_observations_and_irregular_date_gaps():
    values = _ar1_series(0.8, n=40).to_numpy()
    dates = pd.date_range("2025-01-06", periods=40, freq="B")
    regular_daily = pd.Series(values, index=dates)
    with pytest.raises(ValueError, match="regular daily or business-daily"):
        estimate_ornstein_uhlenbeck_params(regular_daily.drop(index=dates[20]))

    values_with_gap = pd.Series(values, index=pd.RangeIndex(40))
    values_with_gap.iloc[20] = np.nan
    with pytest.raises(ValueError, match="missing observations"):
        estimate_ornstein_uhlenbeck_params(values_with_gap)


def test_telemetry_includes_latency_half_life_and_stationarity_flags():
    telemetry = mean_reversion_telemetry(_ar1_series(0.85, seed=4))

    assert telemetry["latency_ms"] >= 0
    assert np.isfinite(telemetry["half_life_days"])
    assert telemetry["is_stationary"] is True
    assert telemetry["is_ou_mean_reverting"] is True
