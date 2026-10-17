"""Tests for Hurst exponent and regime diagnostics."""

import numpy as np
import pandas as pd
import pytest

from app.core.research.hurst import (
    calculate_hurst_exponent,
    calculate_rolling_hurst,
    classify_market_regime,
    hurst_telemetry,
)


def test_random_walk_hurst_is_approximately_one_half():
    rng = np.random.default_rng(402)
    random_walk = pd.Series(np.cumsum(rng.normal(size=8000)))

    hurst = calculate_hurst_exponent(random_walk)

    assert hurst == pytest.approx(0.5, abs=0.12)
    assert classify_market_regime(hurst) == "random_walk"


def test_iid_white_noise_returns_are_aggregated_without_double_differencing():
    rng = np.random.default_rng(1204)
    returns = pd.Series(rng.normal(size=8000))

    hurst = calculate_hurst_exponent(returns, input_type="returns")

    assert hurst == pytest.approx(0.5, abs=0.12)
    assert classify_market_regime(hurst) == "random_walk"


def test_mean_reverting_series_has_low_hurst_exponent():
    rng = np.random.default_rng(92)
    values = np.zeros(6000)
    for index in range(1, values.size):
        values[index] = 0.2 * values[index - 1] + rng.normal()

    hurst = calculate_hurst_exponent(pd.Series(values), max_lag=100)

    assert hurst < 0.4


def test_smooth_trending_geometric_series_has_high_hurst_exponent():
    geometric_trend = pd.Series(np.exp(np.linspace(0.0, 8.0, 1500)))

    hurst = calculate_hurst_exponent(geometric_trend, max_lag=100)

    assert hurst > 0.6


@pytest.mark.parametrize(
    ("hurst", "expected"),
    [
        (0.28, "mean_reverting"),
        (0.45, "random_walk"),
        (0.55, "random_walk"),
        (0.62, "trending"),
    ],
)
def test_classify_market_regime_thresholds(hurst, expected):
    assert classify_market_regime(hurst) == expected


def test_short_series_clamps_max_lag_to_available_history():
    rng = np.random.default_rng(11)
    short_series = pd.Series(np.cumsum(rng.normal(size=20)))

    hurst = calculate_hurst_exponent(short_series, max_lag=100)

    assert np.isfinite(hurst)
    assert np.isnan(calculate_hurst_exponent(pd.Series([1, 2, 3, 4, 5, 6])))


def test_rolling_hurst_preserves_index_and_leaves_incomplete_windows_nan():
    rng = np.random.default_rng(41)
    index = pd.RangeIndex(100, 200, name="observation")
    series = pd.Series(np.cumsum(rng.normal(size=100)), index=index, name="price")

    result = calculate_rolling_hurst(series, window=40, max_lag=20)

    assert result.index.equals(index)
    assert result.name == series.name
    assert result.iloc[:39].isna().all()
    assert result.iloc[39:].notna().all()


def test_rolling_hurst_does_not_compress_missing_observations():
    rng = np.random.default_rng(7)
    values = np.cumsum(rng.normal(size=40))
    values[20] = np.nan
    result = calculate_rolling_hurst(pd.Series(values), window=16, max_lag=10)

    assert result.iloc[15:20].notna().all()
    assert result.iloc[20:36].isna().all()
    assert result.iloc[36:].notna().all()


@pytest.mark.parametrize("window", [0, 7, 3.5, True])
def test_rolling_hurst_rejects_invalid_window(window):
    with pytest.raises(ValueError, match="window"):
        calculate_rolling_hurst(pd.Series(np.arange(20)), window=window)


def test_hurst_telemetry_contains_fit_quality_and_regime():
    rng = np.random.default_rng(100)
    series = pd.Series(np.cumsum(rng.normal(size=1000)))

    telemetry = hurst_telemetry(series, max_lag=60)

    assert telemetry["latency_ms"] >= 0
    assert telemetry["n_observations"] == len(series)
    assert telemetry["n_scales"] >= 3
    assert 0 <= telemetry["r_squared"] <= 1
    assert telemetry["regime"] == classify_market_regime(telemetry["hurst_exponent"])


def test_invalid_inputs_and_nonfinite_regime_are_rejected():
    with pytest.raises(ValueError, match="max_lag"):
        calculate_hurst_exponent(pd.Series(np.arange(20)), max_lag=3)
    with pytest.raises(ValueError, match="finite"):
        calculate_hurst_exponent(pd.Series([1.0, 2.0, np.inf, 4.0]))
    with pytest.raises(ValueError, match="finite"):
        classify_market_regime(float("nan"))
