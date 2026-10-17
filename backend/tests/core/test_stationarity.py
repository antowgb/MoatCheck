import numpy as np
import pandas as pd
import pytest

from app.core.risk.stationarity import (
    make_series_stationary,
    stationarity_telemetry,
    test_augmented_dickey_fuller as run_adf,
    test_cointegration_pair as run_cointegration,
)


def test_stationary_ar1_is_detected():
    rng = np.random.default_rng(140)
    noise = rng.normal(0, 1, 600)
    values = np.empty(noise.size)
    values[0] = noise[0]
    for i in range(1, values.size):
        values[i] = 0.35 * values[i - 1] + noise[i]
    result = run_adf(pd.Series(values))
    assert result["p_value"] < 0.05
    assert result["is_stationary"]
    assert set(result["critical_values"]) == {"1%", "5%", "10%"}
    assert result["used_lags"] >= 0


def test_random_walk_levels_are_nonstationary_and_first_difference_resolves():
    rng = np.random.default_rng(713)
    levels = pd.Series(100 + np.cumsum(rng.normal(0, 1, 700)), name="price")
    level_test = run_adf(levels)
    assert level_test["p_value"] > 0.05
    transformed, degree = make_series_stationary(levels, max_differencing=2)
    assert degree == 1
    assert len(transformed) == len(levels) - 1
    assert transformed.name == "price"
    assert run_adf(transformed)["is_stationary"]


def test_cointegrated_random_walk_pair_is_detected():
    rng = np.random.default_rng(45)
    common_walk = np.cumsum(rng.normal(0, 1, 800))
    spread_noise = rng.normal(0, 1, 800)
    spread = np.empty(800)
    spread[0] = spread_noise[0]
    for i in range(1, spread.size):
        spread[i] = 0.25 * spread[i - 1] + spread_noise[i]
    dates = pd.date_range("2010-01-01", periods=800, freq="B")
    asset_a = pd.Series(100 + common_walk, index=dates)
    asset_b = pd.Series(20 + 1.7 * common_walk + spread, index=dates)
    result = run_cointegration(asset_a, asset_b)
    assert result["p_value"] < 0.05
    assert result["is_cointegrated"]
    assert result["n_observations"] == 800


def test_cointegration_aligns_only_common_dates():
    rng = np.random.default_rng(19)
    a = pd.Series(np.cumsum(rng.normal(size=200)), index=pd.RangeIndex(200))
    b = pd.Series(np.cumsum(rng.normal(size=80)), index=pd.RangeIndex(120, 200))
    result = run_cointegration(a, b)
    assert result["n_observations"] == 80


def test_telemetry_reports_p_value_stationarity_and_shape():
    rng = np.random.default_rng(2)
    result = stationarity_telemetry(pd.Series(rng.normal(size=200)))
    assert result["input_shape"] == [200]
    assert "p_value" in result
    assert isinstance(result["is_stationary"], bool)
    assert result["latency_ms"] >= 0


@pytest.mark.parametrize("series,regression", [
    (pd.Series([1.0, np.nan, 2.0]), "c"),
    (pd.Series(np.arange(20.0)), "bad"),
])
def test_invalid_adf_inputs_raise(series, regression):
    with pytest.raises(ValueError):
        run_adf(series, regression)


def test_max_differencing_must_be_nonnegative_integer():
    with pytest.raises(ValueError):
        make_series_stationary(pd.Series(np.arange(10.0)), max_differencing=-1)
