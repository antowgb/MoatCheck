"""Tests for the pure portfolio return math engine."""

import warnings

import numpy as np
import pandas as pd
import pytest

from app.core.portfolio.returns import (
    annualized_covariance_matrix,
    annualized_mean_returns,
    data_shape_info,
    log_to_simple_return,
    measure_execution_latency,
    to_log_returns,
    to_simple_returns,
)


def test_empty_prices_return_empty_frames_with_columns() -> None:
    prices = pd.DataFrame(columns=["A", "B"], dtype=float)

    assert to_log_returns(prices).empty
    assert list(to_log_returns(prices).columns) == ["A", "B"]
    assert to_simple_returns(prices).empty
    assert list(to_simple_returns(prices).columns) == ["A", "B"]


def test_single_asset_returns_match_manual_values() -> None:
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})

    pd.testing.assert_frame_equal(
        to_log_returns(prices),
        pd.DataFrame({"A": [np.log(1.1), np.log(0.9)]}, index=[1, 2]),
    )
    pd.testing.assert_frame_equal(
        to_simple_returns(prices),
        pd.DataFrame({"A": [0.1, -0.1]}, index=[1, 2]),
    )


def test_log_returns_remain_finite_for_extreme_price_scales() -> None:
    prices = pd.DataFrame({"A": [1e-308, 1e308]})

    result = to_log_returns(prices)

    assert np.isfinite(result.loc[1, "A"])
    assert result.loc[1, "A"] == pytest.approx(
        np.log(1e308) - np.log(1e-308)
    )


@pytest.mark.parametrize("bad_price", [0.0, -1.0])
def test_non_positive_prices_raise(bad_price: float) -> None:
    prices = pd.DataFrame({"A": [100.0, bad_price]})

    with pytest.raises(ValueError, match="strictly positive"):
        to_log_returns(prices)
    with pytest.raises(ValueError, match="strictly positive"):
        to_simple_returns(prices)


def test_annualized_statistics_match_manual_sample_calculations() -> None:
    returns = pd.DataFrame(
        {"A": [0.01, 0.03, -0.02], "B": [0.02, -0.01, 0.04]},
        index=[10, 11, 12],
    )
    days = 2
    expected_mean = returns.mean() * days
    expected_cov = returns.cov(ddof=1) * days

    pd.testing.assert_series_equal(annualized_mean_returns(returns, days), expected_mean)
    pd.testing.assert_frame_equal(annualized_covariance_matrix(returns, days), expected_cov)


def test_single_asset_covariance_keeps_matrix_shape() -> None:
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01]})

    result = annualized_covariance_matrix(returns, 252)

    assert result.shape == (1, 1)
    assert result.loc["A", "A"] == pytest.approx(returns["A"].var(ddof=1) * 252)


def test_statistics_handle_empty_inputs_and_validate_trading_days() -> None:
    returns = pd.DataFrame(columns=["A"], dtype=float)
    one_row = pd.DataFrame({"A": [0.01], "B": [np.nan]})

    assert annualized_mean_returns(returns).isna().all()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        covariance = annualized_covariance_matrix(returns)
        one_row_covariance = annualized_covariance_matrix(one_row)
    assert covariance.shape == (1, 1)
    assert covariance.isna().all().all()
    assert one_row_covariance.shape == (2, 2)
    assert one_row_covariance.isna().all().all()
    with pytest.raises(ValueError, match="positive integer"):
        annualized_mean_returns(returns, 0)
    with pytest.raises(ValueError, match="positive integer"):
        annualized_covariance_matrix(returns, 2.5)  # type: ignore[arg-type]


def test_log_to_simple_return_exact_known_value() -> None:
    assert log_to_simple_return(np.log(1.25)) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="finite"):
        log_to_simple_return(float("inf"))


def test_missing_observations_are_dropped_from_return_rows() -> None:
    prices = pd.DataFrame({"A": [100.0, np.nan, 110.0], "B": [50.0, 55.0, 60.0]})

    assert to_log_returns(prices).empty
    assert to_simple_returns(prices).empty


def test_execution_telemetry_reports_latency_and_shapes() -> None:
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0]})

    result, telemetry = measure_execution_latency(to_simple_returns, prices)

    assert result.shape == (2, 1)
    assert telemetry["latency_ms"] >= 0
    assert telemetry["input_shape"] == {
        "rows": 3, "columns": 1, "shape": [3, 1], "missing_values": 0
    }
    assert telemetry["output_shape"] == {
        "rows": 2, "columns": 1, "shape": [2, 1], "missing_values": 0
    }
    assert data_shape_info(result)["shape"] == [2, 1]
