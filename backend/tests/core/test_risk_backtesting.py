import numpy as np
import pandas as pd
import pytest

from app.core.risk.backtesting import (
    christoffersen_independence_test as run_independence_test,
    evaluate_var_model_backtest,
    kupiec_pof_test as run_kupiec_test,
    var_backtest_telemetry,
)


def _evenly_spaced_exceptions(n=500, count=25):
    returns = np.zeros(n)
    returns[np.arange(count) * (n // count)] = -0.10
    var = np.full(n, -0.05)
    return returns, var


def test_calibrated_model_has_good_coverage_and_green_traffic_light():
    returns, var = _evenly_spaced_exceptions()
    kupiec = run_kupiec_test(returns, var, confidence_level=0.95)
    result = evaluate_var_model_backtest(returns, var, confidence_level=0.95)
    assert kupiec["exception_count"] == 25
    assert kupiec["expected_exception_count"] == pytest.approx(25)
    assert kupiec["p_value"] > 0.05
    assert not kupiec["reject_null"]
    assert result["traffic_light_zone"] == "Green"
    assert result["input_shape"] == [500]


def test_severely_underestimated_var_is_rejected_and_red():
    n = 500
    returns = np.zeros(n)
    returns[:100] = -0.10
    var = np.full(n, -0.05)
    result = evaluate_var_model_backtest(returns, var, confidence_level=0.95)
    assert result["kupiec"]["exception_count"] == 100
    assert result["kupiec"]["p_value"] < 0.05
    assert result["kupiec"]["reject_null"]
    assert result["traffic_light_zone"] == "Red"


def test_clustered_exceptions_fail_independence_but_pass_kupiec_coverage():
    n = 500
    returns = np.zeros(n)
    returns[:25] = -0.10
    var = np.full(n, -0.05)
    kupiec = run_kupiec_test(returns, var, confidence_level=0.95)
    independence = run_independence_test(returns, var, confidence_level=0.95)
    assert kupiec["p_value"] > 0.05
    assert not kupiec["reject_null"]
    assert independence["p_value"] < 0.05
    assert independence["reject_independence"]
    assert independence["n11"] == 24


def test_series_indexes_align_and_nans_are_pairwise_dropped():
    returns = pd.Series([-0.1, 0.0, -0.1], index=["a", "b", "c"])
    var = pd.Series([-0.05, np.nan, -0.05], index=["a", "b", "c"])
    result = run_kupiec_test(returns, var, confidence_level=0.8)
    assert result["total_observations"] == 2
    assert result["exception_count"] == 2


def test_telemetry_includes_latency_rate_zone_and_shape():
    returns, var = _evenly_spaced_exceptions()
    telemetry = var_backtest_telemetry(returns, var)
    assert telemetry["input_shape"] == [500]
    assert telemetry["exception_rate"] == pytest.approx(0.05)
    assert telemetry["traffic_light_zone"] == "Green"
    assert telemetry["latency_ms"] >= 0


@pytest.mark.parametrize("bad_confidence", [0.0, 1.0, np.nan])
def test_invalid_confidence_level_raises(bad_confidence):
    with pytest.raises(ValueError):
        run_kupiec_test(np.zeros(10), np.full(10, -0.1), bad_confidence)


def test_mismatched_array_lengths_raise():
    with pytest.raises(ValueError):
        run_kupiec_test(np.zeros(10), np.zeros(9))
