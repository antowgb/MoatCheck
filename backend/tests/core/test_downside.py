import math

import numpy as np
import pandas as pd
import pytest

from app.core.risk.downside import (
    calculate_downside_deviation,
    calculate_drawdown_series,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    downside_risk_telemetry,
)


def test_downside_deviation_uses_full_period_count_and_annualizes():
    returns = np.array([0.02, -0.01, -0.03])
    expected = np.sqrt(252 * ((0.01**2 + 0.03**2) / 3))
    assert calculate_downside_deviation(returns) == pytest.approx(expected)


def test_strictly_positive_returns_have_zero_downside_and_sortino_infinity():
    returns = pd.Series([0.01, 0.02, 0.03])
    assert calculate_downside_deviation(returns) == 0.0
    assert calculate_sortino_ratio(returns) == math.inf
    assert calculate_sortino_ratio(returns, risk_free_rate=15.12) == -math.inf


def test_sortino_negative_excess_with_zero_downside_is_negative_infinity():
    assert calculate_sortino_ratio(np.array([0.01, 0.02]), risk_free_rate=10.0) == -math.inf


def test_drawdown_path_tracks_running_peak_and_preserves_index():
    returns = pd.Series([0.10, -0.20, 0.25], index=pd.date_range("2025-01-01", periods=3))
    result = calculate_drawdown_series(returns)
    assert result.index.equals(returns.index)
    assert result.tolist() == pytest.approx([0.0, -0.20, 0.0])


def test_full_loss_produces_100_percent_drawdown_trajectory():
    returns = pd.Series([0.0, -1.0, 0.0], index=["peak", "trough", "after"])
    result = calculate_drawdown_series(returns)
    assert result.tolist() == [0.0, -1.0, -1.0]
    summary = calculate_max_drawdown(returns)
    assert summary["max_drawdown"] == -1.0
    assert summary["peak_date"] == "peak"
    assert summary["trough_date"] == "trough"
    assert summary["recovery_duration"] is None


def test_drawdown_recovery_duration_and_unrecovered_peak():
    recovered = calculate_max_drawdown(pd.Series([0.20, -0.25, 1 / 3]))
    assert recovered["max_drawdown"] == pytest.approx(-0.25)
    assert recovered["peak_date"] == 0
    assert recovered["trough_date"] == 1
    assert recovered["recovery_duration"] == 1

    unrecovered = calculate_max_drawdown(pd.Series([0.20, -0.10, -0.10]))
    assert unrecovered["max_drawdown"] < 0
    assert unrecovered["recovery_duration"] is None


def test_monotonically_decreasing_wealth_has_continuous_drawdown():
    drawdowns = calculate_drawdown_series(np.array([-0.10, -0.10, -0.10]))
    assert drawdowns.tolist() == pytest.approx([-0.10, -0.19, -0.271])


def test_telemetry_reports_shape_metrics_and_latency():
    telemetry = downside_risk_telemetry(pd.Series([-0.02, 0.01, -0.01]))
    assert telemetry["input_shape"] == [3]
    assert telemetry["downside_deviation"] > 0
    assert telemetry["max_drawdown"] < 0
    assert telemetry["latency_ms"] >= 0


@pytest.mark.parametrize("bad", [[], [[0.01, 0.02]], [np.nan], [np.inf], [-1.01]])
def test_invalid_return_inputs_raise(bad):
    with pytest.raises((TypeError, ValueError)):
        calculate_drawdown_series(bad)
