import numpy as np
import pandas as pd
import pytest

from app.core.risk.stress_testing import (
    apply_macro_scenario_shock,
    replay_historical_crisis,
    run_reverse_stress_test,
    stress_testing_telemetry,
)


def test_zero_weight_portfolio_has_zero_crisis_return_and_drawdown():
    dates = pd.date_range("2020-03-01", periods=4)
    returns = pd.DataFrame({"A": [-0.10, -0.20, 0.10, -0.05], "B": [0.03, -0.4, 0.0, -0.1]}, index=dates)
    result = replay_historical_crisis(np.array([0.0, 0.0]), returns, "2020-03-02", "2020-03-04")
    assert result["n_observations"] == 3
    assert result["cumulative_return"] == 0.0
    assert result["worst_drawdown"] == 0.0
    assert np.all(result["drawdown_path"].to_numpy() == 0.0)


def test_single_asset_replay_matches_asset_peak_to_trough_loss():
    dates = pd.date_range("2008-09-15", periods=3)
    returns = pd.DataFrame({"market": [-0.10, -0.20, 0.10]}, index=dates)
    result = replay_historical_crisis(np.array([1.0]), returns, "2008-09-15", "2008-09-17")
    assert result["cumulative_return"] == pytest.approx(-0.208)
    assert result["worst_drawdown"] == pytest.approx(-0.28)
    assert result["asset_contributions"]["market"] == pytest.approx(-0.30)
    assert result["input_shape"] == [3, 1]


def test_crisis_window_is_inclusive_and_worst_drawdown_is_reported():
    dates = pd.date_range("2020-03-01", periods=5)
    returns = pd.DataFrame({"a": [0.1, -0.2, -0.1, 0.2, 0.0]}, index=dates)
    result = replay_historical_crisis(np.array([1.0]), returns, "2020-03-02", "2020-03-04")
    assert result["n_observations"] == 3
    assert result["input_shape"] == [3, 1]
    assert result["worst_drawdown"] < 0


def test_reverse_stress_matches_closed_form_minimum_norm_solution():
    weights = np.array([0.6, 0.4])
    loadings = pd.DataFrame(
        [[1.0, 0.5], [0.2, 1.0]], index=["asset_a", "asset_b"], columns=["growth", "rates"]
    )
    target = -0.30
    result = run_reverse_stress_test(weights, np.eye(2), loadings, target)
    exposure = weights @ loadings.to_numpy()
    expected = target * exposure / (exposure @ exposure)
    assert np.allclose(result["factor_shocks"].to_numpy(), expected, atol=1e-8)
    assert result["achieved_portfolio_loss"] == pytest.approx(target, abs=1e-8)
    assert result["minimum_shock_magnitude"] == pytest.approx(np.linalg.norm(expected))
    assert result["optimization_success"]


def test_zero_exposure_reverse_stress_is_infeasible():
    loadings = pd.DataFrame([[1.0, -1.0], [-1.0, 1.0]], columns=["f1", "f2"])
    with pytest.raises(ValueError, match="zero factor exposure"):
        run_reverse_stress_test(np.array([0.5, 0.5]), np.eye(2), loadings)


def test_macro_factor_shock_maps_named_factors_to_portfolio_impact():
    weights = np.array([0.5, 0.5])
    loadings = pd.DataFrame(
        [[-2.0, 1.0], [-1.0, 0.5]], columns=["rates", "growth"]
    )
    impact = apply_macro_scenario_shock(weights, loadings, {"rates": 0.02, "growth": -0.15})
    expected = weights @ loadings.to_numpy() @ np.array([0.02, -0.15])
    assert impact == pytest.approx(expected)
    assert impact < 0


def test_telemetry_reports_shock_magnitude_and_input_shape():
    weights = np.array([0.5, 0.5])
    loadings = pd.DataFrame([[1.0, 0.2], [0.5, 1.0]], columns=["market", "rates"])
    result = stress_testing_telemetry(weights, np.eye(2), loadings, -0.25)
    assert result["minimum_shock_magnitude"] > 0
    assert result["input_shape"] == [2, 2]
    assert result["latency_ms"] >= 0


def test_invalid_crisis_dates_and_macro_factor_names_raise():
    returns = pd.DataFrame({"a": [0.0, -0.1]}, index=pd.date_range("2020-01-01", periods=2))
    with pytest.raises(ValueError):
        replay_historical_crisis(np.array([1.0]), returns, "2020-01-02", "2020-01-01")
    with pytest.raises(ValueError):
        apply_macro_scenario_shock(np.array([1.0]), pd.DataFrame([[1.0]], columns=["rates"]), {"growth": 0.1})
