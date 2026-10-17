import numpy as np
import pandas as pd
import pytest

from app.core.risk.garch import (
    fit_garch11,
    forecast_future_volatility,
    garch_telemetry,
    predict_conditional_volatility,
)


def _simulate_garch(n=2500, omega=0.08, alpha=0.12, beta=0.82, seed=31):
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=n)
    variance = np.empty(n)
    returns = np.empty(n)
    variance[0] = omega / (1.0 - alpha - beta)
    returns[0] = np.sqrt(variance[0]) * shocks[0]
    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]
        returns[t] = np.sqrt(variance[t]) * shocks[t]
    return returns


def test_iid_gaussian_fit_returns_feasible_stationary_garch_parameters():
    rng = np.random.default_rng(406)
    returns = pd.Series(rng.normal(0, 0.01, 1200))
    fitted = fit_garch11(returns)
    assert fitted["omega"] > 0
    assert fitted["alpha"] >= 0
    assert fitted["beta"] >= 0
    assert fitted["alpha"] < 0.3
    assert fitted["persistence"] < 1.0
    assert fitted["unconditional_variance"] > 0
    assert np.isfinite(fitted["log_likelihood"])
    assert fitted["persistence_stationary"]


def test_synthetic_volatility_clusters_are_tracked_by_fitted_model():
    values = _simulate_garch()
    fitted = fit_garch11(values)
    conditional = predict_conditional_volatility(values, fitted)
    assert fitted["alpha"] > 0
    assert fitted["alpha"] + fitted["beta"] < 1.0
    assert conditional.shape == values.shape
    assert np.isfinite(conditional).all()
    # The volatility recursion should vary through the high and low variance
    # clusters instead of returning a constant unconditional estimate.
    assert conditional.max() > conditional.quantile(0.50) * 1.25
    assert fitted["optimization_success"]


def test_conditional_volatility_preserves_series_index():
    values = pd.Series(_simulate_garch(500), index=pd.date_range("2020-01-01", periods=500))
    fitted = fit_garch11(values)
    result = predict_conditional_volatility(values, fitted)
    assert result.index.equals(values.index)
    assert (result > 0).all()


def test_forecasts_converge_toward_long_run_volatility():
    params = {"omega": 0.01, "alpha": 0.10, "beta": 0.80}
    forecasts = forecast_future_volatility(params, last_volatility=2.0, horizon=20)
    long_run_vol = np.sqrt(params["omega"] / (1 - params["alpha"] - params["beta"]))
    assert forecasts.shape == (20,)
    assert np.all(np.diff(forecasts) < 0)
    assert abs(forecasts[-1] - long_run_vol) < abs(forecasts[0] - long_run_vol)


def test_telemetry_contains_optimizer_and_persistence_diagnostics():
    result = garch_telemetry(np.random.default_rng(1).normal(size=400))
    assert result["input_shape"] == [400]
    assert result["latency_ms"] >= 0
    assert result["optimization_success"]
    assert result["persistence_stationary"]


@pytest.mark.parametrize("bad", [[], [1.0, 2.0], [[1.0] * 30], [0.01] * 29, [np.nan] * 40])
def test_fit_rejects_too_small_or_invalid_inputs(bad):
    with pytest.raises(ValueError):
        fit_garch11(np.asarray(bad))


def test_forecast_rejects_invalid_parameters_and_horizon():
    with pytest.raises(ValueError):
        forecast_future_volatility({"omega": 1, "alpha": 0.6, "beta": 0.5}, 0.2)
    with pytest.raises(ValueError):
        forecast_future_volatility({"omega": 1, "alpha": 0.1, "beta": 0.8}, 0.2, horizon=0)
