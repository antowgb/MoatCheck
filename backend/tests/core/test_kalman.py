import numpy as np
import pandas as pd
import pytest

from app.core.risk.kalman import (
    kalman_beta_telemetry,
    kalman_predict_step,
    kalman_update_step,
    track_dynamic_beta,
)


def test_predict_and_update_scalar_equations():
    predicted_mean, predicted_cov = kalman_predict_step(1.2, 0.4, 0.01)
    assert predicted_mean == pytest.approx(1.2)
    assert predicted_cov == pytest.approx(0.41)
    mean, covariance, gain = kalman_update_step(predicted_mean, predicted_cov, 0.02, 0.01, 1e-5)
    assert gain > 0
    assert covariance < predicted_cov
    assert mean > predicted_mean


def test_constant_beta_filter_converges_close_to_ols_beta():
    rng = np.random.default_rng(141)
    market = rng.normal(0.0002, 0.01, 1600)
    asset = 1.35 * market + rng.normal(0.0, 0.002, market.size)
    index = pd.date_range("2020-01-01", periods=market.size, freq="B")
    beta, variance = track_dynamic_beta(pd.Series(asset, index=index), pd.Series(market, index=index))
    ols_beta = float(market @ asset / (market @ market))
    assert beta.index.equals(index)
    assert variance.index.equals(index)
    assert beta.iloc[-1] == pytest.approx(ols_beta, abs=0.15)
    assert (variance >= 0).all()


def test_beta_estimates_are_causal_and_do_not_use_future_observations():
    rng = np.random.default_rng(199)
    n, cutoff = 120, 70
    market = rng.normal(0.0, 0.01, n)
    asset = 1.1 * market + rng.normal(0.0, 0.002, n)
    changed_market = market.copy()
    changed_asset = asset.copy()
    changed_market[cutoff + 1 :] = rng.normal(0.0, 0.03, n - cutoff - 1)
    changed_asset[cutoff + 1 :] = rng.normal(0.0, 0.05, n - cutoff - 1)

    beta, _ = track_dynamic_beta(pd.Series(asset), pd.Series(market))
    changed_beta, _ = track_dynamic_beta(
        pd.Series(changed_asset), pd.Series(changed_market)
    )

    # The value at t may use observations through t, but no observations t+k.
    assert np.allclose(beta.iloc[: cutoff + 1], changed_beta.iloc[: cutoff + 1])


def test_filter_tracks_step_change_in_market_sensitivity():
    rng = np.random.default_rng(73)
    n = 1200
    market = rng.normal(0.0, 0.012, n)
    true_beta = np.r_[np.full(n // 2, 1.2), np.full(n // 2, 0.55)]
    asset = true_beta * market + rng.normal(0.0, 0.0015, n)
    beta, _ = track_dynamic_beta(pd.Series(asset), pd.Series(market), delta=2e-4)
    assert beta.iloc[n // 2 - 1] > 0.95
    assert beta.iloc[-1] < 0.75
    assert beta.iloc[-1] < beta.iloc[n // 2 - 1]


def test_zero_market_variance_is_rejected_as_unidentifiable():
    market = pd.Series(np.zeros(100))
    asset = pd.Series(np.random.default_rng(1).normal(size=100))
    with pytest.raises(ValueError, match="zero variance"):
        track_dynamic_beta(asset, market)


def test_alignment_drops_missing_pairs_and_telemetry_reports_shape():
    market = pd.Series(np.random.default_rng(3).normal(size=100), index=np.arange(100))
    asset = 0.8 * market + pd.Series(np.random.default_rng(4).normal(scale=0.01, size=100), index=market.index)
    asset.iloc[5] = np.nan
    beta, _ = track_dynamic_beta(asset, market)
    assert len(beta) == 99
    telemetry = kalman_beta_telemetry(asset, market)
    assert telemetry["input_shape"] == [99]
    assert np.isfinite(telemetry["mean_beta"])
    assert np.isfinite(telemetry["final_beta"])
    assert telemetry["latency_ms"] >= 0


@pytest.mark.parametrize("bad_delta", [-1.0, np.nan, np.inf])
def test_invalid_process_noise_is_rejected(bad_delta):
    with pytest.raises(ValueError):
        track_dynamic_beta(pd.Series(np.arange(10.0)), pd.Series(np.arange(10.0)), delta=bad_delta)
