"""Pure NumPy/Pandas scalar Kalman filter for time-varying market beta."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


def _finite_number(value: float, name: str) -> float:
    """Convert and validate a finite scalar, in O(1) time and space."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def kalman_predict_step(
    state_mean: float, state_cov: float, process_noise: float
) -> tuple[float, float]:
    r"""Predict a scalar random-walk state before receiving the next observation.

    For :math:`\beta_t=\beta_{t-1}+\eta_t`,
    :math:`\eta_t\sim N(0,Q)`, the prediction is
    :math:`\hat\beta_{t|t-1}=\hat\beta_{t-1|t-1}` and
    :math:`P_{t|t-1}=P_{t-1|t-1}+Q`. A beta estimate of 1.20 therefore remains
    1.20 in expectation, while its uncertainty grows by process noise before
    each new return pair. Complexity: :math:`O(1)` time and space.
    """
    mean = _finite_number(state_mean, "state_mean")
    covariance = _finite_number(state_cov, "state_cov")
    noise = _finite_number(process_noise, "process_noise")
    if covariance < 0.0 or noise < 0.0:
        raise ValueError("state_cov and process_noise must be non-negative")
    return mean, covariance + noise


def kalman_update_step(
    pred_mean: float,
    pred_cov: float,
    observation: float,
    market_return: float,
    obs_noise: float,
) -> tuple[float, float, float]:
    r"""Update scalar beta from one asset/market return measurement.

    The observation model is :math:`R_{a,t}=x_t\beta_t+\epsilon_t`, where
    :math:`x_t=R_{m,t}` and :math:`\epsilon_t\sim N(0,R)`. Innovation variance
    is :math:`S_t=x_t^2P_{t|t-1}+R`, gain
    :math:`K_t=P_{t|t-1}x_t/S_t`, and update
    :math:`\hat\beta_{t|t}=\hat\beta_{t|t-1}+K_t(R_{a,t}-x_t\hat\beta_{t|t-1})`,
    :math:`P_{t|t}=(1-K_tx_t)P_{t|t-1}`. Thus a stock beta can move from 1.20
    toward 0.85 as subsequent observations support lower market sensitivity.
    Complexity: :math:`O(1)` time and space.
    """
    mean = _finite_number(pred_mean, "pred_mean")
    covariance = _finite_number(pred_cov, "pred_cov")
    asset = _finite_number(observation, "observation")
    market = _finite_number(market_return, "market_return")
    noise = _finite_number(obs_noise, "obs_noise")
    if covariance < 0.0 or noise <= 0.0:
        raise ValueError("pred_cov must be non-negative and obs_noise strictly positive")
    innovation_variance = market**2 * covariance + noise
    gain = covariance * market / innovation_variance
    updated_mean = mean + gain * (asset - market * mean)
    updated_covariance = max((1.0 - gain * market) * covariance, 0.0)
    return float(updated_mean), float(updated_covariance), float(gain)


def _aligned_returns(
    asset_returns: pd.Series, market_returns: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Inner-align and validate paired return observations; O(n) time/space."""
    if not isinstance(asset_returns, pd.Series) or not isinstance(market_returns, pd.Series):
        raise TypeError("asset_returns and market_returns must be pandas Series")
    aligned = pd.concat(
        [asset_returns.rename("asset"), market_returns.rename("market")],
        axis=1,
        join="inner",
    ).dropna()
    try:
        values = aligned.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("returns must contain numeric values") from exc
    if values.ndim != 2 or values.shape[0] < 3 or not np.isfinite(values).all():
        raise ValueError("at least 3 paired finite return observations are required")
    if float(np.var(values[:, 1])) <= np.finfo(float).eps:
        raise ValueError("market returns have zero variance; beta is not identifiable")
    return aligned["asset"], aligned["market"]


def track_dynamic_beta(
    asset_returns: pd.Series, market_returns: pd.Series, delta: float = 1e-4
) -> tuple[pd.Series, pd.Series]:
    r"""Estimate time-varying beta and posterior state variance recursively.

    The state is a random walk, :math:`\beta_t=\beta_{t-1}+\eta_t`, and
    observations follow :math:`R_{a,t}=\beta_tR_{m,t}+\epsilon_t`. The filter
    starts from a fixed diffuse prior :math:`\beta_0=0, P_0=1` and a fixed
    daily-return observation variance :math:`R=10^{-4}`; neither is estimated
    from the complete sample or a future calibration window. Each observation
    is incorporated exactly once, so the estimate at date :math:`t` depends
    only on paired returns through :math:`t`. At each
    date the scalar Kalman predict/update equations advance the beta and its
    posterior variance. ``delta`` is the process variance :math:`Q` in beta
    units; larger values make beta react more quickly to regime shifts. The
    observation-noise variance is fixed ex ante rather than estimated from a
    calibration window. A
    shift from beta 1.20 to 0.85 is followed as new returns arrive. Complexity:
    :math:`O(T)` time and :math:`O(T)` output space.
    """
    q = _finite_number(delta, "delta")
    if q < 0.0:
        raise ValueError("delta must be non-negative")
    asset, market = _aligned_returns(asset_returns, market_returns)
    y = asset.to_numpy(dtype=float)
    x = market.to_numpy(dtype=float)
    n = y.size
    # Fixed, diffuse initialization avoids fitting on a window and then
    # replaying that same window into early filtered estimates.
    obs_noise = 1e-4
    means = np.empty(n, dtype=float)
    variances = np.empty(n, dtype=float)
    state_mean = 0.0
    state_cov = 1.0
    for t in range(n):
        predicted_mean, predicted_cov = kalman_predict_step(state_mean, state_cov, q)
        state_mean, state_cov, _ = kalman_update_step(
            predicted_mean, predicted_cov, y[t], x[t], obs_noise
        )
        means[t] = state_mean
        variances[t] = state_cov
    return (
        pd.Series(means, index=asset.index, name="dynamic_beta"),
        pd.Series(variances, index=asset.index, name="beta_variance"),
    )


def kalman_beta_telemetry(
    asset_returns: pd.Series, market_returns: pd.Series, delta: float = 1e-4
) -> dict[str, Any]:
    r"""Report dynamic-beta runtime, input shape, and posterior beta summaries.

    The telemetry includes mean/final :math:`\beta_t` and final state variance,
    making a move such as 1.20 to 0.85 visible to downstream diagnostics.
    Complexity: :math:`O(T)` time and space.
    """
    started = perf_counter()
    beta, variance = track_dynamic_beta(asset_returns, market_returns, delta)
    return {
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "input_shape": [int(beta.size)],
        "n_observations": int(beta.size),
        "mean_beta": float(beta.mean()),
        "final_beta": float(beta.iloc[-1]),
        "final_beta_variance": float(variance.iloc[-1]),
    }
