"""Pure mean-reversion diagnostics for rolling signals and OU dynamics."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _validated_series_values(series: pd.Series, *, min_observations: int) -> np.ndarray:
    r"""Validate a numeric series and return its ordered non-missing values.

    The AR(1) fit uses consecutive daily observations :math:`X_t` and
    :math:`X_{t-1}`. Missing observations are rejected rather than dropped so
    two non-adjacent dates cannot be silently treated as one daily transition.
    For example, a P/E history moving from 25 toward 17 must remain
    chronological. Complexity: :math:`O(n)` time and :math:`O(n)` space.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if not series.index.is_unique:
        raise ValueError("series index must be unique")
    if not series.index.is_monotonic_increasing:
        raise ValueError("series index must be in chronological order")
    if isinstance(series.index, pd.DatetimeIndex) and len(series) >= 3:
        inferred_frequency = pd.infer_freq(series.index)
        if inferred_frequency not in {"D", "B", "C"}:
            raise ValueError(
                "DatetimeIndex must have a regular daily or business-daily frequency; "
                "regularize missing dates before OU estimation"
            )
    try:
        values = series.to_numpy(dtype=float, na_value=np.nan, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("series must contain numeric values") from exc
    if np.isinf(values).any():
        raise ValueError("series must be finite wherever present")
    if np.isnan(values).any():
        raise ValueError(
            "series cannot contain missing observations; fill or remove gaps "
            "without compressing the time index"
        )
    if values.size < min_observations:
        raise ValueError(f"series must contain at least {min_observations} non-missing observations")
    return values


def _fit_ar1(series: pd.Series) -> tuple[Any, np.ndarray, float]:
    r"""Fit :math:`X_t=a+\phi X_{t-1}+\epsilon_t` with OLS.

    This discrete regression is the sampled Ornstein-Uhlenbeck approximation
    to :math:`dX_t=\kappa(\theta-X_t)dt+\sigma dW_t`. A P/E series that
    fluctuates around 25 with fitted :math:`0<\phi<1` has a finite pull-back
    speed toward its estimated long-term level. Complexity: :math:`O(n)` time
    and :math:`O(n)` space for one predictor.
    """
    values = _validated_series_values(series, min_observations=3)
    lagged = values[:-1]
    current = values[1:]
    if float(np.var(lagged, ddof=1)) == 0.0:
        raise ValueError("series must vary to estimate an AR(1) process")
    design = sm.add_constant(lagged, has_constant="add")
    regression = sm.OLS(current, design, missing="raise").fit()
    residual_sigma = float(math.sqrt(max(float(regression.mse_resid), 0.0)))
    return regression, values, residual_sigma


def calculate_rolling_z_score(series: pd.Series, window: int = 60) -> pd.Series:
    r"""Calculate a trailing rolling Z-score using sample standard deviation.

    At time :math:`t`, :math:`Z_t=(X_t-\bar X_{t,w})/s_{t,w}`, where the mean
    and sample standard deviation use the current observation and previous
    :math:`w-1` values. For example, a P/E of 17 versus a 25 rolling mean is
    below its valuation norm; its Z-score expresses that gap in rolling
    standard deviations. The first :math:`w-1` outputs are NaN because a full
    window is required. Complexity: :math:`O(n)` time using Pandas' rolling
    kernels and :math:`O(n)` output space.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)):
        raise ValueError("window must be an integer between 2 and len(series)")
    if window < 2 or window > len(series):
        raise ValueError("window must be an integer between 2 and len(series)")
    try:
        values = series.to_numpy(dtype=float, na_value=np.nan, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("series must contain numeric values") from exc
    if np.isinf(values).any():
        raise ValueError("series must be finite wherever present")
    rolling = series.astype(float).rolling(window=int(window), min_periods=int(window))
    mean = rolling.mean()
    std = rolling.std(ddof=1)
    result = (series.astype(float) - mean) / std
    result.name = series.name
    return result


def estimate_ornstein_uhlenbeck_params(series: pd.Series) -> dict[str, Any]:
    r"""Estimate OU reversion speed, long-run level, and diffusion volatility.

    For daily observations, OLS fits :math:`X_t=a+\phi X_{t-1}+\epsilon_t`.
    When :math:`0<\phi<1`, the exact sampled OU mapping is
    :math:`\kappa=-\ln(\phi)`, :math:`\theta=a/(1-\phi)`, and
    :math:`\sigma=s_\epsilon\sqrt{2\kappa/(1-\phi^2)}` for unit time step.
    This corresponds to :math:`dX_t=\kappa(\theta-X_t)dt+\sigma dW_t`.
    For intuition, if a P/E gap from 25 to 17 halves in 30 days, the fitted
    half-life is 30 days and :math:`\kappa=\ln(2)/30`. Non-OU or nonstationary
    coefficients are marked rather than forced into finite OU parameters.
    Complexity: :math:`O(n)` time and :math:`O(n)` space.
    """
    start = perf_counter()
    regression, values, residual_sigma = _fit_ar1(series)
    intercept = float(regression.params[0])
    phi = float(regression.params[1])
    if phi > 0.0:
        kappa = float(-math.log(phi))
    else:
        kappa = float("nan")
    is_stationary = bool(abs(phi) < 1.0)
    is_ou_mean_reverting = bool(0.0 < phi < 1.0)
    theta = (
        float(intercept / (1.0 - phi))
        if not math.isclose(phi, 1.0, rel_tol=0.0, abs_tol=1e-12)
        else float("nan")
    )
    sigma = (
        float(residual_sigma * math.sqrt(2.0 * kappa / (1.0 - phi**2)))
        if is_ou_mean_reverting
        else float("nan")
    )
    half_life_days = (
        float(math.log(2.0) / kappa)
        if is_ou_mean_reverting and kappa > 0.0
        else (float("inf") if math.isfinite(kappa) and kappa <= 0.0 else float("nan"))
    )
    return {
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "ar1_intercept": intercept,
        "ar1_coefficient": phi,
        "residual_sigma": residual_sigma,
        "r_squared": float(regression.rsquared),
        "n_observations": int(values.size),
        "is_stationary": is_stationary,
        "is_ou_mean_reverting": is_ou_mean_reverting,
        "half_life_days": half_life_days,
        "latency_ms": float((perf_counter() - start) * 1000.0),
    }


def calculate_half_life(series: pd.Series) -> float:
    r"""Calculate the OU half-life in observation days from the fitted speed.

    The half-life is :math:`t_{1/2}=\ln(2)/\kappa`, the time for the expected
    distance from :math:`\theta` to shrink by one-half. For example, a P/E
    ratio at 17 reverting toward 25 with :math:`\kappa=\ln(2)/30` has a
    30-observation-day half-life. If :math:`\kappa\le0`, the process is not
    mean-reverting and this function returns infinity; an undefined OU mapping
    returns NaN. Complexity: :math:`O(n)` time and :math:`O(n)` temporary
    space because the AR(1) parameters are estimated from the series.
    """
    params = estimate_ornstein_uhlenbeck_params(series)
    kappa = float(params["kappa"])
    if not math.isfinite(kappa):
        return float("nan")
    if kappa <= 0.0:
        return float("inf")
    return float(math.log(2.0) / kappa)


def mean_reversion_telemetry(series: pd.Series) -> dict[str, Any]:
    r"""Report AR(1) processing time, half-life, and stationarity diagnostics.

    The telemetry exposes :math:`t_{1/2}=\ln(2)/\kappa`, a stationarity flag,
    the estimated :math:`\kappa`, :math:`\theta`, and :math:`\sigma`, plus
    sample size. For example, a P/E series reverting from 17 toward 25 with a
    30-day half-life reports that duration and a positive stationarity flag.
    Complexity: :math:`O(n)` time and :math:`O(n)` space for one AR(1) fit.
    """
    start = perf_counter()
    params = estimate_ornstein_uhlenbeck_params(series)
    telemetry = {
        "latency_ms": float((perf_counter() - start) * 1000.0),
        "half_life_days": float(params["half_life_days"]),
        "is_stationary": bool(params["is_stationary"]),
        "is_ou_mean_reverting": bool(params["is_ou_mean_reverting"]),
        "kappa": float(params["kappa"]),
        "theta": float(params["theta"]),
        "sigma": float(params["sigma"]),
        "n_observations": int(params["n_observations"]),
    }
    return telemetry
