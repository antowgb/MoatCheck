"""Pure NumPy/SciPy GARCH(1,1) estimation and volatility forecasting."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_PERSISTENCE_LIMIT = 0.999


def _returns_array(returns: pd.Series | np.ndarray, *, minimum: int = 30) -> tuple[np.ndarray, pd.Index]:
    """Validate finite one-dimensional simple returns and preserve index; O(n)."""
    if not isinstance(returns, (pd.Series, np.ndarray)):
        raise TypeError("returns must be a pandas Series or NumPy array")
    try:
        values = (returns.to_numpy(dtype=float, na_value=np.nan, copy=True)
                  if isinstance(returns, pd.Series) else np.asarray(returns, dtype=float))
    except (TypeError, ValueError) as exc:
        raise TypeError("returns must contain numeric values") from exc
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite observations")
    if values.size < minimum:
        raise ValueError(f"at least {minimum} observations are required")
    if isinstance(returns, pd.Series):
        index = returns.index
    else:
        index = pd.RangeIndex(values.size)
    return values, index


def _variance_path(
    residuals: np.ndarray, omega: float, alpha: float, beta: float, initial_variance: float
) -> np.ndarray:
    """Compute conditional variances by the GARCH recursion in O(n) time/space."""
    variance = np.empty(residuals.size, dtype=float)
    variance[0] = max(initial_variance, omega / max(1.0 - alpha - beta, 1e-6))
    for t in range(1, residuals.size):
        variance[t] = omega + alpha * residuals[t - 1] ** 2 + beta * variance[t - 1]
    return np.maximum(variance, np.finfo(float).tiny)


def _validate_params(garch_params: dict[str, Any]) -> tuple[float, float, float, float]:
    """Validate GARCH parameters and return omega, alpha, beta, mean; O(1)."""
    if not isinstance(garch_params, dict):
        raise TypeError("garch_params must be a dictionary")
    try:
        omega = float(garch_params["omega"])
        alpha = float(garch_params["alpha"])
        beta = float(garch_params["beta"])
        mean = float(garch_params.get("mu", 0.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("garch_params must include numeric omega, alpha, and beta") from exc
    if not all(math.isfinite(x) for x in (omega, alpha, beta, mean)):
        raise ValueError("GARCH parameters must be finite")
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0:
        raise ValueError("parameters must satisfy omega > 0, alpha,beta >= 0, alpha + beta < 1")
    return omega, alpha, beta, mean


def fit_garch11(returns: pd.Series | np.ndarray) -> dict[str, Any]:
    r"""Fit a zero-mean-around-:math:`\mu` Gaussian GARCH(1,1) by MLE.

    Conditional variance follows
    :math:`\sigma_t^2=\omega+\alpha\epsilon_{t-1}^2+\beta\sigma_{t-1}^2`,
    where :math:`\epsilon_t=R_t-\mu`. The Gaussian log-likelihood is
    :math:`\ell=-\frac12\sum_t[\log(2\pi)+\log(\sigma_t^2)+
    \epsilon_t^2/\sigma_t^2]`. SLSQP enforces positive :math:`\omega`,
    nonnegative :math:`\alpha,\beta`, and :math:`\alpha+\beta<1` so the
    unconditional variance :math:`\omega/(1-\alpha-\beta)` exists. A 12%
    volatility baseline can rise toward 28% after a large shock through the
    :math:`\alpha\epsilon^2` term, then decay according to persistence. The
    likelihood evaluation is :math:`O(n)`; optimization is approximately
    :math:`O(K n I)` for :math:`K` parameter starts and :math:`I` iterations.
    """
    started = perf_counter()
    values, _ = _returns_array(returns)
    mu = float(np.mean(values))
    residuals = values - mu
    sample_variance = float(np.var(residuals, ddof=1))
    if not math.isfinite(sample_variance) or sample_variance <= np.finfo(float).tiny:
        raise ValueError("returns must have non-zero variance")
    initial = sample_variance
    lower_omega = max(sample_variance * 1e-10, np.finfo(float).tiny)
    upper_omega = max(sample_variance * 10.0, lower_omega * 10.0)

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        omega, alpha, beta = map(float, parameters)
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e100
        variances = _variance_path(residuals, omega, alpha, beta, initial)
        ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(variances) + residuals**2 / variances)
        return float(-ll) if math.isfinite(float(ll)) else 1e100

    constraints = [{"type": "ineq", "fun": lambda p: _PERSISTENCE_LIMIT - p[1] - p[2]}]
    starts = [
        np.array([sample_variance * (1.0 - a - b), a, b])
        for a, b in ((0.05, 0.90), (0.10, 0.80), (0.05, 0.50), (0.01, 0.10))
    ]
    results = [
        minimize(
            negative_log_likelihood,
            x0=np.maximum(start, [lower_omega, 1e-6, 1e-6]),
            method="SLSQP",
            bounds=[(lower_omega, upper_omega), (0.0, 0.999), (0.0, 0.999)],
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        for start in starts
    ]
    feasible = [
        result for result in results
        if np.isfinite(result.fun) and result.x[0] > 0
        and result.x[1] >= 0 and result.x[2] >= 0
        and result.x[1] + result.x[2] < 1.0
    ]
    if not feasible:
        raise RuntimeError("GARCH maximum-likelihood optimization failed to find feasible parameters")
    best = min(feasible, key=lambda result: float(result.fun))
    omega, alpha, beta = map(float, best.x)
    persistence = alpha + beta
    log_likelihood = -float(best.fun)
    elapsed_ms = float((perf_counter() - started) * 1000.0)
    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "mu": mu,
        "persistence": persistence,
        "unconditional_variance": omega / (1.0 - persistence),
        "log_likelihood": log_likelihood,
        "optimization_success": bool(best.success),
        "optimization_status": int(best.status),
        "optimization_message": str(best.message),
        "n_iterations": int(getattr(best, "nit", 0)),
        "persistence_stationary": bool(persistence < 1.0),
        "latency_ms": elapsed_ms,
        "input_shape": [int(values.size)],
        "n_observations": int(values.size),
    }


def predict_conditional_volatility(
    returns: pd.Series | np.ndarray, garch_params: dict[str, Any]
) -> pd.Series:
    r"""Compute the in-sample conditional standard deviation at each timestamp.

    The recursion is :math:`h_t=\omega+\alpha(R_{t-1}-\mu)^2+\beta h_{t-1}`;
    output is :math:`\sigma_t=\sqrt{h_t}` and retains the input index. A shock
    that raises volatility from 12% toward 28% affects following observations
    through :math:`\alpha\epsilon_t^2`. Complexity: :math:`O(n)` time and
    :math:`O(n)` space.
    """
    values, index = _returns_array(returns, minimum=1)
    omega, alpha, beta, mu = _validate_params(garch_params)
    residuals = values - mu
    unconditional = omega / (1.0 - alpha - beta)
    variance = _variance_path(residuals, omega, alpha, beta, unconditional)
    return pd.Series(np.sqrt(variance), index=index, name="conditional_volatility")


def forecast_future_volatility(
    garch_params: dict[str, Any], last_volatility: float, horizon: int = 10
) -> np.ndarray:
    r"""Forecast future conditional standard deviations at a fixed horizon.

    With :math:`\rho=\alpha+\beta` and long-run variance
    :math:`\bar h=\omega/(1-\rho)`, the multi-step variance forecast is
    :math:`E_t[h_{t+k}]=\bar h+\rho^k(h_t-\bar h)`. Here ``last_volatility``
    is :math:`\sqrt{h_t}`; returned values forecast :math:`k=1,\ldots,H`,
    converging toward :math:`\sqrt{\bar h}`. A volatility spike after a 28%
    shock gradually forecasts back toward its long-run level, such as 12%.
    Complexity: :math:`O(H)` time and space.
    """
    omega, alpha, beta, _ = _validate_params(garch_params)
    try:
        last_sigma = float(last_volatility)
    except (TypeError, ValueError) as exc:
        raise ValueError("last_volatility must be a finite positive number") from exc
    if not math.isfinite(last_sigma) or last_sigma <= 0.0:
        raise ValueError("last_volatility must be a finite positive number")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    persistence = alpha + beta
    long_run_variance = omega / (1.0 - persistence)
    last_variance = last_sigma**2
    variances = np.array([
        long_run_variance + persistence**step * (last_variance - long_run_variance)
        for step in range(1, horizon + 1)
    ])
    return np.sqrt(np.maximum(variances, 0.0))


def garch_telemetry(returns: pd.Series | np.ndarray) -> dict[str, Any]:
    r"""Expose GARCH fit latency, persistence check, shape, and convergence.

    This helper delegates to :func:`fit_garch11`, exposing the persistence
    condition :math:`\alpha+\beta<1` and optimizer success flag. Complexity
    matches MLE, approximately :math:`O(K n I)`.
    """
    return fit_garch11(returns)
