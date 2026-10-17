"""Pure NumPy/Pandas/SciPy Peak-Over-Threshold extreme-value risk metrics."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import genpareto


def _loss_vector(losses: pd.Series | np.ndarray) -> np.ndarray:
    """Validate a finite one-dimensional loss sample; O(n) time and space."""
    if not isinstance(losses, (pd.Series, np.ndarray)):
        raise TypeError("losses must be a pandas Series or NumPy array")
    try:
        values = (losses.to_numpy(dtype=float, na_value=np.nan, copy=True)
                  if isinstance(losses, pd.Series) else np.asarray(losses, dtype=float))
    except (TypeError, ValueError) as exc:
        raise TypeError("losses must contain numeric values") from exc
    if values.ndim != 1 or values.size < 2:
        raise ValueError("losses must be one-dimensional with at least two observations")
    if not np.isfinite(values).all():
        raise ValueError("losses must contain only finite values")
    return values


def _validate_quantiles(confidence_level: float, threshold_quantile: float) -> tuple[float, float]:
    """Validate quantile probabilities and require the VaR level above threshold; O(1)."""
    try:
        confidence, threshold_q = float(confidence_level), float(threshold_quantile)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence and threshold quantiles must be finite probabilities") from exc
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if not math.isfinite(threshold_q) or not 0.0 < threshold_q < 1.0:
        raise ValueError("threshold_quantile must be strictly between 0 and 1")
    if confidence <= threshold_q:
        raise ValueError("confidence_level must exceed threshold_quantile for POT extrapolation")
    return confidence, threshold_q


def _fit_exceedances(excesses: np.ndarray) -> tuple[float, float, Any]:
    """Fit GPD shape and scale by constrained MLE; each likelihood pass is O(Nu)."""
    count = excesses.size
    if count < 10:
        raise ValueError(f"at least 10 threshold exceedances are required; received {count}")
    if not np.isfinite(excesses).all() or np.any(excesses <= 0.0):
        raise ValueError("threshold excesses must be strictly positive and finite")

    # scipy's specialized GPD fitter supplies a strong MLE starting value;
    # generic constrained minimization below makes the likelihood/objective
    # explicit and records convergence telemetry.
    try:
        initial_xi, _, initial_beta = genpareto.fit(excesses, floc=0.0)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"GPD initial fit failed: {exc}") from exc
    initial = np.array([
        float(np.clip(initial_xi, -0.45, 2.0)),
        max(float(initial_beta), np.finfo(float).tiny),
    ])
    upper_beta = max(float(np.max(excesses)) * 100.0, float(np.mean(excesses)) * 100.0)

    def negative_log_likelihood(params: np.ndarray) -> float:
        xi, beta = map(float, params)
        if beta <= 0.0:
            return 1e100
        support = 1.0 + xi * excesses / beta
        if np.any(support <= 0.0):
            return 1e100
        if abs(xi) < 1e-7:
            value = count * math.log(beta) + float(np.sum(excesses)) / beta
        else:
            value = count * math.log(beta) + (1.0 + 1.0 / xi) * float(np.sum(np.log1p(xi * excesses / beta)))
        return float(value) if math.isfinite(value) else 1e100

    constraints = [{
        "type": "ineq",
        "fun": lambda params: 1.0 + params[0] * excesses / params[1] - 1e-10,
    }]
    optimized = minimize(
        negative_log_likelihood,
        initial,
        method="SLSQP",
        bounds=[(-0.45, 2.0), (max(np.finfo(float).tiny, float(np.mean(excesses)) * 1e-10), upper_beta)],
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    candidate = np.asarray(optimized.x, dtype=float)
    if (
        candidate.shape != (2,) or not np.isfinite(candidate).all()
        or candidate[1] <= 0
        or np.any(1.0 + candidate[0] * excesses / candidate[1] <= 0)
    ):
        # scipy fit itself is an MLE and is retained only as a numerical
        # fallback if the explicit refinement fails its support constraints.
        candidate = initial
    return float(candidate[0]), float(candidate[1]), optimized


def fit_gpd_pot(
    losses: pd.Series | np.ndarray, threshold_quantile: float = 0.95
) -> dict[str, Any]:
    r"""Fit a Generalized Pareto distribution to losses exceeding a threshold.

    The Balkema-De Haan-Pickands result says that, for a broad class of parent
    distributions, sufficiently high-threshold excesses are approximately
    GPD: :math:`P(L-u\le y\mid L>u)\approx G_{\xi,\beta}(y)`. Its CDF is
    :math:`G(y)=1-(1+\xi y/\beta)^{-1/\xi}` (with exponential limit
    :math:`1-e^{-y/\beta}` when :math:`\xi=0`) on
    :math:`y\ge0,1+\xi y/\beta>0`. MLE minimizes the negative log-likelihood
    of threshold excesses. A Black Swan tail can make a Gaussian 99.9% VaR
    materially smaller than EVT VaR because positive :math:`\xi` models a
    power-law tail. Complexity: :math:`O(N_u I)` time per optimizer run plus
    threshold selection :math:`O(n\log n)`; memory :math:`O(n)`.
    """
    started = perf_counter()
    values = _loss_vector(losses)
    try:
        quantile = float(threshold_quantile)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold_quantile must be a probability between 0 and 1") from exc
    if not math.isfinite(quantile) or not 0.0 < quantile < 1.0:
        raise ValueError("threshold_quantile must be strictly between 0 and 1")
    threshold = float(np.quantile(values, quantile, method="linear"))
    excesses = values[values > threshold] - threshold
    xi, beta, result = _fit_exceedances(excesses)
    return {
        "threshold": threshold,
        "threshold_quantile": quantile,
        "xi": xi,
        "beta": beta,
        "exceedance_count": int(excesses.size),
        "n_observations": int(values.size),
        "tail_fraction": float(excesses.size / values.size),
        "log_likelihood": float(-_gpd_nll(excesses, xi, beta)),
        "optimization_success": bool(result.success),
        "optimization_status": int(result.status),
        "optimization_message": str(result.message),
        "n_iterations": int(getattr(result, "nit", 0)),
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "input_shape": [int(values.size)],
    }


def _gpd_nll(excesses: np.ndarray, xi: float, beta: float) -> float:
    """Evaluate GPD negative log-likelihood for supported parameters in O(Nu)."""
    support = 1.0 + xi * excesses / beta
    if beta <= 0 or np.any(support <= 0):
        return math.inf
    if abs(xi) < 1e-7:
        return float(excesses.size * math.log(beta) + np.sum(excesses) / beta)
    return float(excesses.size * math.log(beta) + (1.0 + 1.0 / xi) * np.sum(np.log1p(xi * excesses / beta)))


def _evt_fit_and_quantile(
    losses: pd.Series | np.ndarray, confidence_level: float, threshold_quantile: float
) -> tuple[np.ndarray, dict[str, Any], float]:
    """Fit POT model and calculate extrapolated quantile multiplier; O(n log n + Nu I)."""
    confidence, threshold_q = _validate_quantiles(confidence_level, threshold_quantile)
    values = _loss_vector(losses)
    fit = fit_gpd_pot(values, threshold_q)
    tail_probability = (values.size / fit["exceedance_count"]) * (1.0 - confidence)
    if not 0.0 < tail_probability < 1.0:
        raise ValueError("requested confidence level is outside the fitted POT tail")
    return values, fit, float(tail_probability)


def calculate_evt_var(
    losses: pd.Series | np.ndarray,
    confidence_level: float = 0.999,
    threshold_quantile: float = 0.95,
) -> float:
    r"""Calculate high-confidence loss VaR by extrapolating the fitted GPD tail.

    For threshold :math:`u` and :math:`N_u` exceedances among :math:`n`
    observations, POT VaR is
    :math:`VaR_\alpha=u+\frac{\beta}{\xi}[((n/N_u)(1-\alpha))^{-\xi}-1]`.
    When :math:`\xi\to0`, the continuous limit is
    :math:`u+\beta\log((N_u/n)/(1-\alpha))`. At 99.9%, a positive-shape
    Black Swan model can place risk above Gaussian VaR by extrapolating the
    observed heavy tail rather than assuming normality. Complexity:
    :math:`O(n\log n+N_u I)` time and :math:`O(n)` space.
    """
    values, fit, tail_probability = _evt_fit_and_quantile(losses, confidence_level, threshold_quantile)
    xi, beta = fit["xi"], fit["beta"]
    if abs(xi) < 1e-7:
        return float(fit["threshold"] - beta * math.log(tail_probability))
    return float(fit["threshold"] + (beta / xi) * (tail_probability ** (-xi) - 1.0))


def calculate_evt_cvar(
    losses: pd.Series | np.ndarray,
    confidence_level: float = 0.999,
    threshold_quantile: float = 0.95,
) -> float:
    r"""Calculate GPD Expected Shortfall above EVT VaR for a finite-mean tail.

    For :math:`\xi<1`, the conditional tail expectation is
    :math:`CVaR_\alpha=VaR_\alpha+\frac{\beta+\xi(VaR_\alpha-u)}{1-\xi}`.
    This is the mean loss conditional on exceeding VaR, and is infinite for
    :math:`\xi\ge1`, where the fitted tail has no finite first moment. A
    Black Swan EVT CVaR is typically larger than its VaR and can substantially
    exceed a Gaussian tail estimate. Complexity: :math:`O(n\log n+N_u I)`
    time and :math:`O(n)` space.
    """
    _, fit, tail_probability = _evt_fit_and_quantile(losses, confidence_level, threshold_quantile)
    xi, beta, threshold = fit["xi"], fit["beta"], fit["threshold"]
    if xi >= 1.0:
        return math.inf
    if abs(xi) < 1e-7:
        var = threshold - beta * math.log(tail_probability)
        return float(var + beta)
    var = threshold + (beta / xi) * (tail_probability ** (-xi) - 1.0)
    return float(var + (beta + xi * (var - threshold)) / (1.0 - xi))


def evt_telemetry(
    losses: pd.Series | np.ndarray,
    confidence_level: float = 0.999,
    threshold_quantile: float = 0.95,
) -> dict[str, Any]:
    r"""Return EVT fit latency, shape :math:`\xi`, exceedances, and VaR/CVaR.

    ``input_shape`` and ``exceedance_count`` expose how much tail information
    supports the Black Swan estimate, alongside optimizer status. Complexity:
    :math:`O(n\log n+N_u I)` time and :math:`O(n)` space.
    """
    started = perf_counter()
    fit = fit_gpd_pot(losses, threshold_quantile)
    var = calculate_evt_var(losses, confidence_level, threshold_quantile)
    cvar = calculate_evt_cvar(losses, confidence_level, threshold_quantile)
    return {
        **fit,
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "confidence_level": float(confidence_level),
        "evt_var": var,
        "evt_cvar": cvar,
    }
