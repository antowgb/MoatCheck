"""Gaussian Euler VaR and historical tail-loss attribution utilities.

Covariance-based VaR and historical CVaR attribution are reported as positive
losses. Input returns are simple periodic returns in decimal units.
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


def _validate_weights_covariance(
    weights: np.ndarray, cov_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate finite weights and a symmetric covariance matrix; O(A²)."""
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    if cov.shape != (w.size, w.size):
        raise ValueError("cov_matrix shape must match the number of weights")
    if not np.isfinite(w).all() or not np.isfinite(cov).all():
        raise ValueError("weights and cov_matrix must contain only finite values")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("cov_matrix must be symmetric")
    return w, cov


def _confidence(confidence_level: float) -> float:
    """Return a validated confidence level strictly between 0 and 1; O(1)."""
    try:
        level = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_level must be finite and between 0 and 1") from exc
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    return level


def calculate_marginal_var_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    confidence_level: float = 0.95,
) -> np.ndarray:
    r"""Compute each asset's gradient of parametric Gaussian positive-loss VaR.

    For portfolio mean :math:`\mu_p=w^T\mu` and volatility
    :math:`\sigma_p=\sqrt{w^T\Sigma w}`, positive-loss VaR is
    :math:`VaR_c=-w^T\mu+z_c\sigma_p`, with
    :math:`z_c=\Phi^{-1}(c)`. Thus
    :math:`MVaR_i=\partial VaR/\partial w_i=-\mu_i+z_c(\Sigma w)_i/\sigma_p`.
    The mean vector is omitted by this interface (equivalent to zero expected
    returns), so only the volatility gradient is returned. For three assets,
    this gradient indicates how much a small increase in Tech, Healthcare, or
    Staples exposure changes portfolio VaR. Complexity: :math:`O(A^2)` time,
    :math:`O(A^2)` input space.
    """
    w, cov = _validate_weights_covariance(weights, cov_matrix)
    level = _confidence(confidence_level)
    variance = float(w @ cov @ w)
    if variance <= 0.0:
        raise ValueError("portfolio variance must be strictly positive")
    sigma = math.sqrt(variance)
    return norm.ppf(level) * (cov @ w) / sigma


def calculate_component_var(
    weights: np.ndarray, cov_matrix: np.ndarray, confidence_level: float = 0.95
) -> pd.Series:
    r"""Allocate Gaussian VaR to assets using Euler's homogeneous-function rule.

    Component VaR is :math:`CVaR_i=w_i MVaR_i`. For zero expected returns,
    :math:`VaR=z_c\sqrt{w^T\Sigma w}` is homogeneous of degree one, and
    Euler's theorem gives :math:`\sum_i CVaR_i=VaR`. With three assets, the
    allocations reveal which of Tech, Healthcare, or Staples drives the
    volatility-based loss estimate; tail-event attribution can differ because
    it conditions on realized crash episodes. Complexity: :math:`O(A^2)` time
    and space.
    """
    w, cov = _validate_weights_covariance(weights, cov_matrix)
    marginal = calculate_marginal_var_contribution(w, cov, confidence_level)
    components = w * marginal
    return pd.Series(components, index=[f"asset_{i}" for i in range(w.size)], name="component_var")


def calculate_tail_cvar_attribution(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    confidence_level: float = 0.95,
) -> pd.Series:
    r"""Attribute empirical portfolio expected loss to assets in tail events.

    Compute :math:`R_{p,t}=\sum_iw_iR_{i,t}`, select observations where
    :math:`R_{p,t}\le Q_{1-c}(R_p)`, and report
    :math:`TCVaR_i=-w_i E[R_i\mid R_p\text{ in tail}]`. Consequently
    :math:`\sum_iTCVaR_i=-E[R_p\mid R_p\text{ in tail}]`, the empirical
    positive-loss expected shortfall. For example, a Tech and Energy pair may
    have modest Gaussian volatility contributions but Energy can dominate
    tail attribution if it falls disproportionately during joint crash days.
    Complexity: :math:`O(TA+T\log T)` time and :math:`O(TA)` space.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame")
    values = returns_df.to_numpy(dtype=float, copy=True)
    w = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("returns_df must have at least one row and one asset")
    if w.shape != (values.shape[1],):
        raise ValueError("weights length must match returns_df columns")
    if not np.isfinite(values).all() or not np.isfinite(w).all():
        raise ValueError("returns and weights must be finite")
    level = _confidence(confidence_level)
    portfolio_returns = values @ w
    threshold = float(np.quantile(portfolio_returns, 1.0 - level, method="linear"))
    tail_mask = portfolio_returns <= threshold
    attribution = -w * np.mean(values[tail_mask], axis=0)
    return pd.Series(attribution, index=returns_df.columns, name="tail_cvar_attribution")


def tail_attribution_telemetry(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    r"""Report latency, input shape, and Euler contribution reconciliation.

    ``component_var_sum`` is checked against parametric portfolio VaR
    :math:`z_c\sqrt{w^T\Sigma w}` (under the zero-mean VaR assumption).
    Complexity: :math:`O(TA+A^2)` time and :math:`O(TA)` space.
    """
    started = perf_counter()
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame")
    w, cov = _validate_weights_covariance(weights, cov_matrix)
    if returns_df.shape[1] != w.size:
        raise ValueError("weights and covariance dimensions must match returns_df")
    components = calculate_component_var(w, cov, confidence_level)
    total = float(norm.ppf(_confidence(confidence_level)) * math.sqrt(float(w @ cov @ w)))
    summed = float(components.sum())
    return {
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "input_shape": [int(returns_df.shape[0]), int(returns_df.shape[1])],
        "component_var_sum": summed,
        "total_var": total,
        "contribution_sum_matches_total": bool(np.isclose(summed, total)),
        "percentage_contribution_sum": float(100.0 * summed / total),
    }
