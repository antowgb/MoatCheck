"""Pure Kelly-criterion position sizing and log-growth projections."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import pinvh


def _finite_scalar(value: float, name: str) -> float:
    """Convert a scalar input to finite float; O(1) time and space."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    """Validate a non-empty finite one-dimensional vector; O(N) time/space."""
    try:
        raw = values.to_numpy(dtype=float, copy=True) if isinstance(values, pd.Series) else values
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _covariance(values: np.ndarray, n_assets: int) -> np.ndarray:
    """Validate a symmetric positive-semidefinite covariance matrix; O(N³)."""
    try:
        raw = values.to_numpy(dtype=float, copy=True) if isinstance(values, pd.DataFrame) else values
        matrix = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("cov_matrix must contain numeric values") from exc
    if matrix.shape != (n_assets, n_assets):
        raise ValueError("cov_matrix shape must match the number of assets")
    if not np.isfinite(matrix).all():
        raise ValueError("cov_matrix must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("cov_matrix must be symmetric")
    if float(np.linalg.eigvalsh(matrix).min()) < -1e-10:
        raise ValueError("cov_matrix must be positive semidefinite")
    return matrix


def calculate_univariate_kelly(
    win_rate: float, win_loss_ratio: float, fraction: float = 0.5
) -> float:
    r"""Compute the scaled Kelly stake for a binary win/loss opportunity.

    For win probability :math:`p`, loss probability :math:`q=1-p`, and net
    win-to-loss ratio :math:`b`, the full Kelly fraction is
    :math:`f^*=(pb-q)/b`; fractional Kelly stakes :math:`f=\gamma f^*`.
    With :math:`p=0.60` and :math:`b=1`, full Kelly stakes 20% and half Kelly
    stakes 10%, reducing growth volatility and estimation sensitivity. A
    negative result is retained as the formula's signed optimal stake; callers
    who cannot short should clamp it to zero. Complexity: :math:`O(1)` time
    and space.
    """
    probability = _finite_scalar(win_rate, "win_rate")
    ratio = _finite_scalar(win_loss_ratio, "win_loss_ratio")
    scale = _finite_scalar(fraction, "fraction")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("win_rate must be between 0 and 1")
    if ratio <= 0.0:
        raise ValueError("win_loss_ratio must be strictly positive")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    loss_probability = 1.0 - probability
    return float(scale * (probability * ratio - loss_probability) / ratio)


def calculate_multivariate_kelly(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
    fraction: float = 0.5,
    max_leverage: float = 2.0,
) -> np.ndarray:
    r"""Calculate fractional continuous Kelly allocations with a gross cap.

    For excess-return vector :math:`\mu-r_f\mathbf{1}` and covariance
    :math:`\Sigma`, the continuous Kelly solution is
    :math:`f^*=\Sigma^{-1}(\mu-r_f\mathbf{1})`. Fractional Kelly multiplies
    it by :math:`\gamma`; if gross exposure exceeds the configured cap, all
    positions are scaled proportionally to satisfy
    :math:`\sum_i|f_i|\le L_{max}`. A two-asset example with full-Kelly weights
    ``[1.2, 0.8]`` has 2.0x gross exposure; half Kelly gives ``[0.6, 0.4]``
    and 1.0x gross exposure. For singular covariance matrices, a Hermitian
    pseudoinverse produces the minimum-norm solution on identifiable risk
    directions. Complexity: :math:`O(N^3)` time and :math:`O(N^2)` space.
    """
    mu = _vector(expected_returns, "expected_returns")
    covariance = _covariance(cov_matrix, mu.size)
    rf = _finite_scalar(risk_free_rate, "risk_free_rate")
    scale = _finite_scalar(fraction, "fraction")
    leverage_limit = _finite_scalar(max_leverage, "max_leverage")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    if leverage_limit <= 0.0:
        raise ValueError("max_leverage must be strictly positive")

    excess_returns = mu - rf
    weights = scale * (pinvh(covariance) @ excess_returns)
    gross_leverage = float(np.abs(weights).sum())
    if gross_leverage > leverage_limit:
        weights *= leverage_limit / gross_leverage
    return np.asarray(weights, dtype=float)


def calculate_growth_rate_projection(
    kelly_weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> dict[str, float | list[int]]:
    r"""Project approximate expected log growth and log-wealth variance.

    For portfolio weights :math:`f`, arithmetic mean :math:`\mu`, and return
    covariance :math:`\Sigma`, the second-order log-growth approximation is
    :math:`g(f)\approx f^T\mu-\frac12f^T\Sigma f`; one-period log-wealth
    variance is approximated by :math:`f^T\Sigma f`. Full Kelly maximizes this
    quadratic utility, while half Kelly sacrifices some expected growth for
    lower log-wealth variance. For the two-asset example ``[1.2, 0.8]`` versus
    ``[0.6, 0.4]``, the half-Kelly variance is one quarter of the full-Kelly
    variance under a common covariance model. Complexity: :math:`O(N^2)` time
    and :math:`O(N^2)` space for the covariance input.
    """
    weights = _vector(kelly_weights, "kelly_weights")
    mu = _vector(expected_returns, "expected_returns")
    if weights.size != mu.size:
        raise ValueError("kelly_weights and expected_returns must have equal length")
    covariance = _covariance(cov_matrix, mu.size)
    mean_portfolio_return = float(weights @ mu)
    variance = float(weights @ covariance @ weights)
    variance = max(variance, 0.0)  # avoid tiny negative round-off from PSD matrices
    return {
        "expected_log_growth_rate": mean_portfolio_return - 0.5 * variance,
        "projected_log_wealth_variance": variance,
        "projected_log_wealth_volatility": math.sqrt(variance),
        "expected_arithmetic_return": mean_portfolio_return,
        "input_shape": [int(weights.size)],
    }


def position_sizing_telemetry(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
    fraction: float = 0.5,
    max_leverage: float = 2.0,
) -> dict[str, Any]:
    r"""Summarize Kelly weights, exposure, growth, shapes, and compute latency.

    The telemetry includes milliseconds, input matrix/vector dimensions,
    realized gross leverage :math:`\sum_i|f_i|`, and fractional scale factor.
    For example, half Kelly ideally halves unconstrained full-Kelly gross
    exposure before any maximum-leverage cap. Complexity: :math:`O(N^3)` time
    and :math:`O(N^2)` space due to the covariance pseudoinverse.
    """
    started = perf_counter()
    mu = _vector(expected_returns, "expected_returns")
    covariance = _covariance(cov_matrix, mu.size)
    weights = calculate_multivariate_kelly(
        mu,
        covariance,
        risk_free_rate=risk_free_rate,
        fraction=fraction,
        max_leverage=max_leverage,
    )
    projection = calculate_growth_rate_projection(weights, mu, covariance)
    return {
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "input_shape": [int(mu.size)],
        "covariance_shape": [int(covariance.shape[0]), int(covariance.shape[1])],
        "weights": weights.tolist(),
        "gross_leverage": float(np.abs(weights).sum()),
        "fraction": float(fraction),
        "max_leverage": float(max_leverage),
        "expected_log_growth_rate": float(projection["expected_log_growth_rate"]),
        "projected_log_wealth_variance": float(
            projection["projected_log_wealth_variance"]
        ),
    }
