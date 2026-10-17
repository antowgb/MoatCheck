"""Pure long-only risk-parity calculations and portfolio optimization."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _validated_weights_and_covariance(
    weights: np.ndarray, cov_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Validate a weight vector and covariance matrix for risk calculations.

    Portfolio variance is :math:`\sigma_p^2=w^T\Sigma w`, so weights and a
    square, finite, symmetric positive-semidefinite covariance matrix must
    have matching dimensions. For example, the covariance of a 20%-volatility
    Tech asset and a 10%-volatility Staples asset has diagonal values
    :math:`0.04` and :math:`0.01`. Complexity: :math:`O(n^3)` time for the PSD
    check and :math:`O(n^2)` space for :math:`n` assets.
    """
    try:
        weight_values = np.asarray(weights, dtype=float)
        covariance = np.asarray(cov_matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("weights and cov_matrix must contain numeric values") from exc
    if weight_values.ndim != 1:
        raise ValueError("weights must be a one-dimensional vector")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("cov_matrix must be a square matrix")
    if covariance.shape[0] != weight_values.size or weight_values.size == 0:
        raise ValueError("weights length must match a non-empty covariance matrix")
    if not np.isfinite(weight_values).all() or not np.isfinite(covariance).all():
        raise ValueError("weights and cov_matrix must contain only finite values")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("cov_matrix must be symmetric")
    covariance = (covariance + covariance.T) / 2.0
    if float(np.linalg.eigvalsh(covariance).min()) < -1e-10:
        raise ValueError("cov_matrix must be positive semidefinite")
    return weight_values, covariance


def calculate_marginal_risk_contribution(
    weights: np.ndarray, cov_matrix: np.ndarray
) -> np.ndarray:
    r"""Calculate each asset's marginal contribution to portfolio volatility.

    The derivative of portfolio volatility with respect to weight :math:`w_i`
    is :math:`MCR_i=\frac{(\Sigma w)_i}{\sigma_p}`, where
    :math:`\sigma_p=\sqrt{w^T\Sigma w}`. This measures the change in total
    portfolio risk from a small change in one holding. For example, an
    uncorrelated 20%-volatility Tech holding has twice the marginal risk of a
    10%-volatility Staples holding at the same portfolio scale. Complexity:
    :math:`O(n^3)` time including covariance validation and :math:`O(n^2)`
    input space.
    """
    weight_values, covariance = _validated_weights_and_covariance(weights, cov_matrix)
    variance = float(weight_values @ covariance @ weight_values)
    if variance <= 0.0:
        raise ValueError("portfolio volatility must be positive to calculate marginal risk")
    return (covariance @ weight_values) / math.sqrt(variance)


def calculate_risk_contributions(
    weights: np.ndarray, cov_matrix: np.ndarray
) -> np.ndarray:
    r"""Calculate normalized absolute risk contributions for all assets.

    The contribution fraction is
    :math:`RC_i=\frac{w_i MCR_i}{\sigma_p}=
    \frac{w_i(\Sigma w)_i}{w^T\Sigma w}`. Contributions sum to one when
    portfolio variance is positive. For two uncorrelated assets with
    volatilities 20% (Tech) and 10% (Staples), inverse-volatility weights
    :math:`(1/3,2/3)` give each asset one-half of total risk. Complexity:
    :math:`O(n^3)` time including covariance validation and :math:`O(n^2)`
    input space.
    """
    weight_values, covariance = _validated_weights_and_covariance(weights, cov_matrix)
    variance = float(weight_values @ covariance @ weight_values)
    if variance <= 0.0:
        raise ValueError("portfolio volatility must be positive to calculate risk contributions")
    contributions = weight_values * (covariance @ weight_values) / variance
    return contributions


def _validate_risk_budget(
    target_risk_budget: np.ndarray | None, n_assets: int
) -> np.ndarray:
    r"""Normalize a requested positive risk budget or create equal budgets.

    Budgets satisfy :math:`b_i>0` and :math:`\sum_i b_i=1`; normalization
    allows inputs such as ``[1, 2]`` to mean one-third versus two-thirds of
    target risk. In the Tech/Staples example, equal budgets call for inverse
    volatility weights of one-third and two-thirds. Complexity: :math:`O(n)`
    time and space.
    """
    if target_risk_budget is None:
        return np.full(n_assets, 1.0 / n_assets, dtype=float)
    try:
        budget = np.asarray(target_risk_budget, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("target_risk_budget must be a numeric vector") from exc
    if budget.ndim != 1 or budget.size != n_assets:
        raise ValueError("target_risk_budget must have one value per asset")
    if not np.isfinite(budget).all() or np.any(budget <= 0.0):
        raise ValueError("target_risk_budget values must be finite and positive")
    total = float(budget.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("target_risk_budget must have a positive finite sum")
    return budget / total


def risk_contribution_dispersion(
    risk_contributions: np.ndarray, target_risk_budget: np.ndarray | None = None
) -> float:
    r"""Measure RMS deviation between realized and target risk shares.

    Dispersion is :math:`\sqrt{\frac{1}{n}\sum_i(RC_i-b_i)^2}`; zero means
    every asset contributes exactly its budget. For equal-risk Tech and
    Staples, contributions ``[0.5, 0.5]`` have zero dispersion, while
    ``[0.6, 0.4]`` have dispersion 0.1. Complexity: :math:`O(n)` time and
    :math:`O(n)` temporary space.
    """
    try:
        contributions = np.asarray(risk_contributions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("risk_contributions must be a numeric vector") from exc
    if contributions.ndim != 1 or contributions.size == 0:
        raise ValueError("risk_contributions must be a non-empty vector")
    if not np.isfinite(contributions).all():
        raise ValueError("risk_contributions must contain only finite values")
    budget = _validate_risk_budget(target_risk_budget, contributions.size)
    return float(np.sqrt(np.mean((contributions - budget) ** 2)))


def _covariance_frame(cov_matrix: pd.DataFrame) -> tuple[np.ndarray, list[Any]]:
    r"""Validate labeled covariance input and preserve its asset ordering.

    The optimizer uses :math:`\Sigma` in the same order for weights and
    reported risk shares; matching unique row and column labels prevents
    silent risk assignment to the wrong assets. For example, Tech and Staples
    variances of ``0.04`` and ``0.01`` correspond to volatilities of 20% and
    10%. Complexity: :math:`O(n^3)` time and :math:`O(n^2)` space.
    """
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError("cov_matrix must be a pandas DataFrame")
    if cov_matrix.empty or not cov_matrix.index.is_unique or not cov_matrix.columns.is_unique:
        raise ValueError("cov_matrix must have non-empty unique asset labels")
    if set(cov_matrix.index) != set(cov_matrix.columns):
        raise ValueError("cov_matrix row and column labels must match")
    ordered = cov_matrix.loc[cov_matrix.index, cov_matrix.index]
    try:
        values = ordered.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("cov_matrix must contain numeric values") from exc
    # Revalidate using a neutral vector so symmetry and PSD checks are shared.
    _, values = _validated_weights_and_covariance(
        np.ones(len(cov_matrix.index), dtype=float), values
    )
    return values, list(cov_matrix.index)


def optimize_risk_parity(
    cov_matrix: pd.DataFrame,
    target_risk_budget: np.ndarray | None = None,
) -> dict[str, Any]:
    r"""Find long-only weights whose risk shares match a target risk budget.

    It minimizes :math:`\sum_i(RC_i-b_i)^2` subject to
    :math:`\sum_iw_i=1` and :math:`0\le w_i\le1`, where
    :math:`RC_i=\frac{w_i(\Sigma w)_i}{w^T\Sigma w}`. Equal budgets create
    risk parity; custom budgets support intentional risk tilts. With
    uncorrelated Tech at 20% volatility and Staples at 10%, equal risk is
    attained by weights :math:`(1/3,2/3)`: the lower-volatility Staples
    position must be twice as large to contribute the same risk. Results
    include risk shares, portfolio volatility, RMS dispersion, convergence,
    iterations, and elapsed milliseconds. Complexity is :math:`O(I n^2+n^3)`
    for :math:`I` SLSQP iterations and :math:`n` assets.
    """
    covariance, assets = _covariance_frame(cov_matrix)
    n_assets = len(assets)
    budget = _validate_risk_budget(target_risk_budget, n_assets)
    asset_volatility = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if np.any(asset_volatility <= 0.0):
        raise ValueError("every asset must have positive standalone volatility")

    initial = budget / asset_volatility
    initial /= initial.sum()

    def objective(weights: np.ndarray) -> float:
        r"""Return squared risk-budget error in :math:`O(n^2)` time."""
        variance = float(weights @ covariance @ weights)
        if not math.isfinite(variance) or variance <= 1e-24:
            return 1e12
        contributions = weights * (covariance @ weights) / variance
        return float(np.sum((contributions - budget) ** 2))

    start = perf_counter()
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}],
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    latency_ms = (perf_counter() - start) * 1000.0
    weights = np.asarray(result.x, dtype=float)
    weight_sum = float(weights.sum())
    bounds_error = max(
        float(np.maximum(-weights, 0.0).max()),
        float(np.maximum(weights - 1.0, 0.0).max()),
    )
    if (
        not result.success
        or not np.isfinite(weights).all()
        or abs(weight_sum - 1.0) > 1e-7
        or bounds_error > 1e-7
    ):
        raise RuntimeError(
            f"risk-parity optimization failed: {result.message}; "
            f"weight_sum_error={abs(weight_sum - 1.0):.3g}, bounds_error={bounds_error:.3g}"
        )

    # Absorb floating-point sum error without changing the optimized solution
    # by more than roundoff. Choose the largest weight for stable correction.
    correction_index = int(np.argmax(weights))
    weights[correction_index] += 1.0 - float(weights.sum())
    risk_contributions = calculate_risk_contributions(weights, covariance)
    dispersion = risk_contribution_dispersion(risk_contributions, budget)
    volatility = math.sqrt(float(weights @ covariance @ weights))
    return {
        "weights": {asset: float(weight) for asset, weight in zip(assets, weights, strict=True)},
        "risk_contributions": {
            asset: float(contribution)
            for asset, contribution in zip(assets, risk_contributions, strict=True)
        },
        "target_risk_budget": {
            asset: float(value) for asset, value in zip(assets, budget, strict=True)
        },
        "portfolio_volatility": float(volatility),
        "risk_contribution_dispersion": float(dispersion),
        "success": bool(result.success),
        "converged": bool(result.success and dispersion < 1e-7),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "latency_ms": float(latency_ms),
        "input_shape": {"cov_matrix": [int(n_assets), int(n_assets)]},
    }
