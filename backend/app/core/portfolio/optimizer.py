"""Database-independent long-only portfolio optimization.

All returns are simple periodic returns in decimal units (e.g. 0.01 = 1%).
Rows are observations and columns are assets. Expected return and covariance
are annualized using ``periods_per_year``. Constraints and results use asset
order matching the input matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

Objective = Literal["min_variance", "max_sharpe", "max_return"]
_SLSQP_MAX_ITERATIONS = 1000
_SLSQP_TOLERANCE = 1e-10


@dataclass(frozen=True)
class OptimizationResult:
    r"""Immutable allocation summary with explicit arithmetic return units.

    Reports :math:`E[R_p]=w^T\mu` and
    :math:`\sigma_p=\sqrt{w^T\Sigma w}`. For example, a 50/50 allocation
    to assets expected to return 8% and 12% has a 10% expected return.
    Storing an already computed result takes :math:`O(A)` space for weights
    and labels; constructing the record itself performs no optimization.
    """

    weights: list[float]
    expected_return: float
    volatility: float
    sharpe_ratio: float | None
    objective: Objective
    success: bool
    message: str
    assets: list[str]

    def to_dict(self) -> dict:
        r"""Convert result fields to a plain mapping for API-independent use.

        This is the identity mapping :math:`result\mapsto fields(result)`;
        it keeps serialization separate from optimization and web frameworks.
        Complexity: :math:`O(m)` time and output space for ``m`` assets.
        """
        return asdict(self)


def _validate_covariance(cov_matrix: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    r"""Validate labels, finiteness, symmetry and positive semidefiniteness.

    A covariance matrix obeys :math:`\Sigma=\Sigma^T` and
    :math:`x^T\Sigma x\ge0`. These properties make portfolio variance a
    meaningful risk measure; invalid matrices are rejected before SLSQP sees
    an ill-posed objective.

    Complexity: :math:`O(m^3)` time for eigenvalue validation and
    :math:`O(m^2)` space for ``m`` assets.
    """
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError("cov_matrix must be a pandas DataFrame")
    if cov_matrix.empty or not cov_matrix.index.is_unique or not cov_matrix.columns.is_unique:
        raise ValueError("covariance matrix must have a non-empty set of unique asset labels")
    assets = [str(label) for label in cov_matrix.index]
    columns = [str(label) for label in cov_matrix.columns]
    if len(set(assets)) != len(assets) or len(set(columns)) != len(columns) or set(assets) != set(columns):
        raise ValueError("covariance row and column labels must identify the same unique assets")
    try:
        ordered = cov_matrix.copy()
        ordered.index = assets
        ordered.columns = columns
        values = ordered.loc[assets, assets].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("covariance matrix must contain numeric values and matching labels") from exc
    if not np.isfinite(values).all():
        raise ValueError("covariance matrix must contain only finite values")
    if not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance matrix must be symmetric")
    values = (values + values.T) / 2.0
    if float(np.linalg.eigvalsh(values).min()) < -1e-10:
        raise ValueError("covariance matrix must be positive semidefinite")
    return values, assets


def _validate_expected_returns(expected_returns: pd.Series, assets: list[str]) -> np.ndarray:
    r"""Align a finite expected-return vector to covariance asset order.

    The return vector :math:`\mu\in\mathbb{R}^m` needs one finite value per
    asset. Explicit label alignment prevents assigning an estimate to the
    wrong holding.

    Complexity: :math:`O(m)` time and space.
    """
    if not isinstance(expected_returns, pd.Series):
        raise TypeError("expected_returns must be a pandas Series")
    if not expected_returns.index.is_unique:
        raise ValueError("expected_returns must have unique asset labels")
    labels = [str(label) for label in expected_returns.index]
    if len(set(labels)) != len(labels) or set(labels) != set(assets):
        raise ValueError("expected_returns and cov_matrix must contain the same asset labels")
    aligned = expected_returns.copy()
    aligned.index = labels
    try:
        values = aligned.loc[assets].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected returns must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("expected returns must contain only finite values")
    return values


def _weights_as_dict(weights: np.ndarray, assets: list[str]) -> dict[str, float]:
    r"""Map weights to assets and absorb the floating-point sum residual.

    Fully invested weights satisfy :math:`\sum_iw_i=1`. Assigning the tiny
    rounding residual to the last asset makes ordinary sequential sums equal
    one while changing the optimum only by numerical precision.

    Complexity: :math:`O(m)` time and output space.
    """
    values = np.clip(np.asarray(weights, dtype=float), 0.0, 1.0)
    values /= float(values.sum())
    if values.size > 1:
        values[-1] = 1.0 - sum(float(value) for value in values[:-1])
    else:
        values[0] = 1.0
    return {asset: float(weight) for asset, weight in zip(assets, values)}


def _result_payload(
    weights: np.ndarray,
    expected_returns: np.ndarray | None,
    covariance: np.ndarray,
    assets: list[str],
    *,
    message: str,
    iterations: int,
    latency_ms: float,
    risk_free_rate: float = 0.0,
) -> dict[str, object]:
    r"""Build a transparent result with portfolio moments and solver telemetry.

    Portfolio moments are :math:`\mu_p=w^T\mu` and
    :math:`\sigma_p=\sqrt{w^T\Sigma w}`. Reporting them with weights and
    optimizer status helps callers audit each allocation.

    Complexity: :math:`O(m^2)` time and :math:`O(m)` output space.
    """
    weight_map = _weights_as_dict(weights, assets)
    ordered = np.asarray([weight_map[asset] for asset in assets], dtype=float)
    expected = float(ordered @ expected_returns) if expected_returns is not None else None
    variance = float(max(ordered @ covariance @ ordered, 0.0))
    volatility = float(np.sqrt(variance))
    sharpe = (expected - risk_free_rate) / volatility if expected is not None and volatility > 0.0 else None
    return {
        "weights": weight_map,
        "assets": assets.copy(),
        "expected_return": expected,
        "variance": variance,
        "volatility": volatility,
        "sharpe_ratio": float(sharpe) if sharpe is not None else None,
        "success": True,
        "message": str(message),
        "iterations": int(iterations),
        "latency_ms": float(latency_ms),
    }


def _run_minimum_variance(
    covariance: np.ndarray,
    assets: list[str],
    expected_returns: np.ndarray | None,
    target_return: float | None = None,
) -> dict[str, object]:
    r"""Solve a long-only minimum-variance QP, optionally at a target return.

    It solves :math:`\min_w w^T\Sigma w`, subject to
    :math:`\mathbf{1}^Tw=1`, :math:`0\le w_i\le1`, and optionally
    :math:`w^T\mu=R^*`. The starting portfolio interpolates between the
    minimum- and maximum-return assets to satisfy :math:`R^*` from the outset.

    Complexity: :math:`O(I m^2)` objective work per solve, with ``I`` solver
    iterations and ``m`` assets.
    """
    n_assets = len(assets)
    initial = np.full(n_assets, 1.0 / n_assets)
    constraints: list[dict] = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    if target_return is not None:
        if expected_returns is None:
            raise ValueError("expected returns are required for a target-return constraint")
        min_idx, max_idx = int(np.argmin(expected_returns)), int(np.argmax(expected_returns))
        span = float(expected_returns[max_idx] - expected_returns[min_idx])
        if span > 0.0:
            max_weight = float(np.clip((target_return - expected_returns[min_idx]) / span, 0.0, 1.0))
            initial = np.zeros(n_assets, dtype=float)
            initial[min_idx] = 1.0 - max_weight
            initial[max_idx] += max_weight
        constraints.append({"type": "eq", "fun": lambda w, target=target_return: float(w @ expected_returns - target)})

    start = perf_counter()
    result = minimize(
        lambda w: float(w @ covariance @ w),
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=constraints,
        options={"maxiter": _SLSQP_MAX_ITERATIONS, "ftol": _SLSQP_TOLERANCE, "disp": False},
    )
    latency_ms = (perf_counter() - start) * 1000.0
    weights = np.asarray(result.x, dtype=float)
    error = abs(float(weights.sum()) - 1.0)
    if target_return is not None:
        if expected_returns is None:
            raise RuntimeError("target-return constraint requires expected returns")
        error = max(error, abs(float(weights @ expected_returns - target_return)))
    bounds_error = max(float(np.maximum(-weights, 0.0).max()), float(np.maximum(weights - 1.0, 0.0).max()))
    if not result.success or error > 1e-7 or bounds_error > 1e-7:
        raise RuntimeError(f"minimum-variance optimization failed: {result.message}; constraint_error={error:.3g}")
    output = _result_payload(
        weights,
        expected_returns,
        covariance,
        assets,
        message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        latency_ms=latency_ms,
    )
    if target_return is not None:
        output["target_return"] = float(target_return)
    return output


def optimize_maximum_sharpe(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.03,
) -> dict[str, object]:
    r"""Find long-only asset weights that maximize Sharpe ratio using SLSQP.

    The program is :math:`\max_w\frac{w^T\mu-r_f}{\sqrt{w^T\Sigma w}}`
    subject to :math:`\mathbf{1}^Tw=1` and :math:`0\le w_i\le1`. SciPy
    minimizes the negative of this objective. In a Tech/Healthcare/Staples
    example, a lower covariance between Healthcare and Tech can make a mixed
    portfolio more efficient than choosing the sector with the highest return
    alone. A 12% return at 8% volatility has Sharpe 1.125 at a 3% risk-free
    rate; 15% at 25% volatility has Sharpe 0.48.

    Expected returns and covariance must use the same annual units. Result
    fields include moments, success, elapsed milliseconds and iterations.
    Complexity: :math:`O(I m^2)` objective work for ``I`` iterations and
    ``m`` assets, plus :math:`O(m^3)` covariance validation.
    """
    covariance, assets = _validate_covariance(cov_matrix)
    mu = _validate_expected_returns(expected_returns, assets)
    try:
        rate = float(risk_free_rate)
    except (TypeError, ValueError) as exc:
        raise TypeError("risk_free_rate must be numeric") from exc
    if not np.isfinite(rate):
        raise ValueError("risk_free_rate must be finite")
    if not np.any(np.diag(covariance) > 0.0):
        raise ValueError("maximum Sharpe ratio is undefined for a zero-risk covariance matrix")
    n_assets = len(assets)

    def negative_sharpe(weights: np.ndarray) -> float:
        r"""Evaluate negative Sharpe; its quadratic risk calculation is :math:`O(m^2)`."""
        variance = max(float(weights @ covariance @ weights), 1e-24)
        return -float((weights @ mu - rate) / np.sqrt(variance))

    start = perf_counter()
    result = minimize(
        negative_sharpe,
        np.full(n_assets, 1.0 / n_assets),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}],
        options={"maxiter": _SLSQP_MAX_ITERATIONS, "ftol": _SLSQP_TOLERANCE, "disp": False},
    )
    latency_ms = (perf_counter() - start) * 1000.0
    weights = np.asarray(result.x, dtype=float)
    error = abs(float(weights.sum()) - 1.0)
    bounds_error = max(float(np.maximum(-weights, 0.0).max()), float(np.maximum(weights - 1.0, 0.0).max()))
    if not result.success or error > 1e-7 or bounds_error > 1e-7:
        raise RuntimeError(f"maximum-Sharpe optimization failed: {result.message}; constraint_error={error:.3g}")
    output = _result_payload(
        weights,
        mu,
        covariance,
        assets,
        message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        latency_ms=latency_ms,
        risk_free_rate=rate,
    )
    output["sharpe_ratio"] = float(-negative_sharpe(np.asarray(list(output["weights"].values()))))
    return output


def optimize_minimum_variance(cov_matrix: pd.DataFrame) -> dict[str, object]:
    r"""Find the long-only portfolio minimizing :math:`w^T\Sigma w`.

    Constraints are :math:`\mathbf{1}^Tw=1` and :math:`0\le w_i\le1`.
    In a Tech/Healthcare/Staples universe, a lower covariance between two
    sectors can make their mixture less volatile than either position alone,
    even if the third asset has a higher expected return. The returned mapping
    contains weights, variance, volatility, solver status, latency, and
    iteration count.

    Complexity: :math:`O(I m^2)` objective work for ``I`` iterations and
    ``m`` assets, plus :math:`O(m^3)` covariance validation.
    """
    covariance, assets = _validate_covariance(cov_matrix)
    return _run_minimum_variance(covariance, assets, None)


def generate_efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    points: int = 20,
) -> list[dict[str, object]]:
    r"""Generate target-return minimum-variance portfolios along the frontier.

    For target :math:`R^*_k`, solve :math:`\min_w w^T\Sigma w` subject to
    :math:`\mathbf{1}^Tw=1`, :math:`w^T\mu=R^*_k`, and :math:`0\le w_i\le1`.
    Targets span the minimum to maximum individual expected returns. In a
    Tech/Healthcare/Staples example, sweeping those targets shows the least
    volatility available at each expected return and how sector diversification
    changes the boundary.

    If all expected returns are equal, the frontier has one unique point.
    Each result includes target, weights, moments, success, latency, and
    iterations. Complexity: :math:`O(P I m^2+m^3)` for ``P`` points and
    ``I`` iterations per solve.
    """
    if isinstance(points, bool) or not isinstance(points, (int, np.integer)) or points < 2:
        raise ValueError("points must be an integer of at least 2")
    covariance, assets = _validate_covariance(cov_matrix)
    mu = _validate_expected_returns(expected_returns, assets)
    low, high = float(mu.min()), float(mu.max())
    if np.isclose(low, high, rtol=1e-12, atol=1e-14):
        return [_run_minimum_variance(covariance, assets, mu, target_return=low)]
    targets = np.linspace(low, high, int(points))
    return [_run_minimum_variance(covariance, assets, mu, float(target)) for target in targets]


def _returns_array(returns: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str]]:
    r"""Validate an observations-by-assets matrix of periodic simple returns.

    Each return :math:`R_{t,i}` must be finite and greater than :math:`-1`.
    This establishes the input domain for estimating sample mean and
    covariance in the existing general-purpose optimizer.

    Complexity: :math:`O(nm)` time and temporary space for ``n`` observations
    and ``m`` assets.
    """
    if isinstance(returns, pd.DataFrame):
        assets = [str(c) for c in returns.columns]
        values = returns.to_numpy(dtype=float, copy=True)
    else:
        values = np.asarray(returns, dtype=float)
        assets = [f"asset_{i}" for i in range(values.shape[1])] if values.ndim == 2 else []
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("returns must be a 2D observations-by-assets matrix with at least 2 rows and 2 assets")
    if not np.isfinite(values).all() or np.any(values <= -1.0):
        raise ValueError("returns must contain only finite values greater than -100%; align and clean observations first")
    if len(set(assets)) != len(assets):
        raise ValueError("asset labels must be unique")
    return values, assets


def optimize_portfolio(
    returns: pd.DataFrame | np.ndarray,
    *,
    objective: Objective = "max_sharpe",
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    target_return: float | None = None,
    target_volatility: float | None = None,
    lower_bounds: float | list[float] = 0.0,
    upper_bounds: float | list[float] = 1.0,
    initial_weights: list[float] | None = None,
    factor_exposures: pd.DataFrame | np.ndarray | None = None,
    factor_tilts: list[float] | None = None,
    linear_exposures: pd.DataFrame | np.ndarray | None = None,
    linear_lower_bounds: list[float] | None = None,
    linear_upper_bounds: list[float] | None = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-9,
) -> OptimizationResult:
    r"""Optimize a fully invested portfolio under box and linear constraints.

    ``factor_tilts`` adds annualized factor exposure reward to expected return
    for max-return/max-Sharpe objectives. For min-variance with target return,
    it adjusts the return constraint. ``linear_exposures`` is an N-assets by
    K-constraints matrix (e.g. sector exposures); optional lower/upper bounds
    constrain each resulting exposure. No database or network access occurs.

    Portfolio moments are :math:`\mu_p=w^T\mu` and
    :math:`\sigma_p^2=w^T\Sigma w`. In a Tech/Healthcare/Staples portfolio,
    lower cross-sector covariance can reduce risk even when all three assets
    remain invested; factor and linear constraints express additional views.

    Complexity: annual moment estimation is :math:`O(nm^2)` and each of
    :math:`I` SLSQP iterations evaluates objectives/constraints in up to
    :math:`O(m^2)` time for ``n`` observations and ``m`` assets.
    """
    values, assets = _returns_array(returns)
    n_obs, n_assets = values.shape
    if objective not in ("min_variance", "max_sharpe", "max_return"):
        raise ValueError(f"unsupported objective: {objective}")
    if periods_per_year < 1 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("periods_per_year, max_iterations, and tolerance must be positive")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")

    mu = values.mean(axis=0) * periods_per_year
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1)) * periods_per_year
    covariance = (covariance + covariance.T) / 2.0
    # Tiny diagonal regularization stabilizes near-singular covariance matrices.
    covariance += np.eye(n_assets) * max(float(np.trace(covariance)) / n_assets, 1.0) * 1e-12

    if factor_tilts is not None:
        if factor_exposures is None:
            raise ValueError("factor_exposures is required when factor_tilts are supplied")
        exposure = np.asarray(factor_exposures, dtype=float)
        tilt = np.asarray(factor_tilts, dtype=float)
        if exposure.ndim != 2 or exposure.shape[0] != n_assets or exposure.shape[1] != tilt.size:
            raise ValueError("factor_exposures must have shape (assets, factors), matching factor_tilts")
        if not np.isfinite(exposure).all() or not np.isfinite(tilt).all():
            raise ValueError("factor exposures and tilts must be finite")
        mu = mu + exposure @ tilt

    def _bound(value: float | list[float], label: str) -> np.ndarray:
        r"""Expand and validate an asset-wise bound :math:`l_i\le w_i\le u_i`.

        A scalar is broadcast uniformly across all assets for convenience.
        Complexity: :math:`O(m)` time and space.
        """
        array = np.full(n_assets, float(value)) if np.isscalar(value) else np.asarray(value, dtype=float)
        if array.shape != (n_assets,) or not np.isfinite(array).all():
            raise ValueError(f"{label} must be a finite scalar or one value per asset")
        return array

    lower = _bound(lower_bounds, "lower_bounds")
    upper = _bound(upper_bounds, "upper_bounds")
    if np.any(lower > upper) or lower.sum() > 1 + tolerance or upper.sum() < 1 - tolerance:
        raise ValueError("weight bounds are infeasible for a fully invested portfolio")
    bounds = list(zip(lower.tolist(), upper.tolist()))

    x0 = np.asarray(initial_weights, dtype=float) if initial_weights is not None else np.full(n_assets, 1 / n_assets)
    if x0.shape != (n_assets,) or not np.isfinite(x0).all():
        raise ValueError("initial_weights must provide one finite value per asset")
    x0 = np.clip(x0, lower, upper)
    # Project the starting point onto the fully-invested box simplex by
    # distributing any residual across available capacity.
    for _ in range(n_assets + 1):
        residual = 1.0 - float(x0.sum())
        if abs(residual) <= tolerance:
            break
        capacity = upper - x0 if residual > 0 else x0 - lower
        available = float(capacity.sum())
        if available <= 0:
            raise ValueError("weight bounds are infeasible")
        x0 += np.sign(residual) * np.minimum(capacity, abs(residual) * capacity / available)

    constraints: list[dict] = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    if target_return is not None:
        if not np.isfinite(target_return):
            raise ValueError("target_return must be finite")
        constraints.append({"type": "eq", "fun": lambda w: float(w @ mu - target_return)})
    if target_volatility is not None:
        if target_volatility <= 0 or not np.isfinite(target_volatility):
            raise ValueError("target_volatility must be finite and positive")
        constraints.append({"type": "ineq", "fun": lambda w: float(target_volatility**2 - w @ covariance @ w)})

    if linear_exposures is not None:
        matrix = np.asarray(linear_exposures, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != n_assets or not np.isfinite(matrix).all():
            raise ValueError("linear_exposures must have shape (assets, constraints) and be finite")
        k = matrix.shape[1]
        lo = np.full(k, -np.inf) if linear_lower_bounds is None else np.asarray(linear_lower_bounds, dtype=float)
        hi = np.full(k, np.inf) if linear_upper_bounds is None else np.asarray(linear_upper_bounds, dtype=float)
        if lo.shape != (k,) or hi.shape != (k,) or np.any(lo > hi):
            raise ValueError("linear lower/upper bounds must match exposure columns and be ordered")
        constraints.append({"type": "ineq", "fun": lambda w: matrix.T @ w - lo})
        constraints.append({"type": "ineq", "fun": lambda w: hi - matrix.T @ w})
    elif linear_lower_bounds is not None or linear_upper_bounds is not None:
        raise ValueError("linear_exposures is required when linear bounds are supplied")

    def variance(w: np.ndarray) -> float:
        r"""Evaluate portfolio variance :math:`w^T\Sigma w` in :math:`O(m^2)`."""
        return float(w @ covariance @ w)

    def negative_sharpe(w: np.ndarray) -> float:
        r"""Return negative Sharpe for SLSQP minimization in :math:`O(m^2)`."""
        vol = np.sqrt(max(variance(w), 1e-24))
        return -float((w @ mu - risk_free_rate) / vol)

    if objective == "min_variance":
        fun = variance
    elif objective == "max_return":
        fun = lambda w: -float(w @ mu)
    else:
        fun = negative_sharpe

    result = minimize(
        fun, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": max_iterations, "ftol": tolerance, "disp": False},
    )
    weights = np.asarray(result.x, dtype=float)
    constraint_error = abs(float(weights.sum()) - 1.0)
    max_error = max(constraint_error, float(np.maximum(lower - weights, 0).max()), float(np.maximum(weights - upper, 0).max()))
    if target_return is not None:
        max_error = max(max_error, abs(float(weights @ mu - target_return)))
    if target_volatility is not None:
        max_error = max(max_error, max(0.0, float(np.sqrt(max(variance(weights), 0.0)) - target_volatility)))
    if linear_exposures is not None:
        exposure_values = np.asarray(linear_exposures, dtype=float).T @ weights
        max_error = max(max_error, float(np.maximum(lo - exposure_values, 0).max()),
                        float(np.maximum(exposure_values - hi, 0).max()))
    if not result.success or max_error > max(1e-6, tolerance * 100):
        raise RuntimeError(f"portfolio optimization failed constraint/convergence checks: {result.message}; max_error={max_error:.3g}")
    port_return = float(weights @ mu)
    port_vol = float(np.sqrt(max(variance(weights), 0.0)))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else None
    return OptimizationResult(
        weights=weights.tolist(), expected_return=port_return, volatility=port_vol,
        sharpe_ratio=float(sharpe) if sharpe is not None else None,
        objective=objective, success=True, message=str(result.message), assets=assets,
    )
