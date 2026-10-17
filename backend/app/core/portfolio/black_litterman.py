"""Pure Black-Litterman posterior estimation and long-only allocation."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _validated_market_inputs(
    market_weights: pd.Series, cov_matrix: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    r"""Align and validate market weights and a positive-definite covariance.

    Equilibrium returns require :math:`\Pi=\lambda\Sigma w_{mkt}`, with
    weights ordered exactly as the covariance assets. For example, the MSFT
    and AAPL weights ``[0.6, 0.4]`` must pair with the correspondingly labeled
    covariance rows and columns. Complexity: :math:`O(n^3)` time for the
    positive-definiteness check and :math:`O(n^2)` space for :math:`n` assets.
    """
    if not isinstance(market_weights, pd.Series):
        raise TypeError("market_weights must be a pandas Series")
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError("cov_matrix must be a pandas DataFrame")
    if not market_weights.index.is_unique:
        raise ValueError("market_weights index must be unique")
    if (
        cov_matrix.empty
        or not cov_matrix.index.is_unique
        or not cov_matrix.columns.is_unique
    ):
        raise ValueError("cov_matrix must have non-empty unique asset labels")
    assets = list(cov_matrix.index)
    if set(cov_matrix.index) != set(cov_matrix.columns):
        raise ValueError("cov_matrix row and column labels must match")
    if set(market_weights.index) != set(assets):
        raise ValueError("market_weights and cov_matrix must contain the same asset labels")
    aligned_weights = market_weights.reindex(assets)
    try:
        weights = aligned_weights.to_numpy(dtype=float, copy=True)
        covariance = cov_matrix.loc[assets, assets].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("market weights and covariance must be numeric") from exc
    if not np.isfinite(weights).all() or not np.isfinite(covariance).all():
        raise ValueError("market weights and covariance must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("market weights must be long-only and non-negative")
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0 or not np.isclose(weight_sum, 1.0, rtol=1e-9, atol=1e-10):
        raise ValueError("market_weights must sum to 1.0")
    weights = weights / weight_sum
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("cov_matrix must be symmetric")
    covariance = (covariance + covariance.T) / 2.0
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("cov_matrix must be positive definite") from exc
    return weights, covariance, assets


def _validate_risk_aversion(risk_aversion: float) -> float:
    r"""Validate positive risk aversion used in equilibrium and utility.

    The scalar :math:`\lambda>0` scales market risk into equilibrium return;
    for example, :math:`\lambda=2.5` means higher covariance-weighted market
    exposure requires proportionally higher expected return. Complexity:
    :math:`O(1)` time and space.
    """
    try:
        value = float(risk_aversion)
    except (TypeError, ValueError) as exc:
        raise TypeError("risk_aversion must be numeric") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("risk_aversion must be finite and positive")
    return value


def _validated_views(
    views_P: np.ndarray,
    views_Q: np.ndarray,
    omega: np.ndarray | None,
    n_assets: int,
    tau: float,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Validate view dimensions and construct view uncertainty :math:`\Omega`.

    Each row of :math:`P` describes a relative or absolute view, :math:`Q`
    gives its expected return, and :math:`\Omega` describes uncertainty. If
    omitted, diagonal uncertainty is set to the prior variance of each view,
    :math:`\operatorname{diag}(P\tau\Sigma P^T)`. For example, an MSFT-minus-
    AAPL view uses :math:`P=[1,-1]`. Complexity: :math:`O(kn^2+k^3)` time
    and :math:`O(n^2+k^2)` space for :math:`k` views and :math:`n` assets.
    """
    try:
        p = np.asarray(views_P, dtype=float)
        q = np.asarray(views_Q, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("views_P and views_Q must be numeric arrays") from exc
    if p.ndim != 2 or p.shape[1] != n_assets:
        raise ValueError("views_P must have shape (number of views, number of assets)")
    q = q.reshape(-1) if q.ndim in (1, 2) else q
    if q.ndim != 1 or q.size != p.shape[0]:
        raise ValueError("views_Q must contain one value for each row of views_P")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("views_P and views_Q must contain only finite values")
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")

    n_views = p.shape[0]
    if omega is None:
        if n_views == 0:
            omega_matrix = np.empty((0, 0), dtype=float)
        else:
            prior_view_covariance = p @ (tau * covariance) @ p.T
            omega_matrix = np.diag(np.diag(prior_view_covariance))
    else:
        try:
            omega_matrix = np.asarray(omega, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("omega must be a numeric matrix") from exc
        if omega_matrix.shape != (n_views, n_views):
            raise ValueError("omega must have shape (number of views, number of views)")
        if not np.isfinite(omega_matrix).all():
            raise ValueError("omega must contain only finite values")
        if not np.allclose(omega_matrix, omega_matrix.T, rtol=1e-10, atol=1e-12):
            raise ValueError("omega must be symmetric")
        omega_matrix = (omega_matrix + omega_matrix.T) / 2.0
        if n_views and float(np.linalg.eigvalsh(omega_matrix).min()) <= 0.0:
            raise ValueError("omega must be positive definite for non-empty views")
    return p, q, omega_matrix


def implied_market_returns(
    market_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 2.5,
) -> pd.Series:
    r"""Calculate equilibrium returns implied by market capitalization weights.

    The reverse-optimization formula is :math:`\Pi=\lambda\Sigma w_{mkt}`.
    It finds expected returns that make observed market weights optimal under
    mean-variance utility. For example, with MSFT/AAPL market weights
    :math:`[0.6,0.4]`, the covariance-weighted portfolio exposures establish
    the equilibrium prior before adding a Moat view. Complexity:
    :math:`O(n^2)` time and :math:`O(n^2)` input space.
    """
    weights, covariance, assets = _validated_market_inputs(market_weights, cov_matrix)
    risk_aversion_value = _validate_risk_aversion(risk_aversion)
    implied = risk_aversion_value * (covariance @ weights)
    return pd.Series(implied, index=assets, name="implied_returns", dtype=float)


def black_litterman_posterior(
    implied_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    views_P: np.ndarray,
    views_Q: np.ndarray,
    omega: np.ndarray | None = None,
    tau: float = 0.05,
) -> tuple[pd.Series, pd.DataFrame]:
    r"""Blend equilibrium returns with analyst views using a Bayesian update.

    With prior mean :math:`\Pi`, prior uncertainty :math:`\tau\Sigma`, and
    view :math:`Q=P\mu+\epsilon`, posterior mean and covariance are
    :math:`\mu_{BL}=\Pi+\tau\Sigma P^T(P\tau\Sigma P^T+\Omega)^{-1}
    (Q-P\Pi)` and
    :math:`M_{BL}=\tau\Sigma-\tau\Sigma P^T(P\tau\Sigma P^T+\Omega)^{-1}
    P\tau\Sigma`. The covariance returned is posterior uncertainty in the
    expected-return vector. For two assets MSFT/AAPL, a positive
    :math:`Q` on :math:`P=[1,-1]` raises the MSFT-over-AAPL view; high
    :math:`\Omega` keeps the posterior close to equilibrium. Complexity:
    :math:`O(n^3+k^3+kn^2)` time and :math:`O(n^2+k^2)` space.
    """
    if not isinstance(implied_returns, pd.Series):
        raise TypeError("implied_returns must be a pandas Series")
    if not implied_returns.index.is_unique:
        raise ValueError("implied_returns index must be unique")
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError("cov_matrix must be a pandas DataFrame")
    if not cov_matrix.index.is_unique or not cov_matrix.columns.is_unique:
        raise ValueError("cov_matrix labels must be unique")
    assets = list(cov_matrix.index)
    if set(cov_matrix.index) != set(cov_matrix.columns):
        raise ValueError("cov_matrix row and column labels must match")
    if set(implied_returns.index) != set(assets):
        raise ValueError("implied_returns and cov_matrix must contain the same asset labels")
    try:
        prior = implied_returns.reindex(assets).to_numpy(dtype=float, copy=True)
        covariance = cov_matrix.loc[assets, assets].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("implied returns and covariance must be numeric") from exc
    if not np.isfinite(prior).all() or not np.isfinite(covariance).all():
        raise ValueError("implied returns and covariance must be finite")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("cov_matrix must be symmetric")
    covariance = (covariance + covariance.T) / 2.0
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("cov_matrix must be positive definite") from exc
    try:
        tau_value = float(tau)
    except (TypeError, ValueError) as exc:
        raise TypeError("tau must be numeric") from exc
    if not math.isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError("tau must be finite and positive")

    p, q, omega_matrix = _validated_views(
        views_P, views_Q, omega, len(assets), tau_value, covariance
    )
    tau_covariance = tau_value * covariance
    if p.shape[0] == 0:
        posterior_mean = prior.copy()
        posterior_uncertainty = tau_covariance.copy()
    else:
        view_covariance = p @ tau_covariance @ p.T + omega_matrix
        try:
            gain_residual = np.linalg.solve(view_covariance, q - p @ prior)
            gain_projection = np.linalg.solve(view_covariance, p @ tau_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("view covariance must be invertible") from exc
        posterior_mean = prior + tau_covariance @ p.T @ gain_residual
        posterior_uncertainty = tau_covariance - tau_covariance @ p.T @ gain_projection
        posterior_uncertainty = (posterior_uncertainty + posterior_uncertainty.T) / 2.0
    return (
        pd.Series(posterior_mean, index=assets, name="posterior_returns", dtype=float),
        pd.DataFrame(posterior_uncertainty, index=assets, columns=assets, dtype=float),
    )


def view_impact_telemetry(
    market_weights: pd.Series,
    optimized_weights: pd.Series,
    implied_returns: pd.Series,
    posterior_returns: pd.Series,
    latency_ms: float,
) -> dict[str, Any]:
    r"""Summarize how analyst views moved the allocation and return prior.

    The weight shift is :math:`\Delta w=w_{BL}-w_{mkt}` and the return
    revision is :math:`\Delta\mu=\mu_{BL}-\Pi`. For example, an MSFT weight
    rising from 60% to 65% gives :math:`\Delta w_{MSFT}=+5` percentage points.
    Latency is supplied by the complete pipeline caller. Complexity:
    :math:`O(n)` time and output space.
    """
    series = (market_weights, optimized_weights, implied_returns, posterior_returns)
    if any(not isinstance(value, pd.Series) for value in series):
        raise TypeError("telemetry inputs must be pandas Series")
    labels = market_weights.index
    if any(not value.index.is_unique or set(value.index) != set(labels) for value in series):
        raise ValueError("telemetry Series must have identical unique asset labels")
    try:
        latency = float(latency_ms)
        market = market_weights.reindex(labels).to_numpy(dtype=float)
        weights = optimized_weights.reindex(labels).to_numpy(dtype=float)
        prior = implied_returns.reindex(labels).to_numpy(dtype=float)
        posterior = posterior_returns.reindex(labels).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("telemetry values must be numeric") from exc
    if not math.isfinite(latency) or latency < 0.0:
        raise ValueError("latency_ms must be finite and non-negative")
    if not all(np.isfinite(values).all() for values in (market, weights, prior, posterior)):
        raise ValueError("telemetry values must be finite")
    weight_delta = weights - market
    return {
        "latency_ms": latency,
        "view_return_delta": pd.Series(posterior - prior, index=labels, dtype=float).to_dict(),
        "view_weight_delta": pd.Series(weight_delta, index=labels, dtype=float).to_dict(),
        "weight_l1_delta": float(np.abs(weight_delta).sum()),
    }


def optimize_black_litterman(
    market_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    views_P: np.ndarray,
    views_Q: np.ndarray,
    omega: np.ndarray | None = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> dict[str, Any]:
    r"""Run the full Black-Litterman update and long-only mean-variance solve.

    It computes :math:`\Pi=\lambda\Sigma w_{mkt}`, updates that prior with
    the analyst view posterior, and maximizes
    :math:`w^T\mu_{BL}-\frac{\lambda}{2}w^T\Sigma w` subject to
    :math:`\mathbf{1}^Tw=1` and :math:`0\le w_i\le1`. In the MSFT/AAPL
    example, market weights ``[0.6, 0.4]`` are the no-view baseline; a
    confident Moat view that MSFT will outperform can shift weight toward
    MSFT, while 1% pure alpha remains distinct from the factor/view-driven
    revision. Results include market/prior/posterior values, optimization
    status, latency, and view impact. Complexity: :math:`O(n^3+k^3+In^2)`
    time and :math:`O(n^2+k^2)` space for :math:`I` SLSQP iterations.
    """
    start = perf_counter()
    weights, covariance, assets = _validated_market_inputs(market_weights, cov_matrix)
    risk_aversion_value = _validate_risk_aversion(risk_aversion)
    implied = pd.Series(
        risk_aversion_value * (covariance @ weights),
        index=assets,
        name="implied_returns",
        dtype=float,
    )
    posterior, posterior_covariance = black_litterman_posterior(
        implied,
        cov_matrix,
        views_P,
        views_Q,
        omega=omega,
        tau=tau,
    )
    expected = posterior.to_numpy(dtype=float)

    def negative_utility(candidate: np.ndarray) -> float:
        r"""Return negative mean-variance utility in :math:`O(n^2)` time."""
        return -float(
            candidate @ expected
            - 0.5 * risk_aversion_value * (candidate @ covariance @ candidate)
        )

    result = minimize(
        negative_utility,
        weights.copy(),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(assets),
        constraints=[{"type": "eq", "fun": lambda candidate: float(candidate.sum() - 1.0)}],
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    optimized = np.asarray(result.x, dtype=float)
    bounds_error = max(
        float(np.maximum(-optimized, 0.0).max()),
        float(np.maximum(optimized - 1.0, 0.0).max()),
    )
    if (
        not result.success
        or not np.isfinite(optimized).all()
        or abs(float(optimized.sum()) - 1.0) > 1e-7
        or bounds_error > 1e-7
    ):
        raise RuntimeError(
            f"Black-Litterman optimization failed: {result.message}; "
            f"bounds_error={bounds_error:.3g}"
        )
    correction_index = int(np.argmax(optimized))
    optimized[correction_index] += 1.0 - float(optimized.sum())
    optimized_series = pd.Series(optimized, index=assets, name="weights", dtype=float)
    elapsed_ms = (perf_counter() - start) * 1000.0
    telemetry = view_impact_telemetry(
        market_weights.reindex(assets),
        optimized_series,
        implied,
        posterior,
        elapsed_ms,
    )
    return {
        "weights": optimized_series,
        "implied_returns": implied,
        "posterior_returns": posterior,
        "posterior_covariance": posterior_covariance,
        "portfolio_expected_return": float(optimized @ expected),
        "portfolio_volatility": math.sqrt(float(optimized @ covariance @ optimized)),
        "success": bool(result.success),
        "converged": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "latency_ms": float(elapsed_ms),
        "input_shape": {
            "market_weights": [int(len(assets))],
            "cov_matrix": [int(len(assets)), int(len(assets))],
            "views_P": [int(value) for value in np.asarray(views_P).shape],
            "views_Q": [int(value) for value in np.asarray(views_Q).shape],
        },
        "view_return_delta": telemetry["view_return_delta"],
        "view_weight_delta": telemetry["view_weight_delta"],
        "weight_l1_delta": telemetry["weight_l1_delta"],
    }
