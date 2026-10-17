"""Pure copula fitting and joint simulation for cross-asset returns."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau, norm


def _validated_returns(returns_df: pd.DataFrame, *, bivariate: bool = False) -> np.ndarray:
    """Validate a finite time-by-asset return matrix in O(TA) time and space."""
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame")
    values = returns_df.to_numpy(dtype=float, copy=True)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError("returns_df must contain at least 3 observations and 2 assets")
    if bivariate and values.shape[1] != 2:
        raise ValueError("Clayton copula fitting requires exactly two assets")
    if not np.isfinite(values).all():
        raise ValueError("returns_df must contain only finite values")
    if np.any(np.std(values, axis=0) <= np.finfo(float).eps):
        raise ValueError("each asset must have non-zero return variance")
    return values


def _empirical_uniforms(values: np.ndarray) -> np.ndarray:
    """Map each empirical marginal to open-interval uniforms by ranks; O(TA log T)."""
    n_rows, n_assets = values.shape
    uniforms = np.empty_like(values, dtype=float)
    for column in range(n_assets):
        ranks = stats.rankdata(values[:, column], method="average")
        uniforms[:, column] = ranks / (n_rows + 1.0)
    return uniforms


def fit_gaussian_copula(returns_df: pd.DataFrame) -> dict[str, Any]:
    r"""Fit a Gaussian copula using empirical probability integral transforms.

    Sklar's theorem decomposes a joint CDF as
    :math:`F(x_1,\ldots,x_d)=C(F_1(x_1),\ldots,F_d(x_d))`. Empirical CDF
    ranks map returns to uniforms :math:`U_j\in(0,1)`, then
    :math:`Z_j=\Phi^{-1}(U_j)` and the Gaussian copula is
    :math:`C_R(u)=\Phi_R(\Phi^{-1}(u_1),\ldots,\Phi^{-1}(u_d))`; :math:`R`
    is estimated as the latent normal-score correlation matrix. For two
    assets with Pearson correlation 0.6, Gaussian coupling spreads dependence
    across ordinary and crash periods, while a Clayton copula can model
    stronger lower-tail association despite a similar Pearson correlation.
    Complexity: :math:`O(TA\log T+TA^2)` time and :math:`O(TA+A^2)` space.
    """
    started = perf_counter()
    values = _validated_returns(returns_df)
    uniforms = _empirical_uniforms(values)
    latent = norm.ppf(uniforms)
    correlation = np.corrcoef(latent, rowvar=False)
    correlation = np.atleast_2d(correlation)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    if not np.isfinite(correlation).all():
        raise ValueError("Gaussian copula correlation could not be estimated")
    return {
        "copula_type": "gaussian",
        "correlation_matrix": correlation,
        "uniform_marginals": uniforms,
        "latent_normal_scores": latent,
        "n_observations": int(values.shape[0]),
        "n_assets": int(values.shape[1]),
        "input_shape": [int(values.shape[0]), int(values.shape[1])],
        "tail_dependence_lower": 0.0,
        "tail_dependence_upper": 0.0,
        "latency_ms": float((perf_counter() - started) * 1000.0),
    }


def fit_clayton_copula(returns_df: pd.DataFrame) -> dict[str, Any]:
    r"""Fit a bivariate Clayton copula via Kendall's tau method of moments.

    Clayton's Archimedean generator is
    :math:`\varphi(t)=(t^{-\theta}-1)/\theta` for :math:`\theta>0`, and
    :math:`C(u,v)=(u^{-\theta}+v^{-\theta}-1)^{-1/\theta}`. For
    :math:`\theta>0`, lower-tail dependence is
    :math:`\lambda_L=2^{-1/\theta}` and upper-tail dependence is zero. The
    estimate :math:`\hat\theta=2\hat\tau/(1-\hat\tau)` follows
    :math:`\tau=\theta/(\theta+2)`; negative sample tau maps to independence
    because the standard Clayton family cannot represent negative dependence.
    Two assets may have modest linear correlation yet share a high
    :math:`\lambda_L`, indicating joint downside crashes. Complexity:
    :math:`O(T\log T)` time and :math:`O(T)` space.
    """
    started = perf_counter()
    values = _validated_returns(returns_df, bivariate=True)
    tau, p_value = kendalltau(values[:, 0], values[:, 1], variant="b", nan_policy="raise")
    if not math.isfinite(float(tau)):
        raise ValueError("Kendall tau could not be estimated")
    tau_value = float(tau)
    theta = max(0.0, 2.0 * tau_value / (1.0 - tau_value)) if tau_value < 1.0 else 1e6
    theta = min(theta, 1e6)
    lambda_lower = 0.0 if theta == 0.0 else float(2.0 ** (-1.0 / theta))
    uniforms = _empirical_uniforms(values)
    return {
        "copula_type": "clayton",
        "theta": float(theta),
        "kendall_tau": tau_value,
        "kendall_tau_p_value": float(p_value),
        "uniform_marginals": uniforms,
        "n_observations": int(values.shape[0]),
        "n_assets": 2,
        "input_shape": [int(values.shape[0]), 2],
        "tail_dependence_lower": lambda_lower,
        "tail_dependence_upper": 0.0,
        "latency_ms": float((perf_counter() - started) * 1000.0),
    }


def _marginal_ppf(specification: dict[str, Any], uniforms: np.ndarray) -> np.ndarray:
    """Apply one configured SciPy marginal quantile function; O(M log M) or distribution-specific."""
    if not isinstance(specification, dict):
        raise TypeError("each marginal specification must be a dictionary")
    distribution = specification.get("distribution", specification.get("name"))
    if distribution is None:
        raise ValueError("marginal specification requires a 'distribution' or 'name'")
    args = specification.get("args", ())
    params = specification.get("params", {})
    if not isinstance(args, (list, tuple)) or not isinstance(params, dict):
        raise ValueError("marginal 'args' must be a sequence and 'params' must be a dictionary")
    if isinstance(distribution, str):
        dist = getattr(stats, distribution, None)
        if dist is None or not callable(getattr(dist, "ppf", None)):
            raise ValueError(f"unknown SciPy distribution: {distribution}")
    elif callable(getattr(distribution, "ppf", None)):
        dist = distribution
    else:
        raise TypeError("distribution must be a SciPy distribution name or object with ppf")
    try:
        result = np.asarray(dist.ppf(uniforms, *args, **params), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid parameters for marginal distribution {distribution}") from exc
    if result.shape != uniforms.shape or not np.isfinite(result).all():
        raise ValueError("marginal quantile function produced invalid samples")
    return result


def simulate_copula_joint_returns(
    copula_params: dict[str, Any],
    marginal_distributions: list[dict[str, Any]],
    n_samples: int = 10000,
    copula_type: str = "gaussian",
) -> np.ndarray:
    r"""Generate dependent returns and transform uniforms through chosen marginals.

    The simulation implements Sklar's composition
    :math:`X_j=F_j^{-1}(U_j)`. Gaussian uniforms use a correlated normal draw
    :math:`Z\sim N(0,R), U_j=\Phi(Z_j)`. Clayton uniforms use the generator
    :math:`\varphi(t)=(t^{-\theta}-1)/\theta`, sampled by the shared Gamma
    frailty representation. For example, a bivariate crash-sensitive Clayton
    pair can use Student-t marginals to combine heavy marginal tails with
    positive lower-tail dependence; Pearson correlation alone may obscure that
    joint downside tendency. Complexity: Gaussian :math:`O(Md^3)` for dense
    factorization, Clayton :math:`O(Md)`; storage :math:`O(Md+d^2)`.
    """
    if not isinstance(copula_params, dict):
        raise TypeError("copula_params must be a dictionary")
    if not isinstance(marginal_distributions, list) or not marginal_distributions:
        raise ValueError("marginal_distributions must be a non-empty list")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if copula_type not in {"gaussian", "clayton"}:
        raise ValueError("copula_type must be 'gaussian' or 'clayton'")
    dimensions = len(marginal_distributions)
    rng = np.random.default_rng(copula_params.get("seed"))
    tiny = np.finfo(float).eps

    if copula_type == "gaussian":
        correlation = np.asarray(copula_params.get("correlation_matrix"), dtype=float)
        if correlation.shape != (dimensions, dimensions) or not np.isfinite(correlation).all():
            raise ValueError("correlation_matrix must be finite and match the number of marginals")
        if not np.allclose(correlation, correlation.T, atol=1e-10):
            raise ValueError("correlation_matrix must be symmetric")
        if not np.allclose(np.diag(correlation), 1.0, atol=1e-8):
            raise ValueError("correlation_matrix must have unit diagonal")
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        if float(eigenvalues.min()) < -1e-8:
            raise ValueError("correlation_matrix must be positive semidefinite")
        factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
        latent = rng.standard_normal((n_samples, dimensions)) @ factor.T
        uniforms = norm.cdf(latent)
    else:
        if dimensions != 2:
            raise ValueError("Clayton simulation currently supports exactly two marginals")
        try:
            theta = float(copula_params["theta"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Clayton copula_params must include numeric theta") from exc
        if not math.isfinite(theta) or theta < 0.0:
            raise ValueError("Clayton theta must be finite and non-negative")
        if theta <= 1e-12:
            uniforms = rng.uniform(tiny, 1.0 - tiny, size=(n_samples, dimensions))
        else:
            frailty = rng.gamma(shape=1.0 / theta, scale=1.0, size=n_samples)
            exponentials = rng.exponential(scale=1.0, size=(n_samples, dimensions))
            uniforms = (1.0 + exponentials / frailty[:, None]) ** (-1.0 / theta)
    uniforms = np.clip(uniforms, tiny, 1.0 - tiny)
    output = np.column_stack([
        _marginal_ppf(specification, uniforms[:, column])
        for column, specification in enumerate(marginal_distributions)
    ])
    return output


def copula_telemetry(
    returns_df: pd.DataFrame, copula_type: str = "gaussian"
) -> dict[str, Any]:
    r"""Expose copula fit runtime, input shape, dependence and tail coefficients.

    Gaussian copulas have :math:`\lambda_L=\lambda_U=0` asymptotically unless
    correlation is perfect; Clayton reports lower-tail dependence
    :math:`\lambda_L=2^{-1/\theta}` and zero upper-tail dependence. Complexity
    follows the corresponding fit: :math:`O(TA\log T+TA^2)` for Gaussian and
    :math:`O(T\log T)` for bivariate Clayton.
    """
    started = perf_counter()
    if copula_type == "gaussian":
        fitted = fit_gaussian_copula(returns_df)
    elif copula_type == "clayton":
        fitted = fit_clayton_copula(returns_df)
    else:
        raise ValueError("copula_type must be 'gaussian' or 'clayton'")
    telemetry = {
        key: fitted[key]
        for key in (
            "copula_type", "input_shape", "latency_ms", "tail_dependence_lower",
            "tail_dependence_upper", "correlation_matrix", "theta", "kendall_tau",
        )
        if key in fitted
    }
    telemetry["latency_ms"] = float((perf_counter() - started) * 1000.0)
    return telemetry
