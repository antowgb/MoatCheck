"""Vectorized Monte Carlo simulation helpers for quantitative finance.

The module is an in-memory numerical engine: it performs no filesystem,
network, database, or web-framework operations. Simulation arrays are
time-major so each time slice can be processed efficiently with NumPy.
"""

from __future__ import annotations

import math
from time import perf_counter_ns
from typing import Any

import numpy as np
from scipy.stats import binom


def _finite_scalar(value: float, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    r"""Convert and validate a finite scalar in :math:`O(1)` time and space.

    This centralizes input checks without touching external state. For example,
    an annual volatility of ``0.2`` is valid, while ``NaN`` and negative sigma
    are rejected before any simulation arrays are allocated.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_int(value: int, name: str) -> int:
    r"""Validate a positive integer parameter in :math:`O(1)` time and space."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _normal_draws(
    rng: np.random.Generator, n_steps: int, n_sims: int, use_antithetic: bool
) -> np.ndarray:
    r"""Generate standard-normal shocks, pairing antithetic draws when requested.

    For antithetic sampling, paired shocks are :math:`Z` and :math:`-Z`, which
    preserves the Gaussian marginal while reducing variance of many smooth
    estimators. Complexity and memory are :math:`O(n_{steps}n_{sims})`.
    """
    if not use_antithetic or n_sims == 1:
        return rng.standard_normal((n_steps, n_sims))
    n_pairs = n_sims // 2
    base = rng.standard_normal((n_steps, n_pairs))
    shocks = np.empty((n_steps, n_sims), dtype=np.float64)
    shocks[:, :n_pairs] = base
    shocks[:, n_pairs : 2 * n_pairs] = -base
    if n_sims % 2:
        shocks[:, -1] = rng.standard_normal(n_steps)
    return shocks


def _normal_draws_multi(
    rng: np.random.Generator,
    n_steps: int,
    n_sims: int,
    n_assets: int,
    use_antithetic: bool,
) -> np.ndarray:
    r"""Generate independent normals by asset with simulation-wise antithetic pairs.

    The output has shape ``(steps, simulations, assets)``. Pairing along the
    simulation axis ensures corresponding asset shocks are negated together
    before correlation is applied. Runtime and storage are :math:`O(TMA)`.
    """
    if not use_antithetic or n_sims == 1:
        return rng.standard_normal((n_steps, n_sims, n_assets))
    n_pairs = n_sims // 2
    base = rng.standard_normal((n_steps, n_pairs, n_assets))
    shocks = np.empty((n_steps, n_sims, n_assets), dtype=np.float64)
    shocks[:, :n_pairs, :] = base
    shocks[:, n_pairs : 2 * n_pairs, :] = -base
    if n_sims % 2:
        shocks[:, -1, :] = rng.standard_normal((n_steps, n_assets))
    return shocks


def _covariance_square_root(covariance: np.ndarray) -> tuple[np.ndarray, str, int]:
    r"""Factor a covariance matrix by Cholesky, with a PSD spectral fallback.

    Positive-definite matrices use :math:`\Sigma=LL^T` from Cholesky. If
    Cholesky fails because the matrix is singular or semidefinite, eigenvalue
    decomposition supplies :math:`A=V\Lambda_+^{1/2}` such that
    :math:`AA^T\approx\Sigma`; only roundoff-scale negative eigenvalues are
    clipped. Work is :math:`O(A^3)` and factor storage is :math:`O(A^2)` for
    :math:`A` assets. For example, identical asset returns produce a singular
    covariance and use the spectral fallback, while a strictly positive
    covariance uses Cholesky.
    """
    try:
        return np.linalg.cholesky(covariance), "cholesky", 0
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tolerance = np.finfo(np.float64).eps * max(covariance.shape) * max(
            float(np.max(np.abs(eigenvalues))), 1.0
        )
        if float(eigenvalues.min()) < -tolerance:
            raise ValueError("covariance_matrix is not positive semidefinite")
        clipped = int(np.count_nonzero(eigenvalues < 0.0))
        factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
        return factor, "spectral_fallback", clipped


def simulate_geometric_brownian_motion(
    s0: float,
    mu: float,
    sigma: float,
    n_sims: int,
    n_steps: int,
    dt: float,
    use_antithetic: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    r"""Simulate non-negative GBM price paths with exact lognormal transitions.

    The model is :math:`dS_t=\mu S_tdt+\sigma S_tdW_t`, with transition
    :math:`S_{t+1}=S_t\exp((\mu-\frac12\sigma^2)\Delta t+
    \sigma\sqrt{\Delta t}Z_t)`. The exponential update enforces non-negative
    prices even for high volatility and negative drift; overflow is rejected
    and underflow may reach the zero floor. With
    antithetic variates, paired paths use :math:`Z_t` and :math:`-Z_t`; for an
    estimator :math:`\hat\theta`, the paired estimator is
    :math:`(g(Z)+g(-Z))/2`, whose variance is often below that of independent
    draws. A 10,000-path example for a €100 asset creates a matrix of
    ``(n_steps + 1, 10_000)`` prices and averages paired random shocks to
    stabilize terminal statistics. Complexity and output memory are both
    :math:`O(n_{steps}n_{sims})`.

    ``seed`` is an optional reproducibility control; it extends the requested
    interface while leaving the default call behavior stochastic.
    """
    initial = _finite_scalar(s0, "s0", positive=True)
    drift = _finite_scalar(mu, "mu")
    volatility = _finite_scalar(sigma, "sigma", nonnegative=True)
    step = _finite_scalar(dt, "dt", positive=True)
    simulations = _positive_int(n_sims, "n_sims")
    steps = _positive_int(n_steps, "n_steps")
    if not isinstance(use_antithetic, (bool, np.bool_)):
        raise ValueError("use_antithetic must be a boolean")
    if seed is not None and (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))):
        raise ValueError("seed must be an integer or None")

    rng = np.random.default_rng(seed)
    shocks = _normal_draws(rng, steps, simulations, bool(use_antithetic))
    log_increments = (drift - 0.5 * volatility**2) * step + volatility * math.sqrt(step) * shocks
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        factors = np.exp(log_increments)
    paths = np.empty((steps + 1, simulations), dtype=np.float64)
    paths[0] = initial
    paths[1:] = initial * np.cumprod(factors, axis=0)
    if not np.isfinite(paths).all() or np.any(paths < 0.0):
        raise ValueError("simulation produced invalid prices; check parameters and horizon")
    return paths


def calculate_monte_carlo_var_cvar(
    paths: np.ndarray, confidence_level: float = 0.99
) -> dict[str, Any]:
    r"""Calculate terminal loss VaR/CVaR and an order-statistic VaR interval.

    Each path's simple terminal return is :math:`R=S_T/S_0-1` and its loss is
    :math:`L=-R`. Positive VaR is the :math:`q` quantile of losses, with
    :math:`q=confidence\_level`; CVaR is the average loss in the upper tail,
    :math:`E[L\mid L\ge VaR_q]`. A two-sided confidence interval for the
    population quantile is estimated from binomial order-statistic ranks. For
    10,000 simulated outcomes, a 99% VaR means roughly the worst 100 terminal
    returns determine the loss threshold. Runtime is :math:`O(M\log M)` due to
    quantiles; working memory is :math:`O(M)` for :math:`M` paths.

    Returns loss amounts as positive numbers when the simulated outcome is a
    loss, plus equivalent signed return quantiles and terminal telemetry.
    """
    level = _finite_scalar(confidence_level, "confidence_level")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    try:
        matrix = np.asarray(paths, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("paths must be a numeric two-dimensional array") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("paths must have shape (time, simulations) with non-empty axes")
    if not np.isfinite(matrix).all() or np.any(matrix[0] <= 0.0):
        raise ValueError("paths must be finite and have positive initial prices")

    returns = matrix[-1] / matrix[0] - 1.0
    losses = -returns
    var_loss = float(np.quantile(losses, level))
    tail_losses = losses[losses >= var_loss]
    cvar_loss = float(tail_losses.mean())
    n = losses.size
    alpha = 1.0 - level
    lower_rank = int(binom.ppf(alpha / 2.0, n, level))
    upper_rank = int(binom.ppf(1.0 - alpha / 2.0, n, level))
    ordered = np.sort(losses)
    low_index = int(np.clip(lower_rank - 1, 0, n - 1))
    high_index = int(np.clip(upper_rank, 0, n - 1))
    interval = (float(ordered[low_index]), float(ordered[high_index]))
    return {
        "confidence_level": level,
        "n_simulations": int(n),
        "var": var_loss,
        "cvar": cvar_loss,
        "var_loss_amount": var_loss,
        "cvar_loss_amount": cvar_loss,
        "var_return_quantile": -var_loss,
        "cvar_return_tail_mean": -cvar_loss,
        "var_confidence_interval": interval,
        "terminal_return_mean": float(returns.mean()),
        "terminal_return_std": float(returns.std(ddof=1)) if n > 1 else 0.0,
        "terminal_mean": float(matrix[-1].mean()),
        "terminal_std": float(matrix[-1].std(ddof=1)) if n > 1 else 0.0,
        "input_shape": tuple(matrix.shape),
        "tail_observations": int(tail_losses.size),
    }


class MonteCarloEngine:
    r"""Simulate correlated multi-asset GBM paths and portfolio wealth.

    The covariance matrix :math:`\Sigma` is annualized covariance of asset
    returns. Positive-definite matrices use Cholesky
    :math:`\Sigma=LL^T`, yielding :math:`Z_c=ZL^T`; singular positive-
    semidefinite matrices use a spectral square root fallback. Each step applies
    the exact lognormal transition
    :math:`S_{t+1}=S_t\odot\exp((\mu-\frac12\operatorname{diag}(\Sigma))\Delta t+
    Z_c\sqrt{\Delta t})`, so prices stay non-negative. This gives
    portfolio risk a common shock structure rather than simulating assets
    independently. For instance, a two-asset technology/healthcare portfolio
    with positive covariance will experience more joint moves than an
    independent simulation. A 10,000-path, 252-step, 10-asset run stores about
    202 MB of float64 asset paths. Complexity is :math:`O(TMA^2)` including
    Cholesky multiplication, and memory is :math:`O(TMA)`, for steps :math:`T`,
    simulations :math:`M`, and assets :math:`A`.
    """

    def __init__(
        self,
        initial_values: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> None:
        r"""Validate and retain portfolio inputs in :math:`O(A^2)` time/space.

        ``initial_values`` are positive asset notionals or prices, expected
        returns and covariance are annualized. For two assets, a covariance
        matrix ``[[.04, .01], [.01, .09]]`` represents 20% and 30% volatility
        with positive co-movement. Matrix validation costs :math:`O(A^2)`; the
        stored arrays require :math:`O(A^2)` space.
        """
        values = np.asarray(initial_values, dtype=np.float64)
        means = np.asarray(expected_returns, dtype=np.float64)
        covariance = np.asarray(covariance_matrix, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("initial_values must be a non-empty one-dimensional vector")
        n_assets = values.size
        if means.shape != (n_assets,) or covariance.shape != (n_assets, n_assets):
            raise ValueError("expected_returns and covariance_matrix dimensions must match initial_values")
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("initial_values must be finite and positive")
        if not np.isfinite(means).all() or not np.isfinite(covariance).all():
            raise ValueError("expected_returns and covariance_matrix must be finite")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("covariance_matrix must be symmetric")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if float(eigenvalues.min()) < -1e-10:
            raise ValueError("covariance_matrix must be positive semidefinite")
        self.initial_values = values.copy()
        self.expected_returns = means.copy()
        self.covariance_matrix = covariance.copy()
        self.n_assets = n_assets

    def simulate(
        self,
        n_sims: int = 10_000,
        n_steps: int = 252,
        dt: float = 1 / 252,
        *,
        use_antithetic: bool = True,
        seed: int | None = None,
    ) -> dict[str, Any]:
        r"""Run correlated GBM paths and return assets, portfolio, and telemetry.

        Correlated shocks are :math:`Z_c=ZA^T`, with :math:`AA^T=\Sigma`; the
        factor :math:`A` is Cholesky for positive-definite matrices and a
        spectral square root only when Cholesky fails on a semidefinite/singular
        covariance. Exact log increments are
        :math:`(\mu_i-\frac12\Sigma_{ii})\Delta t+\sqrt{\Delta t}Z_{c,i}`.
        Asset paths have shape
        :math:`(T+1,M,A)` and portfolio value is their per-period sum. Runtime
        is :math:`O(TMA^2)` and stored path memory is :math:`O(TMA)`. A sample
        10,000-path simulation exposes terminal mean/std and throughput so
        users can compare a €100 starting asset's simulated distribution with
        its theoretical growth.
        """
        simulations = _positive_int(n_sims, "n_sims")
        steps = _positive_int(n_steps, "n_steps")
        step = _finite_scalar(dt, "dt", positive=True)
        if not isinstance(use_antithetic, (bool, np.bool_)):
            raise ValueError("use_antithetic must be a boolean")
        if seed is not None and (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))):
            raise ValueError("seed must be an integer or None")
        started = perf_counter_ns()
        rng = np.random.default_rng(seed)
        z = _normal_draws_multi(rng, steps, simulations, self.n_assets, bool(use_antithetic))
        covariance_factor, covariance_factorization, clipped_eigenvalues = _covariance_square_root(
            self.covariance_matrix
        )
        correlated = z @ covariance_factor.T
        log_increments = (
            self.expected_returns[None, None, :]
            - 0.5 * np.diag(self.covariance_matrix)[None, None, :]
        ) * step + correlated * math.sqrt(step)
        gross_factors = np.exp(log_increments)
        asset_paths = np.empty((steps + 1, simulations, self.n_assets), dtype=np.float64)
        asset_paths[0] = self.initial_values
        asset_paths[1:] = self.initial_values[None, None, :] * np.cumprod(gross_factors, axis=0)
        if not np.isfinite(asset_paths).all():
            raise ValueError("simulation produced non-finite values; check parameters and horizon")
        portfolio_paths = asset_paths.sum(axis=2)
        elapsed_ms = (perf_counter_ns() - started) / 1_000_000
        terminal = portfolio_paths[-1]
        return {
            "asset_paths": asset_paths,
            "portfolio_paths": portfolio_paths,
            "input_shape": {"initial_values": self.initial_values.shape, "covariance_matrix": self.covariance_matrix.shape},
            "asset_paths_shape": tuple(asset_paths.shape),
            "portfolio_paths_shape": tuple(portfolio_paths.shape),
            "latency_ms": float(elapsed_ms),
            "simulation_throughput_paths_per_second": float(simulations / max(elapsed_ms / 1000, np.finfo(float).tiny)),
            "terminal_mean": float(terminal.mean()),
            "terminal_std": float(terminal.std(ddof=1)) if simulations > 1 else 0.0,
            "memory_bytes": int(asset_paths.nbytes + portfolio_paths.nbytes),
            "n_sims": simulations,
            "n_steps": steps,
            "n_assets": self.n_assets,
            "seed": seed,
            "covariance_factorization": covariance_factorization,
            "spectral_clipped_eigenvalues": clipped_eigenvalues,
        }
