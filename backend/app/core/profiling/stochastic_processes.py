"""Markov-chain simulation and martingale diagnostics for quant research.

All routines are in-memory numerical computations using NumPy, Pandas, and
SciPy. The module performs no filesystem, database, network, or web I/O.
"""

from __future__ import annotations

import math
from time import perf_counter_ns
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.stats import chi2, norm


def _validate_transition_matrix(transition_matrix: np.ndarray) -> np.ndarray:
    r"""Validate a finite row-stochastic matrix in :math:`O(S^2)` time.

    Every entry must satisfy :math:`P_{ij}\ge0` and each row must satisfy
    :math:`\sum_jP_{ij}=1`. For a three-regime Bull/Bear/Sideways model, a row
    such as ``[0.8, 0.1, 0.1]`` means an 80% chance that a Bull regime persists.
    Validation uses :math:`O(S^2)` storage for numeric conversion, with
    :math:`S` states.
    """
    try:
        matrix = np.asarray(transition_matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("transition_matrix must be a numeric square array") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition_matrix must be a non-empty square matrix")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("transition_matrix must contain finite non-negative probabilities")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("each transition_matrix row must sum to 1.0")
    return matrix


def calculate_stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    r"""Calculate a stationary state distribution using the left eigenspace of ``P``.

    The stationary row vector :math:`\pi` satisfies
    :math:`\pi P=\pi`, :math:`\sum_i\pi_i=1`, and :math:`\pi_i\ge0`. The
    implementation finds the null space of :math:`P^T-I`, which is the
    eigenvector condition for eigenvalue one, then normalizes the result. When
    there are multiple stationary classes, the projection of a uniform initial
    distribution onto the stationary eigenspace selects a valid mixture. For
    example, a 3-state Bull/Bear/Sideways matrix can yield ``[0.6, 0.2, 0.2]``,
    meaning the long-run regime occupancy is 60%, 20%, and 20%. Runtime is
    :math:`O(S^3)` due to eigen/null-space decomposition; storage is
    :math:`O(S^2)`.
    """
    matrix = _validate_transition_matrix(transition_matrix)
    n_states = matrix.shape[0]
    basis = null_space(matrix.T - np.eye(n_states), rcond=1e-12)
    if basis.shape[1] == 0:
        raise ValueError("transition matrix has no numerically stable stationary eigenspace")
    if basis.shape[1] == 1:
        candidate = np.real_if_close(basis[:, 0]).astype(np.float64)
        if candidate.sum() < 0:
            candidate = -candidate
        candidate = np.maximum(candidate, 0.0)
        if candidate.sum() == 0.0:
            candidate = np.abs(np.real(basis[:, 0]))
    else:
        # A convex projection of the uniform prior onto the stationary space
        # avoids choosing an arbitrary signed vector for reducible chains.
        projection = basis @ (basis.T @ np.full(n_states, 1.0 / n_states))
        candidate = np.maximum(np.real(projection), 0.0)
        if candidate.sum() <= np.finfo(float).tiny:
            candidate = np.full(n_states, 1.0 / n_states)
    stationary = candidate / candidate.sum()
    # Remove tiny numerical residuals, then renormalize. This improves the
    # returned vector's probability interpretation while preserving invariance.
    stationary[np.abs(stationary) < 1e-15] = 0.0
    stationary /= stationary.sum()
    return stationary


def simulate_markov_chain_path(
    transition_matrix: np.ndarray,
    initial_state: int,
    n_steps: int,
    seed: int | None = None,
) -> np.ndarray:
    r"""Simulate categorical states from a row-stochastic transition matrix.

    The Markov property is :math:`P(X_{t+1}=j\mid X_t=i,\mathcal F_{t-1})
    =P_{ij}`. At every step, one categorical draw selects the next state from
    row :math:`i`. For example, a Bull/Bear/Sideways model can use a Bull row
    ``[0.8, 0.05, 0.15]`` to retain Bull with 80% probability. The result has
    length ``n_steps + 1`` and includes the initial state. Runtime is
    :math:`O(n_{steps}S)` for categorical sampling; storage is
    :math:`O(n_{steps})`. An optional seed enables exact replay.
    """
    matrix = _validate_transition_matrix(transition_matrix)
    if isinstance(n_steps, (bool, np.bool_)) or not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    n_states = matrix.shape[0]
    if isinstance(initial_state, (bool, np.bool_)) or not isinstance(initial_state, (int, np.integer)):
        raise ValueError("initial_state must be an integer state index")
    if not 0 <= initial_state < n_states:
        raise ValueError("initial_state is outside the transition matrix state range")
    if seed is not None and (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))):
        raise ValueError("seed must be an integer or None")
    rng = np.random.default_rng(seed)
    states = np.empty(int(n_steps) + 1, dtype=np.int64)
    states[0] = int(initial_state)
    for step in range(1, int(n_steps) + 1):
        states[step] = rng.choice(n_states, p=matrix[states[step - 1]])
    return states


def test_martingale_property(price_series: pd.Series, max_lags: int = 5) -> dict[str, Any]:
    r"""Test log-return martingale-difference diagnostics and variance-ratio scaling.

    A martingale price process obeys :math:`E[X_{t+1}\mid\mathcal F_t]=X_t`;
    equivalently price increments have zero conditional mean. This routine
    uses log returns :math:`r_t=\log(P_t/P_{t-1})` as a practical diagnostic,
    testing their unconditional mean and serial autocorrelation plus the
    variance-ratio null :math:`VR(k)=Var(\sum_{j=0}^{k-1}r_{t-j})/(kVar(r_t))=1`.
    For example, a simulated random walk with Bull/Bear/Sideways labels that
    do not affect its zero-drift shocks should generally have variance ratios
    near one and non-significant MDS diagnostics. Non-rejection is not proof of
    a martingale, and log-price martingality is not identical to price
    martingality. Complexity is :math:`O(NL + L^2)` time for :math:`N` prices
    and :math:`L` tested lags (including autocorrelation sums); working memory
    is :math:`O(N)`.
    """
    started = perf_counter_ns()
    if not isinstance(price_series, pd.Series):
        raise TypeError("price_series must be a pandas Series")
    if isinstance(max_lags, (bool, np.bool_)) or not isinstance(max_lags, (int, np.integer)) or max_lags < 1:
        raise ValueError("max_lags must be a positive integer")
    prices = pd.to_numeric(price_series, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if prices.size < 3:
        raise ValueError("price_series must contain at least three valid prices")
    if not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ValueError("price_series must contain finite positive prices")
    log_returns = np.diff(np.log(prices))
    n_obs = log_returns.size
    if n_obs < 2:
        raise ValueError("at least two valid log returns are required")
    centered = log_returns - log_returns.mean()
    variance = float(np.dot(centered, centered) / n_obs)
    if variance <= np.finfo(float).tiny:
        raise ValueError("log returns must have positive variance")

    max_test_lag = min(int(max_lags), n_obs - 1)
    variance_ratios: dict[int, dict[str, float]] = {}
    for lag in range(2, max_test_lag + 1):
        aggregate = np.convolve(log_returns, np.ones(lag), mode="valid")
        ratio = float(np.var(aggregate, ddof=1) / (lag * np.var(log_returns, ddof=1)))
        standard_error = math.sqrt(2.0 * (2 * lag - 1) * (lag - 1) / (3.0 * lag * n_obs))
        z_stat = (ratio - 1.0) / standard_error
        variance_ratios[lag] = {
            "variance_ratio": ratio,
            "z_statistic": float(z_stat),
            "p_value": float(2.0 * norm.sf(abs(z_stat))),
        }

    # Sample mean test against zero, with asymptotic normal standard error.
    mean = float(log_returns.mean())
    return_std = float(log_returns.std(ddof=1))
    mean_se = return_std / math.sqrt(n_obs)
    mean_z = mean / mean_se if mean_se > 0 else math.copysign(math.inf, mean)
    mean_p = float(2.0 * norm.sf(abs(mean_z)))

    # Ljung-Box portmanteau Q statistic tests the null that autocorrelations
    # through the requested lag are jointly zero (a necessary, not sufficient,
    # martingale-difference condition).
    lag_count = min(int(max_lags), n_obs - 1)
    autocorrelations = []
    for lag in range(1, lag_count + 1):
        autocorrelations.append(float(np.dot(centered[lag:], centered[:-lag]) / np.dot(centered, centered)))
    ljung_box_q = float(
        n_obs * (n_obs + 2.0)
        * sum(rho * rho / (n_obs - lag) for lag, rho in enumerate(autocorrelations, start=1))
    )
    ljung_box_p = float(chi2.sf(ljung_box_q, df=lag_count))
    elapsed_ms = (perf_counter_ns() - started) / 1_000_000.0
    diagnostic_p_values = [mean_p, ljung_box_p] + [item["p_value"] for item in variance_ratios.values()]
    return {
        "is_martingale_like": bool(all(p_value >= 0.05 for p_value in diagnostic_p_values)),
        "log_return_mean": mean,
        "log_return_std": return_std,
        "mean_zero_test": {"statistic": float(mean_z), "p_value": mean_p},
        "martingale_difference_test": {
            "statistic": ljung_box_q,
            "p_value": ljung_box_p,
            "lags": lag_count,
        },
        "variance_ratio_tests": variance_ratios,
        "variance_ratio_max_abs_deviation": float(
            max((abs(item["variance_ratio"] - 1.0) for item in variance_ratios.values()), default=0.0)
        ),
        "n_prices": int(prices.size),
        "n_log_returns": int(n_obs),
        "input_shape": tuple(price_series.shape),
        "returns_shape": tuple(log_returns.shape),
        "latency_ms": float(elapsed_ms),
    }


def stochastic_process_telemetry(
    transition_matrix: np.ndarray,
    stationary_distribution: np.ndarray | None = None,
    latency_ms: float = 0.0,
) -> dict[str, Any]:
    r"""Summarize transition validity, stationary error, timing, and matrix shape.

    The central diagnostic is the infinity norm
    :math:`||\pi P-\pi||_\infty`; it should be close to machine precision for a
    valid stationary vector. In a three-regime Bull/Bear/Sideways example, the
    report might show shape ``(3, 3)`` and error ``2e-16``. Work is
    :math:`O(S^2)` for matrix multiplication, and report storage is
    :math:`O(S)`.
    """
    matrix = _validate_transition_matrix(transition_matrix)
    try:
        elapsed = float(latency_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("latency_ms must be finite and non-negative") from exc
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("latency_ms must be finite and non-negative")
    error: float | None = None
    if stationary_distribution is not None:
        distribution = np.asarray(stationary_distribution, dtype=np.float64)
        if distribution.shape != (matrix.shape[0],) or not np.isfinite(distribution).all():
            raise ValueError("stationary_distribution must be a finite vector matching the state count")
        if np.any(distribution < 0.0) or not np.isclose(distribution.sum(), 1.0, atol=1e-12):
            raise ValueError("stationary_distribution must be non-negative and sum to 1")
        error = float(np.max(np.abs(distribution @ matrix - distribution)))
    return {
        "input_shape": tuple(matrix.shape),
        "latency_ms": elapsed,
        "stationary_convergence_error": error,
        "transition_rows_sum_to_one": bool(np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12)),
    }
