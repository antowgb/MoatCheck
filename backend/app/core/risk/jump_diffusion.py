"""Pure NumPy jump-arrival and Merton jump-diffusion simulation tools."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np


def _finite_parameter(value: float, name: str) -> float:
    """Convert and validate one finite scalar parameter in O(1)."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_dimensions(n_steps: int, n_simulations: int, dt: float) -> float:
    """Validate positive simulation dimensions/time step in O(1)."""
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if isinstance(n_simulations, bool) or not isinstance(n_simulations, int) or n_simulations < 1:
        raise ValueError("n_simulations must be a positive integer")
    step = _finite_parameter(dt, "dt")
    if step <= 0:
        raise ValueError("dt must be strictly positive")
    return step


def simulate_poisson_jumps(
    lambda_param: float,
    n_steps: int,
    n_simulations: int,
    dt: float = 1 / 252,
    seed: int | None = None,
) -> np.ndarray:
    r"""Generate discrete Poisson jump arrival counts per step and simulation.

    If arrivals follow a Poisson process with intensity :math:`\lambda`, then
    :math:`N_{t+\Delta t}-N_t\sim\operatorname{Poisson}(\lambda\Delta t)`.
    The returned integer array has shape :math:`(n_{steps},n_{simulations})`;
    an entry of 2 means two jumps occurred during that step. A €100 stock
    simulated over one trading year at :math:`\lambda=2` has about two jump
    events per path on average, independent of their eventual price impact.
    Complexity: :math:`O(n_{steps}n_{simulations})` time and output memory.
    """
    intensity = _finite_parameter(lambda_param, "lambda_param")
    step = _validate_dimensions(n_steps, n_simulations, dt)
    if intensity < 0.0:
        raise ValueError("lambda_param must be non-negative")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
        raise ValueError("seed must be an integer or None")
    return np.random.default_rng(seed).poisson(
        lam=intensity * step, size=(n_steps, n_simulations)
    )


def _simulate_merton_core(
    s0: float,
    mu: float,
    sigma: float,
    lambda_param: float,
    mu_j: float,
    sigma_j: float,
    dt: float,
    n_steps: int,
    n_simulations: int,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate Merton paths and jump diagnostics; O(TM) time and memory."""
    initial = _finite_parameter(s0, "s0")
    drift = _finite_parameter(mu, "mu")
    volatility = _finite_parameter(sigma, "sigma")
    intensity = _finite_parameter(lambda_param, "lambda_param")
    jump_mean = _finite_parameter(mu_j, "mu_j")
    jump_volatility = _finite_parameter(sigma_j, "sigma_j")
    step = _validate_dimensions(n_steps, n_simulations, dt)
    if initial <= 0.0 or volatility < 0.0 or intensity < 0.0 or jump_volatility < 0.0:
        raise ValueError("s0 must be positive; sigma, lambda_param and sigma_j must be non-negative")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
        raise ValueError("seed must be an integer or None")

    rng = np.random.default_rng(seed)
    # Draw Brownian shocks first to exactly match simulate_gbm_paths under the
    # same seed when jump intensity is zero.
    brownian = rng.standard_normal((n_steps, n_simulations))
    arrivals = rng.poisson(intensity * step, size=(n_steps, n_simulations))
    jump_normals = rng.standard_normal((n_steps, n_simulations))

    # Individual gross jump multipliers are Y=exp(J), J~N(mu_j,sigma_j²).
    # Conditional sums of N normal log-jumps remain normal, enabling exact,
    # vectorized simulation without materializing a 3D jump array.
    log_jump_sums = arrivals * jump_mean + np.sqrt(arrivals) * jump_volatility * jump_normals
    jump_compensator = math.exp(jump_mean + 0.5 * jump_volatility**2) - 1.0
    log_increments = (
        (drift - intensity * jump_compensator - 0.5 * volatility**2) * step
        + volatility * math.sqrt(step) * brownian
        + log_jump_sums
    )
    log_paths = np.vstack((np.zeros(n_simulations), np.cumsum(log_increments, axis=0)))
    with np.errstate(over="ignore", invalid="ignore"):
        paths = initial * np.exp(log_paths)
    if not np.isfinite(paths).all():
        raise ValueError("simulation produced non-finite prices; check drift, jump size, and horizon")
    return paths, arrivals, log_jump_sums


def simulate_merton_jump_diffusion(
    s0: float,
    mu: float,
    sigma: float,
    lambda_param: float,
    mu_j: float,
    sigma_j: float,
    dt: float = 1 / 252,
    n_steps: int = 252,
    n_simulations: int = 10000,
    seed: int | None = None,
) -> np.ndarray:
    r"""Simulate exact Merton jump-diffusion price paths with lognormal jumps.

    The compensated price SDE is
    :math:`dS_t/S_t=(\mu-\lambda k)dt+\sigma dW_t+(Y_t-1)dN_t`,
    where :math:`k=E[Y-1]=e^{\mu_J+\sigma_J^2/2}-1`. Its exact log-price at
    :math:`t_n=n\Delta t` is
    :math:`\log(S_{t_n}/S_0)=(\mu-\lambda k-\sigma^2/2)t_n+
    \sigma W_{t_n}+\sum_{j=1}^{N_{t_n}}J_j`, with
    :math:`J_j=\log Y_j\sim N(\mu_J,\sigma_J^2)`. A €100 stock can follow
    smooth GBM paths in ordinary periods while an earnings surprise adds a
    sudden gap that pure GBM cannot represent. When :math:`\lambda=0`, paths
    exactly match pure GBM under the same seed and discretization. Complexity:
    :math:`O(n_{steps}n_{simulations})` time and output memory.
    """
    paths, _, _ = _simulate_merton_core(
        s0, mu, sigma, lambda_param, mu_j, sigma_j, dt, n_steps, n_simulations, seed
    )
    return paths


def _excess_kurtosis(values: np.ndarray) -> float:
    """Calculate population excess kurtosis in O(n) time and O(1) space."""
    centered = values - float(np.mean(values))
    variance = float(np.mean(centered**2))
    return 0.0 if variance <= np.finfo(float).eps else float(np.mean(centered**4) / variance**2 - 3.0)


def calculate_jump_impact_diagnostics(paths: np.ndarray, gbm_paths: np.ndarray) -> dict[str, Any]:
    r"""Compare jump paths with matched GBM paths to isolate jump-tail effects.

    Daily simple returns are :math:`r_t=S_t/S_{t-1}-1`; diagnostics report
    their excess kurtosis, the most negative paired return gap
    :math:`\min(r_t^{jump}-r_t^{GBM})`, and downside divergence between the
    terminal 5th-percentile outcomes. For a €100 stock, an earnings-gap path
    may show a more negative daily return and a lower 5th-percentile terminal
    price than smooth GBM. Complexity: :math:`O(TM)` time and :math:`O(TM)`
    temporary memory for :math:`T` dates and :math:`M` paths.
    """
    jump_paths = np.asarray(paths, dtype=float)
    diffusion_paths = np.asarray(gbm_paths, dtype=float)
    if (
        jump_paths.ndim != 2 or diffusion_paths.shape != jump_paths.shape
        or jump_paths.shape[0] < 2 or jump_paths.shape[1] < 1
    ):
        raise ValueError("paths and gbm_paths must have identical (n_steps+1, n_simulations) shapes")
    if (
        not np.isfinite(jump_paths).all() or not np.isfinite(diffusion_paths).all()
        or np.any(jump_paths <= 0.0) or np.any(diffusion_paths <= 0.0)
    ):
        raise ValueError("price paths must be finite and strictly positive")
    jump_returns = jump_paths[1:] / jump_paths[:-1] - 1.0
    gbm_returns = diffusion_paths[1:] / diffusion_paths[:-1] - 1.0
    return_gap = jump_returns - gbm_returns
    terminal_jump_return = jump_paths[-1] / jump_paths[0] - 1.0
    terminal_gbm_return = diffusion_paths[-1] / diffusion_paths[0] - 1.0
    jump_p5 = float(np.quantile(terminal_jump_return, 0.05))
    gbm_p5 = float(np.quantile(terminal_gbm_return, 0.05))
    return {
        "excess_kurtosis": _excess_kurtosis(jump_returns.ravel()),
        "gbm_excess_kurtosis": _excess_kurtosis(gbm_returns.ravel()),
        "excess_kurtosis_delta": float(
            _excess_kurtosis(jump_returns.ravel()) - _excess_kurtosis(gbm_returns.ravel())
        ),
        "max_negative_daily_return_gap": float(np.min(return_gap)),
        "jump_terminal_5th_percentile_return": jump_p5,
        "gbm_terminal_5th_percentile_return": gbm_p5,
        "tail_loss_divergence": float(gbm_p5 - jump_p5),
        "input_shape": [int(jump_paths.shape[0]), int(jump_paths.shape[1])],
    }


def jump_diffusion_telemetry(
    s0: float,
    mu: float,
    sigma: float,
    lambda_param: float,
    mu_j: float,
    sigma_j: float,
    dt: float = 1 / 252,
    n_steps: int = 252,
    n_simulations: int = 10000,
    seed: int | None = None,
) -> dict[str, Any]:
    r"""Expose simulation latency, realized jumps, mean jump size, and output shape.

    ``mean_jump_magnitude`` is the average absolute aggregate simple-return
    effect on intervals containing jumps; it is zero if no arrivals occur.
    Telemetry uses :math:`N\sim Poisson(\lambda\Delta t)` and reports total
    realized arrivals. A €100 stock's earnings gap therefore appears in both
    total jump count and the simulated path memory estimate. Complexity:
    :math:`O(TM)` time and output memory.
    """
    started = perf_counter()
    paths, arrivals, jump_log_sums = _simulate_merton_core(
        s0, mu, sigma, lambda_param, mu_j, sigma_j, dt, n_steps, n_simulations, seed
    )
    occurred = arrivals > 0
    if occurred.any():
        with np.errstate(over="ignore", invalid="ignore"):
            magnitudes = np.abs(np.expm1(jump_log_sums[occurred]))
        mean_magnitude = float(np.mean(magnitudes)) if np.isfinite(magnitudes).all() else math.inf
    else:
        mean_magnitude = 0.0
    return {
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "total_jumps_realized": int(np.sum(arrivals)),
        "mean_jump_magnitude": mean_magnitude,
        "input_shape": [int(paths.shape[0]), int(paths.shape[1])],
        "memory_bytes": int(paths.nbytes),
        "lambda_param": float(lambda_param),
        "mu_j": float(mu_j),
        "sigma_j": float(sigma_j),
        "n_steps": int(n_steps),
        "n_simulations": int(n_simulations),
        "seed": seed,
    }
