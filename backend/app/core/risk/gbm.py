"""Pure NumPy/Pandas simulation utilities for Geometric Brownian Motion."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any, Literal

import numpy as np

BarrierType = Literal["down_and_out", "up_and_out"]


def _validate_paths(paths: np.ndarray) -> np.ndarray:
    """Validate a finite non-negative time-by-simulation path matrix in O(TM)."""
    try:
        matrix = np.asarray(paths, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("paths must be a numeric array") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("paths must have shape (time_steps, simulations) with non-empty axes")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("paths must contain only finite non-negative prices")
    return matrix


def simulate_gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    dt: float = 1 / 252,
    n_steps: int = 252,
    n_simulations: int = 10000,
    seed: int | None = None,
) -> np.ndarray:
    r"""Simulate exact-discretization Geometric Brownian Motion price paths.

    GBM follows :math:`dS_t/S_t=\mu\,dt+\sigma\,dW_t`, with exact step
    :math:`S_{t+\Delta t}=S_t\exp((\mu-\frac12\sigma^2)\Delta t+
    \sigma\sqrt{\Delta t}Z_t)`, :math:`Z_t\sim N(0,1)`. The returned matrix
    has shape :math:`(n_{steps}+1,n_{simulations})`, including :math:`S_0`.
    For a €100 stock over 252 daily steps, each path accumulates a random
    sequence of log returns while its expected terminal price is
    :math:`100e^\mu`. Complexity: :math:`O(n_{steps}n_{simulations})` time
    and output memory.
    """
    try:
        initial_price, drift, volatility, step = map(float, (s0, mu, sigma, dt))
    except (TypeError, ValueError) as exc:
        raise ValueError("s0, mu, sigma, and dt must be finite numbers") from exc
    if not all(math.isfinite(v) for v in (initial_price, drift, volatility, step)):
        raise ValueError("s0, mu, sigma, and dt must be finite numbers")
    if initial_price <= 0.0 or volatility < 0.0 or step <= 0.0:
        raise ValueError("s0 and dt must be positive and sigma must be non-negative")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if isinstance(n_simulations, bool) or not isinstance(n_simulations, int) or n_simulations < 1:
        raise ValueError("n_simulations must be a positive integer")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
        raise ValueError("seed must be an integer or None")

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_steps, n_simulations))
    log_increments = (drift - 0.5 * volatility**2) * step + volatility * math.sqrt(step) * shocks
    log_paths = np.vstack((np.zeros(n_simulations), np.cumsum(log_increments, axis=0)))
    # Exponentiate in log space; overflow is mathematically possible for extreme
    # inputs, so reject it rather than returning invalid infinite price paths.
    with np.errstate(over="ignore", invalid="ignore"):
        paths = initial_price * np.exp(log_paths)
    if not np.isfinite(paths).all():
        raise ValueError("simulation produced non-finite prices; check drift/volatility and horizon")
    return paths


def calculate_path_terminal_stats(paths: np.ndarray) -> dict[str, float | int]:
    r"""Summarize simulated terminal prices and a 5th-percentile downside proxy.

    Given terminal prices :math:`S_T^{(1)},\ldots,S_T^{(M)}`, calculate their
    mean, sample standard deviation, median, 5th and 95th percentiles. The
    5th-percentile field is a terminal-price quantile; its loss-equivalent
    versus the initial price is :math:`S_0-Q_{0.05}(S_T)`. For 10,000 paths of
    a €100 stock over 252 days, these values describe the simulated terminal
    distribution rather than any one realized future. Complexity:
    :math:`O(M\log M)` time for quantiles and :math:`O(M)` extra space.
    """
    matrix = _validate_paths(paths)
    terminal = matrix[-1]
    p5, p95 = np.quantile(terminal, [0.05, 0.95])
    return {
        "n_simulations": int(terminal.size),
        "terminal_mean": float(np.mean(terminal)),
        "terminal_std": float(np.std(terminal, ddof=1)) if terminal.size > 1 else 0.0,
        "terminal_median": float(np.median(terminal)),
        "terminal_5th_percentile": float(p5),
        "terminal_95th_percentile": float(p95),
    }


def calculate_breach_probability(
    paths: np.ndarray, barrier_level: float, barrier_type: str = "down_and_out"
) -> float:
    r"""Estimate the fraction of simulated paths that touch a price barrier.

    For a down-and-out barrier, a path breaches if
    :math:`\min_{0\le t\le T}S_t\le B`; for up-and-out,
    :math:`\max_{0\le t\le T}S_t\ge B`. The initial price row is included.
    For example, with a €100 starting price and a €80 down barrier, the
    estimate is the share of paths that ever reach €80 or lower. Complexity:
    :math:`O(TM)` time and :math:`O(M)` auxiliary space.
    """
    matrix = _validate_paths(paths)
    try:
        barrier = float(barrier_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("barrier_level must be a finite positive number") from exc
    if not math.isfinite(barrier) or barrier <= 0.0:
        raise ValueError("barrier_level must be a finite positive number")
    if barrier_type not in {"down_and_out", "up_and_out"}:
        raise ValueError("barrier_type must be 'down_and_out' or 'up_and_out'")
    touched = (
        np.any(matrix <= barrier, axis=0)
        if barrier_type == "down_and_out"
        else np.any(matrix >= barrier, axis=0)
    )
    return float(np.mean(touched))


def gbm_telemetry(
    s0: float,
    mu: float,
    sigma: float,
    dt: float = 1 / 252,
    n_steps: int = 252,
    n_simulations: int = 10000,
    seed: int | None = None,
) -> dict[str, Any]:
    r"""Report simulation runtime, terminal mean, parameters, and memory use.

    The simulation generates exact lognormal transitions for a €100 stock;
    telemetry includes matrix shape and estimated NumPy buffer bytes. The
    expected terminal mean under :math:`dS/S=\mu dt+\sigma dW` is
    :math:`S_0e^{\mu n_{steps}dt}`. Complexity and memory are both
    :math:`O(n_{steps}n_{simulations})`.
    """
    started = perf_counter()
    paths = simulate_gbm_paths(s0, mu, sigma, dt, n_steps, n_simulations, seed)
    stats = calculate_path_terminal_stats(paths)
    return {
        "latency_ms": float((perf_counter() - started) * 1000.0),
        "input_shape": [int(n_steps + 1), int(n_simulations)],
        "terminal_mean": stats["terminal_mean"],
        "s0": float(s0),
        "mu": float(mu),
        "sigma": float(sigma),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "n_simulations": int(n_simulations),
        "seed": seed,
        "memory_bytes": int(paths.nbytes),
    }
