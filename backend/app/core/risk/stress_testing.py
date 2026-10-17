"""Pure portfolio stress replay, macro shock, and reverse-stress calculations."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _weights(weights: np.ndarray, n_assets: int) -> np.ndarray:
    """Validate finite asset weights with matching dimension; O(A)."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size != n_assets:
        raise ValueError("weights must be a one-dimensional vector matching the asset dimension")
    if not np.isfinite(values).all():
        raise ValueError("weights must contain only finite values")
    return values


def _loadings(factor_loadings: pd.DataFrame, n_assets: int) -> np.ndarray:
    """Validate an assets-by-factors numeric loading matrix; O(AK)."""
    if not isinstance(factor_loadings, pd.DataFrame):
        raise TypeError("factor_loadings must be a pandas DataFrame")
    matrix = factor_loadings.to_numpy(dtype=float, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != n_assets or matrix.shape[1] == 0:
        raise ValueError("factor_loadings rows must match assets and include at least one factor")
    if not np.isfinite(matrix).all():
        raise ValueError("factor_loadings must contain only finite values")
    return matrix


def replay_historical_crisis(
    weights: np.ndarray,
    asset_returns: pd.DataFrame,
    scenario_start: str,
    scenario_end: str,
) -> dict[str, Any]:
    r"""Replay portfolio returns over an inclusive historical crisis date window.

    Daily portfolio return is :math:`r_{p,t}=w^Tr_t`; wealth compounds as
    :math:`W_t=\prod_{s\le t}(1+r_{p,s})`, and drawdown is
    :math:`D_t=W_t/\max_{s\le t}W_s-1`. The returned arithmetic asset
    contributions use :math:`w_i r_{i,t}` over the worst peak-to-trough
    segment. The ``asset_contributions`` field contains arithmetic
    asset-level return contributions :math:`w_i r_i` summed over the
    peak-to-trough segment; it is not compounded factor-level attribution.
    A €1,000,000 portfolio replayed through 2008 or March 2020 can reveal both
    the peak-to-trough loss path and which holdings drove it.
    Complexity: :math:`O(TA)` time and space.
    """
    started = perf_counter()
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a pandas DataFrame")
    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise TypeError("asset_returns must use a DatetimeIndex")
    if asset_returns.empty or asset_returns.shape[1] == 0:
        raise ValueError("asset_returns must contain dates and asset columns")
    w = _weights(weights, asset_returns.shape[1])
    values = asset_returns.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all() or np.any(values < -1.0):
        raise ValueError("asset returns must be finite simple returns greater than or equal to -100%")
    try:
        start, end = pd.Timestamp(scenario_start), pd.Timestamp(scenario_end)
    except (TypeError, ValueError) as exc:
        raise ValueError("scenario_start and scenario_end must be parseable dates") from exc
    if start > end:
        raise ValueError("scenario_start must be on or before scenario_end")
    mask = (asset_returns.index >= start) & (asset_returns.index <= end)
    scenario = asset_returns.loc[mask]
    if scenario.empty:
        raise ValueError("no asset return observations fall inside the scenario window")

    scenario_values = scenario.to_numpy(dtype=float)
    portfolio_returns = scenario_values @ w
    if np.any(portfolio_returns < -1.0):
        raise ValueError("weighted portfolio return below -100% is not a valid simple return")
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + portfolio_returns)))
    peaks = np.maximum.accumulate(wealth)
    drawdowns = wealth / peaks - 1.0
    trough_position = int(np.argmin(drawdowns))
    worst_drawdown = float(drawdowns[trough_position])
    peak_position = int(np.argmax(wealth[:trough_position + 1]))
    index = scenario.index

    def date_label(position: int) -> pd.Timestamp:
        return index[0] if position == 0 else index[position - 1]

    segment_returns = scenario_values[peak_position:trough_position]
    if trough_position == 0 or trough_position <= peak_position:
        segment_returns = np.empty((0, scenario.shape[1]))
    driver_values = (segment_returns * w[None, :]).sum(axis=0) if segment_returns.size else np.zeros(w.size)
    drivers = pd.Series(driver_values, index=scenario.columns, name="peak_to_trough_asset_driver")
    drawdown_series = pd.Series(drawdowns[1:], index=index, name="drawdown")
    portfolio_series = pd.Series(portfolio_returns, index=index, name="portfolio_return")
    return {
        "scenario_start": index[0],
        "scenario_end": index[-1],
        "n_observations": int(len(scenario)),
        "cumulative_return": float(wealth[-1] - 1.0),
        "worst_drawdown": worst_drawdown,
        "peak_date": date_label(peak_position),
        "trough_date": date_label(trough_position),
        "drawdown_path": drawdown_series,
        "portfolio_return_path": portfolio_series,
        "asset_contributions": drivers,
        "input_shape": [int(scenario.shape[0]), int(scenario.shape[1])],
        "latency_ms": float((perf_counter() - started) * 1000.0),
    }


def run_reverse_stress_test(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    factor_loadings: pd.DataFrame,
    target_loss_threshold: float = -0.30,
) -> dict[str, Any]:
    r"""Find the smallest Euclidean factor shock that reaches a portfolio loss.

    Portfolio factor exposure is :math:`e^T=w^TB`; reverse stress solves
    :math:`\min_{\delta_f}\frac12\|\delta_f\|_2^2` subject to
    :math:`w^TB\delta_f=L_{target}`. The analytical minimum-norm solution is
    :math:`\delta_f=L_{target}e/(e^Te)`; SLSQP solves the constrained problem
    and the result is reconciled against this closed form. For example, the
    output is the smallest joint growth/rate shock needed to produce a -30%
    portfolio loss. Complexity: :math:`O(A^3+AK^2+K^3)` time and
    :math:`O(AK+K^2)` space for :math:`A` assets and :math:`K` factors.
    """
    started = perf_counter()
    if isinstance(cov_matrix, pd.DataFrame):
        covariance = cov_matrix.to_numpy(dtype=float, copy=True)
    else:
        covariance = np.asarray(cov_matrix, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("cov_matrix must be a square matrix")
    w = _weights(weights, covariance.shape[0])
    if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, atol=1e-10):
        raise ValueError("cov_matrix must be finite and symmetric")
    if float(np.linalg.eigvalsh(covariance).min()) < -1e-8:
        raise ValueError("cov_matrix must be positive semidefinite")
    loadings = _loadings(factor_loadings, w.size)
    target = float(target_loss_threshold)
    if not math.isfinite(target) or target >= 0.0:
        raise ValueError("target_loss_threshold must be a finite negative return")
    exposure = w @ loadings
    exposure_norm_sq = float(exposure @ exposure)
    if exposure_norm_sq <= np.finfo(float).eps:
        raise ValueError("portfolio has zero factor exposure; target loss is infeasible")
    analytical = target * exposure / exposure_norm_sq

    objective = lambda shock: 0.5 * float(shock @ shock)
    constraint = {"type": "eq", "fun": lambda shock: float(exposure @ shock - target)}
    optimized = minimize(
        objective,
        x0=analytical,
        method="SLSQP",
        constraints=[constraint],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    shock = np.asarray(optimized.x, dtype=float)
    if not optimized.success or not np.isfinite(shock).all() or not np.isclose(exposure @ shock, target, atol=1e-8):
        # The closed form is the exact constrained optimum; this numerical
        # fallback preserves exact target-loss behavior if SLSQP reports a
        # precision-only termination.
        shock = analytical
    achieved_loss = float(exposure @ shock)
    return {
        "factor_shocks": pd.Series(shock, index=factor_loadings.columns, name="shock"),
        "portfolio_factor_exposure": pd.Series(exposure, index=factor_loadings.columns, name="exposure"),
        "target_loss_threshold": target,
        "achieved_portfolio_loss": achieved_loss,
        "minimum_shock_magnitude": float(np.linalg.norm(shock)),
        "analytical_minimum_shock": pd.Series(analytical, index=factor_loadings.columns),
        "optimization_success": bool(optimized.success),
        "optimization_status": int(optimized.status),
        "optimization_message": str(optimized.message),
        "input_shape": [int(w.size), int(loadings.shape[1])],
        "latency_ms": float((perf_counter() - started) * 1000.0),
    }


def apply_macro_scenario_shock(
    weights: np.ndarray,
    factor_loadings: pd.DataFrame,
    macro_shocks: dict[str, float],
) -> float:
    r"""Estimate immediate linear portfolio return under named factor shocks.

    With asset sensitivities :math:`B` and factor shock vector :math:`\Delta f`,
    the first-order portfolio impact is :math:`\Delta R_p=w^TB\Delta f`.
    For example, a +200 bp rates shock is encoded as ``0.02`` when the rate
    loading is return per unit rate, while a -15% growth shock is ``-0.15``.
    A €1,000,000 portfolio with impact -3% has an estimated immediate loss of
    €30,000. Complexity: :math:`O(AK)` time and :math:`O(K)` space.
    """
    if not isinstance(macro_shocks, dict):
        raise TypeError("macro_shocks must be a dictionary mapping factor names to shocks")
    loadings = _loadings(factor_loadings, len(weights))
    w = _weights(weights, loadings.shape[0])
    unknown = set(macro_shocks) - set(factor_loadings.columns)
    if unknown:
        raise ValueError(f"macro_shocks contains unknown factors: {sorted(unknown)}")
    shock = np.zeros(loadings.shape[1], dtype=float)
    for position, factor in enumerate(factor_loadings.columns):
        if factor in macro_shocks:
            shock[position] = float(macro_shocks[factor])
    if not np.isfinite(shock).all():
        raise ValueError("macro shocks must be finite")
    return float(w @ loadings @ shock)


def stress_testing_telemetry(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    factor_loadings: pd.DataFrame,
    target_loss_threshold: float = -0.30,
) -> dict[str, Any]:
    r"""Report reverse-stress runtime, minimum shock, and matrix input shape.

    It exposes the minimum factor-shock magnitude needed for a target loss
    :math:`L` under :math:`w^TB\delta_f=L`. Historical replay separately
    reports worst drawdown. Complexity: :math:`O(AK^2+K^3)` time.
    """
    return run_reverse_stress_test(weights, cov_matrix, factor_loadings, target_loss_threshold)
