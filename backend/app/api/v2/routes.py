"""Version 2 HTTP adapters for the four MoatCheck Quant personas.

These endpoints accept explicit research inputs and do not read or write
Supabase. A later data-service layer can supply the same contract payloads.
"""

from __future__ import annotations

import importlib.util
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from app.api.v2.schemas import (
    BenchmarkRequest, ContributionRequest, FactorScoreRequest, OptimizeRequest,
    RebalanceRequest, RiskRequest, StressRequest, WalkForwardRequest,
)
from app.core.portfolio.optimizer import optimize_portfolio
from app.core.risk.metrics import marginal_contribution_to_risk, portfolio_var_cvar, stress_test
from app.core.backtest.walk_forward import run_walk_forward
from app.core.research.factors import combine_factor_scores
from app.core.trading.rebalance import build_rebalance_orders

router = APIRouter()


def _frame(values: list[list[float]], assets: list[str]) -> pd.DataFrame:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != len(assets):
        raise HTTPException(422, "matrix columns must match the supplied asset list")
    return pd.DataFrame(array, columns=assets)


def _core_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, "Invalid input or calculation parameters.") from exc


# Researcher persona ---------------------------------------------------------

@router.post("/researcher/factor-score")
def factor_score(body: FactorScoreRequest) -> dict[str, Any]:
    """Combine supplied factor scores after normalizing configured weights."""
    return _core_call(combine_factor_scores, body.factor_scores, body.factor_weights)


@router.post("/researcher/optimize")
def researcher_optimize(body: OptimizeRequest) -> dict[str, Any]:
    returns = _frame(body.returns, body.assets)
    result = _core_call(
        optimize_portfolio, returns, objective=body.objective,
        periods_per_year=body.periods_per_year, risk_free_rate=body.risk_free_rate,
        target_return=body.target_return, target_volatility=body.target_volatility,
        lower_bounds=body.lower_bounds if body.lower_bounds is not None else 0.0,
        upper_bounds=body.upper_bounds if body.upper_bounds is not None else 1.0,
        factor_exposures=body.factor_exposures, factor_tilts=body.factor_tilts,
        linear_exposures=body.linear_exposures,
        linear_lower_bounds=body.linear_lower_bounds,
        linear_upper_bounds=body.linear_upper_bounds,
    )
    return result.to_dict()


@router.post("/researcher/walk-forward")
def walk_forward(body: WalkForwardRequest) -> dict[str, Any]:
    """Rank assets using scores from the prior period and compound with costs."""
    return _core_call(run_walk_forward, _frame(body.returns, body.assets),
                      _frame(body.scores, body.assets), assets=body.assets,
                      lookback_periods=body.lookback_periods, rebalance_every=body.rebalance_every,
                      top_n=body.top_n, transaction_cost_bps=body.transaction_cost_bps)


# Quant developer persona ----------------------------------------------------

@router.post("/quant_dev/benchmark")
def benchmark_optimizers(body: BenchmarkRequest) -> dict[str, Any]:
    """Compare SciPy SLSQP with a vectorized random-portfolio NumPy baseline."""
    frame = _frame(body.returns, body.assets)
    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise HTTPException(422, "returns must be finite")
    mu = values.mean(axis=0) * body.periods_per_year
    cov = np.atleast_2d(np.cov(values, rowvar=False)) * body.periods_per_year
    tracemalloc.start()
    t0 = time.perf_counter()
    numpy_weights = np.random.default_rng(7).dirichlet(np.ones(len(body.assets)), size=body.iterations)
    numpy_vol = np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", numpy_weights, cov, numpy_weights), 0))
    numpy_sharpe = (numpy_weights @ mu) / np.maximum(numpy_vol, 1e-15)
    numpy_best = int(np.argmax(numpy_sharpe))
    numpy_seconds = time.perf_counter() - t0
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t0 = time.perf_counter()
    scipy_result = _core_call(optimize_portfolio, frame, objective="max_sharpe", periods_per_year=body.periods_per_year)
    scipy_seconds = time.perf_counter() - t0
    return {
        "benchmark": {
            "numpy_vectorized": {"seconds": numpy_seconds, "iterations": body.iterations,
                                 "best_sharpe": float(numpy_sharpe[numpy_best]), "weights": numpy_weights[numpy_best].tolist()},
            "scipy_slsqp": {"seconds": scipy_seconds, "best_sharpe": scipy_result.sharpe_ratio,
                            "weights": scipy_result.weights},
        },
        "python_tracemalloc": {"current_bytes": current_bytes, "peak_bytes": peak_bytes},
        "optional_backends": {name: importlib.util.find_spec(name) is not None for name in ("cvxpy", "polars")},
        "note": "Microbenchmark uses synthetic candidate weights; timings vary by host and are not route-wide production profiling.",
    }


@router.get("/quant_dev/capabilities")
def quant_dev_capabilities() -> dict[str, Any]:
    """Report installed numerical backends and current-process profiling scope."""
    return {"backends": {name: importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "cvxpy", "polars")},
            "profiling": ["per-operation wall time", "Python allocation peak via tracemalloc"],
            "note": "No process-wide or database query profiler is enabled by this API."}


# Trader persona -------------------------------------------------------------

@router.post("/trader/rebalance")
def rebalance_orders(body: RebalanceRequest) -> dict[str, Any]:
    return _core_call(build_rebalance_orders, body.current_weights, body.target_weights, body.prices,
                      portfolio_value=body.portfolio_value, average_daily_volume=body.average_daily_volume,
                      volatility=body.volatility, transaction_cost_bps=body.transaction_cost_bps,
                      integer_shares=body.integer_shares)


# Risk persona ---------------------------------------------------------------

@router.post("/risk/var-cvar")
def var_cvar(body: RiskRequest) -> dict[str, Any]:
    return _core_call(portfolio_var_cvar, _frame(body.returns, body.assets), body.weights,
                      confidence=body.confidence, method=body.method)


@router.post("/risk/stress-test")
def risk_stress(body: StressRequest) -> dict[str, Any]:
    if len(body.dates) != len(body.returns):
        raise HTTPException(422, "dates length must equal number of return observations")
    frame = _frame(body.returns, body.assets)
    frame.index = pd.to_datetime(body.dates)
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    return _core_call(stress_test, frame, body.weights, body.scenarios)


@router.post("/risk/contribution")
def risk_contribution(body: ContributionRequest) -> dict[str, Any]:
    return _core_call(marginal_contribution_to_risk, _frame(body.returns, body.assets), body.weights,
                      periods_per_year=body.periods_per_year)
