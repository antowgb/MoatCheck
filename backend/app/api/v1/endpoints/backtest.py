"""Execution friction and rebalance simulation endpoints."""

from datetime import datetime, timezone
from time import perf_counter
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
import numpy as np

from app.core.trading.execution_costs import calculate_total_execution_drag
from app.core.trading.rebalance import RebalancingEngine, build_rebalance_orders
from app.schemas.backtest import (
    BacktestResponse,
    ExecutionCostsRequest,
    RebalanceOrdersRequest,
    RebalanceSimulationRequest,
)
from app.api.v1.endpoints._utils import envelope, run_core
from app.api.dependencies import AuthenticatedUser, require_current_user
from app.db.supabase import get_async_supabase_client
from app.models.backtest_run import BacktestRun
from app.repositories.backtests import BacktestRepository
from app.repositories.portfolios import PortfolioRepository
from app.schemas.backtest import StoredPortfolioBacktestRequest

router = APIRouter()


@router.post("/execution-costs", response_model=BacktestResponse)
def execution_costs(request: ExecutionCostsRequest) -> dict:
    """Estimate spread, square-root impact, and broker fees for a batch of orders."""
    result, elapsed = run_core(lambda: calculate_total_execution_drag(
        np.asarray(request.trade_sizes), np.asarray(request.prices), np.asarray(request.advs),
        np.asarray(request.volatilities), np.asarray(request.spreads),
        fixed_fee_per_trade=request.fixed_fee_per_trade, pct_broker_fee=request.pct_broker_fee,
    ))
    n = len(request.trade_sizes)
    return envelope(result, elapsed, {"orders": [n]})


@router.post("/rebalance-orders", response_model=BacktestResponse)
def rebalance_orders(request: RebalanceOrdersRequest) -> dict:
    """Build target-weight orders and estimate their friction without routing them."""
    result, elapsed = run_core(lambda: build_rebalance_orders(
        request.current_weights, request.target_weights, request.prices,
        portfolio_value=request.portfolio_value,
        average_daily_volume=request.average_daily_volume,
        volatility=request.volatility,
        transaction_cost_bps=request.transaction_cost_bps,
        integer_shares=request.integer_shares,
    ))
    return envelope(result, elapsed, {"assets": len(set(request.current_weights) | set(request.target_weights))})


@router.post("/rebalance-simulation", response_model=BacktestResponse)
def rebalance_simulation(request: RebalanceSimulationRequest) -> dict:
    """Run a chronological multi-period wealth and transaction-cost simulation."""
    engine = RebalancingEngine(
        np.asarray(request.initial_weights), np.asarray(request.target_weights), np.asarray(request.prices),
        initial_wealth=request.initial_wealth,
        execution_cost_bps=request.execution_cost_bps,
        max_turnover=request.max_turnover,
        periods_per_year=request.periods_per_year,
    )
    result, elapsed = run_core(engine.run)
    return envelope(result, elapsed, {
        "initial_weights": [len(request.initial_weights)],
        "target_weights": [len(request.target_weights), len(request.tickers)],
        "prices": [len(request.prices), len(request.tickers)],
    })


@router.post("/run", response_model=BacktestResponse)
def run_backtest(request: RebalanceSimulationRequest) -> dict:
    """Frontend-compatible alias for the chronological rebalance simulation."""
    return rebalance_simulation(request)


@router.post("/portfolios/{portfolio_id}/runs")
async def run_stored_portfolio_backtest(
    portfolio_id: UUID,
    request: StoredPortfolioBacktestRequest,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Backtest the saved strategic weights and persist an owner-linked tear sheet.

    The saved portfolio is loaded and ownership-checked before calculation.
    The submitted price panel must follow the stored ticker ordering.
    """
    started = perf_counter()
    client = await get_async_supabase_client()
    portfolio = await PortfolioRepository(client).get_for_owner(portfolio_id, user.id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    prices = np.asarray(request.prices, dtype=float)
    if prices.shape[1] != len(portfolio.tickers):
        raise HTTPException(status_code=422, detail="price columns must match the saved portfolio tickers")
    initial = np.asarray([portfolio.target_weights[ticker] for ticker in portfolio.tickers], dtype=float)
    targets = np.repeat(initial[None, :], len(request.dates) - 1, axis=0)
    engine = RebalancingEngine(
        initial,
        targets,
        prices,
        initial_wealth=request.initial_wealth,
        execution_cost_bps=request.execution_cost_bps,
        max_turnover=request.max_turnover,
        periods_per_year=252,
    )
    result, elapsed = run_core(engine.run)
    wealth = result["wealth_path"]
    net_pnl = (wealth.iloc[1:].to_numpy(dtype=float) - request.initial_wealth).tolist()
    dates = request.dates[1:]
    running_peak = np.maximum.accumulate(wealth.to_numpy(dtype=float))
    drawdown = wealth.to_numpy(dtype=float) / running_peak - 1.0
    total_return = float(wealth.iloc[-1] / request.initial_wealth - 1.0)
    record = BacktestRun(
        portfolio_id=portfolio.id,
        strategy_name=f"Stored allocation: {portfolio.name}",
        start_date=dates[0],
        end_date=dates[-1],
        tickers=portfolio.tickers,
        net_pnl_series=net_pnl,
        observation_dates=dates,
        turnover_series=result["turnover_series"].tolist(),
        total_turnover=result["total_turnover"],
        average_annual_turnover=result["average_annual_turnover"],
        total_transaction_cost=result["total_transaction_cost"],
        total_transaction_drag_bps=result["total_transaction_drag_bps"],
        execution_cost_bps_of_traded_notional=result["execution_cost_bps_of_traded_notional"],
        metrics={"net_total_return": total_return, "max_drawdown": float(np.min(drawdown))},
        telemetry={"execution_time_ms": elapsed, "input_shape": {"prices": list(prices.shape)}},
    )
    saved = await BacktestRepository(client).save_run(record)
    return envelope(saved.model_dump(mode="json"), (perf_counter() - started) * 1000.0, {"prices": list(prices.shape)}, datetime.now(timezone.utc))


@router.get("/portfolios/{portfolio_id}/runs")
async def list_stored_portfolio_backtests(
    portfolio_id: UUID,
    limit: int = 100,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Retrieve backtest tear sheets only after verifying portfolio ownership."""
    started = perf_counter()
    client = await get_async_supabase_client()
    portfolio = await PortfolioRepository(client).get_for_owner(portfolio_id, user.id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    try:
        items = await BacktestRepository(client).list_for_portfolio(portfolio_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return envelope([item.model_dump(mode="json") for item in items], (perf_counter() - started) * 1000.0, {"runs": len(items)})
