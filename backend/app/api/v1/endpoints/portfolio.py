"""Markowitz and risk-parity portfolio optimization endpoints."""

import logging
from uuid import UUID
from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException
import numpy as np
import pandas as pd

from app.core.portfolio.optimizer import generate_efficient_frontier, optimize_maximum_sharpe, optimize_minimum_variance
from app.core.portfolio.risk_parity import optimize_risk_parity
from app.schemas.portfolio import (
    PortfolioOptimizationRequest,
    PortfolioOptimizationResponse,
    RiskParityRequest,
    RiskParityResponse,
)
from app.api.v1.endpoints._utils import envelope, run_core
from app.api.dependencies import AuthenticatedUser, require_current_user
from app.db.supabase import get_async_supabase_client
from app.models.portfolio import PortfolioConfig
from app.repositories.portfolios import PortfolioRepository
from app.schemas.advisor import PortfolioConfigCreate
from app.services.data_pipeline_orchestrator import fetch_and_cache_eod_prices

router = APIRouter()
portfolio_collection_router = APIRouter()
logger = logging.getLogger(__name__)


@portfolio_collection_router.get("")
async def list_saved_portfolios(
    limit: int = 100,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Frontend collection route for portfolios owned by the authenticated user."""
    started = perf_counter()
    client = await get_async_supabase_client()
    try:
        items = await PortfolioRepository(client).list_for_owner(user.id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return envelope(
        [item.model_dump(mode="json") for item in items],
        (perf_counter() - started) * 1000.0,
        {"portfolios": len(items)},
    )


@router.post("/configs")
async def create_portfolio_config(
    request: PortfolioConfigCreate,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Persist a portfolio configuration under the verified user identity."""
    started = perf_counter()
    client = await get_async_supabase_client()
    portfolio = PortfolioConfig(owner_id=user.id, **request.model_dump())
    saved = await PortfolioRepository(client).save(portfolio)
    refreshes = await _refresh_new_tickers(saved.tickers)
    data = saved.model_dump(mode="json")
    data["price_refreshes"] = refreshes
    return envelope(data, (perf_counter() - started) * 1000.0, {"tickers": [len(saved.tickers)]}, saved.updated_at)


@router.get("/configs")
async def list_portfolio_configs(
    limit: int = 100,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """List only portfolio configurations owned by the verified user."""
    started = perf_counter()
    client = await get_async_supabase_client()
    try:
        items = await PortfolioRepository(client).list_for_owner(user.id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return envelope([item.model_dump(mode="json") for item in items], (perf_counter() - started) * 1000.0, {"portfolios": len(items)})


@router.get("/configs/{portfolio_id}")
async def get_portfolio_config(
    portfolio_id: UUID,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Retrieve one portfolio only when it belongs to the verified user."""
    started = perf_counter()
    client = await get_async_supabase_client()
    item = await PortfolioRepository(client).get_for_owner(portfolio_id, user.id)
    if item is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return envelope(item.model_dump(mode="json"), (perf_counter() - started) * 1000.0, {"tickers": [len(item.tickers)]}, item.updated_at)


@router.put("/configs/{portfolio_id}")
async def update_portfolio_config(
    portfolio_id: UUID,
    request: PortfolioConfigCreate,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Update the name, ticker set, weights, and settings of an owned portfolio."""
    started = perf_counter()
    client = await get_async_supabase_client()
    repository = PortfolioRepository(client)
    existing = await repository.get_for_owner(portfolio_id, user.id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    updated = existing.model_copy(update=request.model_dump())
    saved = await repository.save(updated)
    added_tickers = sorted(set(saved.tickers) - set(existing.tickers))
    refreshes = await _refresh_new_tickers(added_tickers)
    data = saved.model_dump(mode="json")
    data["price_refreshes"] = refreshes
    return envelope(
        data,
        (perf_counter() - started) * 1000.0,
        {"tickers": [len(saved.tickers)]},
        saved.updated_at,
    )


@router.delete("/configs/{portfolio_id}")
async def delete_portfolio_config(
    portfolio_id: UUID,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Delete an owned portfolio without exposing another user's record."""
    started = perf_counter()
    client = await get_async_supabase_client()
    deleted = await PortfolioRepository(client).delete_for_owner(portfolio_id, user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return envelope({"deleted": True, "portfolio_id": str(portfolio_id)}, (perf_counter() - started) * 1000.0, {})


def _covariance(tickers: list[str], values: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(values, index=tickers, columns=tickers, dtype=float)


async def _refresh_new_tickers(tickers: list[str]) -> list[dict]:
    """Attempt immediate EOD caching for configured symbols without blocking persistence."""
    outcomes = []
    for ticker in dict.fromkeys(str(value).strip().upper() for value in tickers):
        try:
            outcomes.append(await fetch_and_cache_eod_prices(ticker))
        except Exception as exc:
            logger.info("EOD refresh for newly configured %s remains queued (%s)", ticker, type(exc).__name__)
            outcomes.append({"ticker": ticker, "status": "queued", "error": type(exc).__name__})
    return outcomes


@router.post("/markowitz/max-sharpe", response_model=PortfolioOptimizationResponse)
def markowitz_max_sharpe(request: PortfolioOptimizationRequest) -> dict:
    """Optimize long-only asset weights for maximum Sharpe ratio."""
    covariance = _covariance(request.tickers, request.covariance_matrix)
    expected = pd.Series(request.expected_returns, index=request.tickers, dtype=float)
    result, elapsed = run_core(lambda: optimize_maximum_sharpe(expected, covariance, request.risk_free_rate))
    return envelope(result, elapsed, {"expected_returns": [len(request.tickers)], "covariance_matrix": [len(request.tickers), len(request.tickers)]})


@router.post("/markowitz/min-variance", response_model=PortfolioOptimizationResponse)
def markowitz_min_variance(request: PortfolioOptimizationRequest) -> dict:
    """Find the long-only minimum-variance allocation."""
    covariance = _covariance(request.tickers, request.covariance_matrix)
    result, elapsed = run_core(lambda: optimize_minimum_variance(covariance))
    return envelope(result, elapsed, {"covariance_matrix": [len(request.tickers), len(request.tickers)]})


@router.post("/markowitz/efficient-frontier", response_model=PortfolioOptimizationResponse)
def markowitz_frontier(request: PortfolioOptimizationRequest) -> dict:
    """Generate feasible efficient-frontier portfolios over target returns."""
    covariance = _covariance(request.tickers, request.covariance_matrix)
    expected = pd.Series(request.expected_returns, index=request.tickers, dtype=float)
    result, elapsed = run_core(lambda: generate_efficient_frontier(expected, covariance, request.points))
    return envelope({"frontier": result}, elapsed, {"expected_returns": [len(request.tickers)], "covariance_matrix": [len(request.tickers), len(request.tickers)]})


@router.post("/optimize", response_model=PortfolioOptimizationResponse)
def optimize(request: PortfolioOptimizationRequest) -> dict:
    """Frontend-compatible alias for the Markowitz efficient-frontier handler.

    The payload and response are identical to ``/markowitz/efficient-frontier``;
    the route delegates directly to that existing adapter and core optimizer.
    """
    return markowitz_frontier(request)


@router.post("/risk-parity", response_model=RiskParityResponse)
def risk_parity(request: RiskParityRequest) -> dict:
    """Compute long-only equal or budgeted risk-parity weights."""
    covariance = _covariance(request.tickers, request.covariance_matrix)
    budget = None if request.target_risk_budget is None else np.asarray(request.target_risk_budget, dtype=float)
    result, elapsed = run_core(lambda: optimize_risk_parity(covariance, budget))
    return envelope(result, elapsed, {"covariance_matrix": [len(request.tickers), len(request.tickers)]})
