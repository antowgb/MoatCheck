"""Portfolio VaR, liquidity VaR, and stress-test endpoints."""

from math import sqrt
from datetime import datetime, timezone
from time import perf_counter
from uuid import UUID

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.core.risk.metrics import portfolio_var_cvar
from app.core.risk.crisis_scenarios import list_crisis_scenarios, load_crisis_scenario
from app.core.risk.stress_testing import apply_macro_scenario_shock, replay_historical_crisis, run_reverse_stress_test
from app.core.trading.l_var import calculate_liquidity_adjusted_var
from app.schemas.risk import (
    HistoricalStressRequest,
    HistoricalScenarioReplayRequest,
    LiquidityVaRRequest,
    MacroShockRequest,
    PortfolioReturnsRequest,
    ReverseStressRequest,
    RiskResponse,
)
from app.api.v1.endpoints._utils import envelope, json_safe, run_core
from app.api.dependencies import AuthenticatedUser, require_current_user
from app.db.supabase import get_async_supabase_client
from app.models.stress_test import StressTest
from app.repositories.portfolios import PortfolioRepository
from app.repositories.stress_tests import StressTestRepository

router = APIRouter()


@router.post("/var", response_model=RiskResponse)
def calculate_var(request: PortfolioReturnsRequest) -> dict:
    """Calculate portfolio VaR and expected shortfall as positive loss fractions."""
    returns = pd.DataFrame(request.returns, columns=request.tickers, dtype=float)
    result, elapsed = run_core(lambda: portfolio_var_cvar(returns, request.weights, confidence=request.confidence_level, method=request.method))
    return envelope(result, elapsed, {"returns": [len(request.returns), len(request.tickers)], "weights": [len(request.weights)]})


@router.post("/evaluate", response_model=RiskResponse)
def evaluate_risk(request: PortfolioReturnsRequest) -> dict:
    """Frontend-compatible alias for the existing portfolio VaR/CVaR handler."""
    return calculate_var(request)


@router.post("/liquidity-adjusted-var", response_model=RiskResponse)
def liquidity_adjusted_var(request: LiquidityVaRRequest) -> dict:
    """Calculate VaR scaled to liquidation horizon plus spread and impact costs."""
    returns = pd.Series(request.portfolio_returns, dtype=float)
    result, elapsed = run_core(lambda: calculate_liquidity_adjusted_var(
        returns, np.asarray(request.position_values), np.asarray(request.advs),
        np.asarray(request.volatilities), np.asarray(request.spreads),
        confidence_level=request.confidence_level,
        max_participation_rate=request.max_participation_rate,
        method=request.method,
    ))
    return envelope(result, elapsed, {"portfolio_returns": [len(request.portfolio_returns)], "positions": [len(request.tickers)]})


@router.post("/stress/historical", response_model=RiskResponse)
def historical_stress(request: HistoricalStressRequest) -> dict:
    """Replay a date-bounded historical crisis against supplied portfolio weights."""
    returns = pd.DataFrame(request.asset_returns, index=pd.to_datetime(request.dates), columns=request.tickers, dtype=float)
    result, elapsed = run_core(lambda: replay_historical_crisis(
        np.asarray(request.weights), returns, request.scenario_start.isoformat(), request.scenario_end.isoformat()
    ))
    return envelope(result, elapsed, {"asset_returns": [len(request.dates), len(request.tickers)], "weights": [len(request.weights)]})


@router.post("/portfolios/{portfolio_id}/stress/historical")
async def persist_portfolio_historical_stress(
    portfolio_id: UUID,
    request: HistoricalStressRequest,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Replay and persist historical stress for the saved owner portfolio.

    Ticker ordering and weights must match the stored configuration so callers
    cannot substitute a different allocation under an owned portfolio ID.
    """
    started = perf_counter()
    client = await get_async_supabase_client()
    portfolio = await PortfolioRepository(client).get_for_owner(portfolio_id, user.id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if request.tickers != portfolio.tickers or any(
        abs(request.weights[index] - portfolio.target_weights[ticker]) > 1e-6
        for index, ticker in enumerate(portfolio.tickers)
    ):
        raise HTTPException(status_code=422, detail="stress inputs must use saved portfolio tickers and weights")
    returns = pd.DataFrame(request.asset_returns, index=pd.to_datetime(request.dates), columns=request.tickers, dtype=float)
    result, elapsed = run_core(lambda: replay_historical_crisis(
        np.asarray(request.weights), returns, request.scenario_start.isoformat(), request.scenario_end.isoformat()
    ))
    stored = StressTest(
        portfolio_id=portfolio.id,
        stress_type="historical_crisis",
        scenario_name=f"Historical crisis {request.scenario_start} to {request.scenario_end}",
        scenario_start=request.scenario_start,
        scenario_end=request.scenario_end,
        inputs={"tickers": request.tickers, "weights": request.weights, "n_observations": len(request.dates)},
        results=json_safe(result),
        portfolio_return=float(result["cumulative_return"]),
        worst_drawdown=float(result["worst_drawdown"]),
        telemetry={"execution_time_ms": elapsed, "input_shape": result["input_shape"]},
    )
    saved = await StressTestRepository(client).save(stored)
    return envelope(saved.model_dump(mode="json"), (perf_counter() - started) * 1000.0, {"asset_returns": result["input_shape"]}, datetime.now(timezone.utc))


@router.get("/stress/scenarios", response_model=RiskResponse)
def available_historical_scenarios() -> dict:
    """List bundled, source-documented daily crisis datasets for the Risk Lab."""
    result, elapsed = run_core(list_crisis_scenarios)
    return envelope({"scenarios": result}, elapsed, {"scenarios": len(result)})


@router.post("/stress/scenarios/{scenario_id}/replay", response_model=RiskResponse)
def replay_packaged_crisis(
    scenario_id: str, request: HistoricalScenarioReplayRequest
) -> dict:
    """Replay a bundled historical return panel and include realized-volatility paths.

    All instruments are available only for periods in which the underlying
    actually traded. Equal weights are used by default. The packaged snapshots
    have no runtime data-provider dependency.
    """
    try:
        scenario, returns = load_crisis_scenario(scenario_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Bundled crisis data is invalid") from exc

    tickers = list(returns.columns)
    weights = np.full(len(tickers), 1.0 / len(tickers)) if request.weights is None else np.asarray(request.weights, dtype=float)
    if weights.shape != (len(tickers),):
        raise HTTPException(status_code=422, detail=f"weights must have {len(tickers)} values for this scenario")

    result, elapsed = run_core(lambda: replay_historical_crisis(
        weights, returns, scenario["start_date"], scenario["end_date"]
    ))
    portfolio_returns = returns.to_numpy(dtype=float) @ weights
    portfolio_vol = pd.Series(portfolio_returns, index=returns.index).rolling(window=20, min_periods=5).std(ddof=1) * sqrt(252)
    asset_vol = returns.rolling(window=20, min_periods=5).std(ddof=1) * sqrt(252)
    volatility_path = []
    for i, day in enumerate(returns.index):
        volatility_path.append({
            "date": day.date().isoformat(),
            "portfolio_annualized_volatility": None if pd.isna(portfolio_vol.iloc[i]) else float(portfolio_vol.iloc[i]),
            "asset_annualized_volatility": {
                ticker: None if pd.isna(asset_vol.iloc[i][ticker]) else float(asset_vol.iloc[i][ticker])
                for ticker in tickers
            },
        })
    daily_returns = [
        {"date": day.date().isoformat(), "returns": {ticker: float(returns.iloc[i][ticker]) for ticker in tickers}}
        for i, day in enumerate(returns.index)
    ]
    data = {
        "scenario": scenario,
        "tickers": tickers,
        "weights": weights,
        "replay": result,
        "daily_returns": daily_returns,
        "rolling_volatility_path": volatility_path,
    }
    return envelope(data, elapsed, {"asset_returns": [len(returns), len(tickers)], "weights": [len(weights)]})


@router.post("/stress/reverse", response_model=RiskResponse)
def reverse_stress(request: ReverseStressRequest) -> dict:
    """Find the minimum-norm factor shock that reaches a target loss."""
    loadings = pd.DataFrame(request.factor_loadings, index=request.tickers, columns=request.factor_names, dtype=float)
    result, elapsed = run_core(lambda: run_reverse_stress_test(
        np.asarray(request.weights), np.asarray(request.covariance_matrix), loadings, request.target_loss_threshold
    ))
    return envelope(result, elapsed, {"covariance_matrix": [len(request.tickers), len(request.tickers)], "factor_loadings": [len(request.tickers), len(request.factor_names)]})


@router.post("/stress/macro", response_model=RiskResponse)
def macro_stress(request: MacroShockRequest) -> dict:
    """Apply named deterministic factor shocks and return portfolio impact."""
    loadings = pd.DataFrame(request.factor_loadings, index=request.tickers, columns=request.factor_names, dtype=float)
    result, elapsed = run_core(lambda: apply_macro_scenario_shock(np.asarray(request.weights), loadings, request.macro_shocks))
    return envelope({"portfolio_return_impact": result}, elapsed, {"factor_loadings": [len(request.tickers), len(request.factor_names)]})
