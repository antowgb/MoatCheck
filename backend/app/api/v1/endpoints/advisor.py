"""Owner-scoped medium/long-horizon portfolio advisory API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import perf_counter
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from app.advisor.service import build_advisory_recommendation
from app.services.portfolio_discovery import run_portfolio_discovery
from app.advisor.email import send_recommendation_email
from app.api.dependencies import AuthenticatedUser, get_optional_current_user, require_current_user
from app.api.v1.endpoints._utils import envelope, json_safe
from app.db.supabase import get_async_supabase_client
from app.models.advisory_recommendation import AdvisoryRecommendation
from app.models.portfolio_tracking import PortfolioHolding
from app.repositories.advisory_recommendations import AdvisoryRecommendationRepository
from app.repositories.portfolios import PortfolioRepository
from app.repositories.portfolio_tracking import PortfolioTrackingRepository
from app.schemas.advisor import (
    AdvisorPreviewRequest, AdvisorRecommendationCreateInput,
    AdvisorResponse, ApplyDiscoveryProposalInput, PortfolioHoldingsInput, PortfolioTrackingSettingsInput,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _json_model(model: object) -> dict:
    """Serialize Pydantic records with UUID/date values made JSON safe."""
    return model.model_dump(mode="json")


async def _tracking_context(client: object, portfolio_id: UUID, *, dca_override: float | None = None, horizon_override: int | None = None) -> tuple[list[dict], float, int]:
    tracking = PortfolioTrackingRepository(client)
    holdings = await tracking.list_holdings(portfolio_id)
    settings = await tracking.get_settings(portfolio_id)
    return (
        [holding.model_dump(mode="json") for holding in holdings],
        float(settings.monthly_dca_amount if dca_override is None else dca_override),
        int(settings.target_horizon_years if horizon_override is None else horizon_override),
    )


def _starting_weights(tickers: list[str], saved_weights: dict[str, float], holdings: list[dict]) -> dict[str, float]:
    """Use tracked positions as the starting allocation; otherwise start in cash."""
    if not holdings:
        return {ticker: 0.0 for ticker in tickers}
    return saved_weights


async def _queue_price_refreshes(client: object, tickers: list[str]) -> list[dict[str, str]]:
    """Queue cached-data refresh work without contacting providers in a request."""
    normalized = sorted({str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()})
    if not normalized:
        return []
    try:
        await client.rpc("enqueue_refresh_tickers", {"p_tickers": normalized}).execute()
        return [{"ticker": ticker, "status": "queued"} for ticker in normalized]
    except Exception as exc:
        logger.info("Could not queue EOD refresh for Advisor request (%s)", type(exc).__name__)
        return [{"ticker": ticker, "status": "refresh_deferred"} for ticker in normalized]


@router.get("/portfolios/{portfolio_id}/tracking", response_model=AdvisorResponse)
async def get_portfolio_tracking(
    portfolio_id: UUID,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Return owner-scoped holdings and DCA projection settings."""
    started = perf_counter()
    client = await get_async_supabase_client()
    portfolio = await PortfolioRepository(client).get_for_owner(portfolio_id, user.id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    holdings, dca_amount, horizon = await _tracking_context(client, portfolio_id)
    elapsed = (perf_counter() - started) * 1000.0
    return envelope({"holdings": holdings, "monthly_dca_amount": dca_amount, "target_horizon_years": horizon}, elapsed, {"holdings": len(holdings)})


@router.put("/portfolios/{portfolio_id}/holdings", response_model=AdvisorResponse)
async def replace_portfolio_holdings(
    portfolio_id: UUID,
    request: PortfolioHoldingsInput,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Replace holdings only on a portfolio owned by the authenticated user."""
    started = perf_counter()
    client = await get_async_supabase_client()
    portfolio = await PortfolioRepository(client).get_for_owner(portfolio_id, user.id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    unknown = {holding.ticker for holding in request.holdings} - set(portfolio.tickers)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Holdings must use saved portfolio tickers: {sorted(unknown)}")
    tracking = PortfolioTrackingRepository(client)
    existing = await tracking.list_holdings(portfolio_id)
    previous_tickers = {holding.ticker for holding in existing}
    records = [PortfolioHolding(portfolio_id=portfolio_id, **holding.model_dump()) for holding in request.holdings]
    saved = await tracking.replace_holdings(portfolio_id, records)
    price_refreshes = await _queue_price_refreshes(
        client, sorted({item.ticker for item in records} - previous_tickers)
    )
    elapsed = (perf_counter() - started) * 1000.0
    return envelope({"holdings": [_json_model(item) for item in saved], "price_refreshes": price_refreshes}, elapsed, {"holdings": len(saved)})


@router.put("/portfolios/{portfolio_id}/tracking-settings", response_model=AdvisorResponse)
async def save_portfolio_tracking_settings(
    portfolio_id: UUID,
    request: PortfolioTrackingSettingsInput,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Save the monthly contribution and target projection horizon."""
    started = perf_counter()
    client = await get_async_supabase_client()
    portfolio = await PortfolioRepository(client).get_for_owner(portfolio_id, user.id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    repository = PortfolioTrackingRepository(client)
    existing = await repository.get_settings(portfolio_id)
    settings = existing.model_copy(update={**request.model_dump(), "updated_at": datetime.now(timezone.utc)})
    saved = await repository.save_settings(settings)
    elapsed = (perf_counter() - started) * 1000.0
    return envelope(_json_model(saved), elapsed, {"settings": 1}, saved.updated_at)


@router.post("/recommendations/preview", response_model=AdvisorResponse)
async def preview_recommendation(
    request: AdvisorPreviewRequest,
    user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> dict:
    """Calculate an unpersisted advice preview from an explicit dated snapshot.

    Saved portfolios are loaded only through the verified user's owner scope.
    Inline allocations are allowed without authentication and are never saved.
    """
    started = perf_counter()
    client = await get_async_supabase_client() if request.portfolio_id is not None else None
    portfolio = None
    latest = None
    if request.portfolio_id is not None:
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required for saved portfolios")
        portfolio = await PortfolioRepository(client).get_for_owner(request.portfolio_id, user.id)
        if portfolio is None:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        latest = await AdvisoryRecommendationRepository(client).latest_for_portfolio(portfolio.id, user.id)
        holdings, monthly_dca, horizon_years = await _tracking_context(
            client, portfolio.id, dca_override=request.monthly_dca_amount,
            horizon_override=request.target_horizon_years,
        )
        tickers = portfolio.tickers
        weights = _starting_weights(tickers, portfolio.target_weights, holdings)
        name = portfolio.name
    else:
        holdings = []
        monthly_dca = float(request.monthly_dca_amount or 0.0)
        horizon_years = int(request.target_horizon_years or 5)
        tickers = request.tickers or []
        weights = request.current_weights or {}
        name = request.portfolio_name or "Preview portfolio"

    try:
        result = build_advisory_recommendation(
            portfolio_name=name,
            tickers=tickers,
            current_weights=weights,
            market=request.market,
            policy=request.policy,
            last_decision_at=latest.created_at if latest else None,
            holdings=holdings,
            monthly_dca_amount=monthly_dca,
            target_horizon_years=horizon_years,
        )
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_advisor_input", "message": str(exc)}) from exc
    elapsed = (perf_counter() - started) * 1000.0
    result["latency_ms"] = elapsed
    return envelope(result, elapsed, result["input_shape"], request.market.as_of)


@router.post("/portfolios/{portfolio_id}/recommendations", response_model=AdvisorResponse)
async def create_recommendation(
    portfolio_id: UUID,
    request: AdvisorRecommendationCreateInput,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Calculate and persist a formal recommendation for an owned portfolio."""
    started = perf_counter()
    client = await get_async_supabase_client()
    portfolio = await PortfolioRepository(client).get_for_owner(portfolio_id, user.id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    recommendation_repo = AdvisoryRecommendationRepository(client)
    latest = await recommendation_repo.latest_for_portfolio(portfolio_id, user.id)
    holdings, monthly_dca, horizon_years = await _tracking_context(client, portfolio_id)
    try:
        result = build_advisory_recommendation(
            portfolio_name=portfolio.name,
            tickers=portfolio.tickers,
            current_weights=_starting_weights(portfolio.tickers, portfolio.target_weights, holdings),
            market=request.market,
            policy=request.policy,
            last_decision_at=latest.created_at if latest else None,
            holdings=holdings,
            monthly_dca_amount=monthly_dca,
            target_horizon_years=horizon_years,
        )
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_advisor_input", "message": str(exc)}) from exc
    result_json = json_safe(result)
    item = AdvisoryRecommendation(
        owner_id=user.id,
        portfolio_id=portfolio.id,
        horizon_days=request.policy.holding_period_days,
        minimum_holding_period_days=request.policy.minimum_holding_period_days,
        snapshot_timestamp=request.market.as_of,
        recommendation=result_json,
        rationale=result_json["rationale"],
        telemetry={"latency_ms": result["latency_ms"], "input_shape": result["input_shape"]},
    )
    # Persist before notifying: mail delivery must never discard a valid result.
    saved = await recommendation_repo.save(item)
    saved_data = _json_model(saved)
    if request.email_notification:
        saved_data["email_notification"] = await send_recommendation_email(
            user.email, result_json
        )
    else:
        saved_data["email_notification"] = "not_requested"
    elapsed = (perf_counter() - started) * 1000.0
    return envelope(saved_data, elapsed, result["input_shape"], request.market.as_of)


@router.get("/portfolios/{portfolio_id}/recommendations/latest", response_model=AdvisorResponse)
async def latest_recommendation(
    portfolio_id: UUID,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Return the latest owner-scoped saved recommendation and rationale."""
    started = perf_counter()
    client = await get_async_supabase_client()
    portfolio = await PortfolioRepository(client).get_for_owner(portfolio_id, user.id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    item = await AdvisoryRecommendationRepository(client).latest_for_portfolio(portfolio_id, user.id)
    if item is None:
        raise HTTPException(status_code=404, detail="No recommendation found")
    elapsed = (perf_counter() - started) * 1000.0
    return envelope(_json_model(item), elapsed, {"rationale_steps": len(item.rationale)}, item.snapshot_timestamp)


@router.post("/recommendations/{recommendation_id}/acknowledge", response_model=AdvisorResponse)
async def acknowledge_recommendation(
    recommendation_id: UUID,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Record the authenticated user's human review of a recommendation."""
    started = perf_counter()
    client = await get_async_supabase_client()
    item = await AdvisoryRecommendationRepository(client).acknowledge(recommendation_id, user.id)
    if item is None:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    elapsed = (perf_counter() - started) * 1000.0
    return envelope(_json_model(item), elapsed, {"recommendation": 1}, datetime.now(timezone.utc))


@router.get("/portfolios/{portfolio_id}/proposals", response_model=AdvisorResponse)
async def portfolio_discovery_proposals(
    portfolio_id: UUID,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Return owner-scoped three-year replacement and model portfolio proposals."""
    started = perf_counter()
    client = await get_async_supabase_client()
    portfolio = await PortfolioRepository(client).get_for_owner(portfolio_id, user.id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    try:
        result = await run_portfolio_discovery(client, portfolio)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "discovery_unavailable", "message": str(exc)}) from exc
    except Exception as exc:
        logger.exception("Portfolio discovery failed for %s", portfolio_id)
        raise HTTPException(status_code=503, detail={"code": "discovery_data_unavailable", "message": "Historical market data could not be loaded for portfolio discovery."}) from exc
    elapsed = (perf_counter() - started) * 1000.0
    return envelope(result, elapsed, result.get("telemetry", {}).get("input_shape", {}), result.get("snapshot_timestamp"))


@router.post("/portfolios/{portfolio_id}/apply-proposal", response_model=AdvisorResponse)
async def apply_portfolio_discovery_proposal(
    portfolio_id: UUID,
    request: ApplyDiscoveryProposalInput,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Save a user-selected target allocation; this endpoint never places trades."""
    started = perf_counter()
    client = await get_async_supabase_client()
    repository = PortfolioRepository(client)
    portfolio = await repository.get_for_owner(portfolio_id, user.id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    optimization_parameters = dict(portfolio.optimization_parameters)
    optimization_parameters["last_discovery_proposal_id"] = request.proposal_id
    optimization_parameters["last_discovery_applied_at"] = datetime.now(timezone.utc).isoformat()
    updated = portfolio.model_copy(update={
        "tickers": request.tickers,
        "target_weights": request.target_weights,
        "optimization_parameters": optimization_parameters,
    })
    try:
        saved = await repository.save(updated)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_proposal", "message": str(exc)}) from exc
    refreshes = await _queue_price_refreshes(client, sorted(set(saved.tickers) - set(portfolio.tickers)))
    payload = _json_model(saved)
    payload["price_refreshes"] = refreshes
    payload["trade_execution"] = "not_executed"
    elapsed = (perf_counter() - started) * 1000.0
    return envelope(payload, elapsed, {"tickers": len(saved.tickers)}, saved.updated_at)
