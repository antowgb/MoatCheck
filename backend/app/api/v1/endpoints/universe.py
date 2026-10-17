"""Authenticated on-demand market-symbol ingestion endpoint."""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import AuthenticatedUser, require_current_user
from app.services.data_pipeline_orchestrator import fetch_and_cache_eod_prices
from app.services.universe_manager import UniverseManager, normalize_ticker

router = APIRouter()
logger = logging.getLogger(__name__)


class IngestTickerRequest(BaseModel):
    """Strict user-entered symbol contract."""

    model_config = ConfigDict(extra="forbid")
    ticker: str = Field(min_length=1, max_length=16, pattern=r"^[A-Za-z0-9.^= -]+$")


@router.get("/universe/search")
async def search_market_symbol(
    query: str = Query(..., min_length=1, max_length=16, pattern=r"^[A-Za-z0-9.^= -]+$"),
) -> dict:
    """Look up one exact provider symbol without changing the universe."""
    # Keep normalization in one place without persisting a search result.
    normalized = normalize_ticker(query)
    if not normalized:
        return {"found": False, "ticker": query.strip().upper(), "name": None}
    try:
        lookup = await UniverseManager()._yahoo_lookup(normalized)
    except Exception as exc:
        logger.info("Provider symbol search failed for %s (%s)", normalized, type(exc).__name__)
        raise HTTPException(status_code=503, detail="Market symbol search is temporarily unavailable") from exc
    if not lookup or normalize_ticker(lookup.get("symbol")) != normalized:
        return {"found": False, "ticker": normalized, "name": None}
    return {
        "found": True,
        "ticker": normalized,
        "name": lookup.get("name"),
        "currency": lookup.get("currency"),
        "asset_type": lookup.get("asset_type", "equity"),
    }


@router.post("/universe/ingest", status_code=202)
async def ingest_market_symbol(
    body: IngestTickerRequest,
    background_tasks: BackgroundTasks,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Validate an exact symbol match, then queue its EOD fetch.

    Symbol discovery must stay responsive. The provider fetch is deliberately
    moved out of the request so a slow Yahoo/provider response cannot make the
    portfolio search look frozen or consume the browser's request timeout.
    """
    del user  # Identity is required at the boundary; the symbol is global data.
    manager = UniverseManager()
    try:
        result = await manager.ingest_on_demand(body.ticker)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Market symbol lookup is temporarily unavailable") from exc
    async def refresh_in_background() -> None:
        try:
            await fetch_and_cache_eod_prices(result["ticker"])
        except Exception:
            # The high-priority queue remains available for the scheduled
            # pipeline when an on-demand provider call is unavailable.
            logger.info("On-demand price refresh queued for %s after provider failure", result["ticker"], exc_info=True)

    background_tasks.add_task(refresh_in_background)
    result["price_refresh"] = {"status": "queued"}
    return result
