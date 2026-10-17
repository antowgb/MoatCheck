"""Authenticated on-demand market-symbol ingestion endpoint."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import AuthenticatedUser, require_current_user
from app.services.data_pipeline_orchestrator import fetch_and_cache_eod_prices
from app.services.universe_manager import UniverseManager

router = APIRouter()


class IngestTickerRequest(BaseModel):
    """Strict user-entered symbol contract."""

    model_config = ConfigDict(extra="forbid")
    ticker: str = Field(min_length=1, max_length=16, pattern=r"^[A-Za-z0-9.^= -]+$")


@router.post("/universe/ingest", status_code=202)
async def ingest_market_symbol(
    body: IngestTickerRequest,
    user: AuthenticatedUser = Depends(require_current_user),
) -> dict:
    """Validate an exact symbol match, then fetch/cache EOD bars immediately."""
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
    try:
        result["price_refresh"] = await fetch_and_cache_eod_prices(result["ticker"])
    except Exception as exc:
        # Keep the idempotent high-priority queue item for the next scheduled
        # run. Discovery must still succeed when a provider is temporarily down.
        result["price_refresh"] = {"status": "queued", "error": type(exc).__name__}
    return result
