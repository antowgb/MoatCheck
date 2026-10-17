"""Batch data adapter for three-year portfolio discovery analyses.

The expensive history-loading step is deliberately separate from the pure
proposal math in :mod:`app.advisor.discovery`, so tests can remain fully offline.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any
from uuid import UUID

import pandas as pd

from app.advisor.discovery import DiscoveryConfig, build_portfolio_discovery_proposals
from app.models.portfolio import PortfolioConfig


PRICE_PAGE_SIZE = 1000
TICKER_BATCH_SIZE = 25
DISCOVERY_UNIVERSE_LIMIT = 250
logger = logging.getLogger(__name__)


async def _fetch_price_panel(client: Any, tickers: list[str], start_date: date) -> pd.DataFrame:
    """Fetch dated close prices in bounded batches and return simple returns.

    PostgREST caps result pages, so each ticker group is paged until exhausted.
    ``prices`` is treated as a read-only cache; on-demand data ingestion belongs
    to the data pipeline. Errors are propagated instead of silently presenting
    an incomplete universe as a complete discovery run.
    """
    semaphore = asyncio.Semaphore(8)

    async def fetch_batch(batch: list[str]) -> list[dict[str, Any]]:
        batch_rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            async with semaphore:
                query = (
                    client.table("prices").select("ticker,date,adjusted_close,close")
                    .in_("ticker", batch).gte("date", start_date.isoformat())
                    .order("date").order("ticker")
                    .range(offset, offset + PRICE_PAGE_SIZE - 1)
                )
                response = await query.execute()
            page = response.data or []
            batch_rows.extend(page)
            if len(page) < PRICE_PAGE_SIZE:
                break
            offset += PRICE_PAGE_SIZE
        return batch_rows

    batches = [tickers[index:index + TICKER_BATCH_SIZE] for index in range(0, len(tickers), TICKER_BATCH_SIZE)]
    def to_price_panel(rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_convert(None)
        adjusted = pd.to_numeric(frame.get("adjusted_close"), errors="coerce")
        close = pd.to_numeric(frame.get("close"), errors="coerce")
        frame["price"] = adjusted.where(adjusted.notna() & adjusted.gt(0), close)
        frame = frame.dropna(subset=["ticker", "date", "price"])
        frame = frame.loc[frame["price"] > 0]
        return frame.pivot_table(index="date", columns="ticker", values="price", aggfunc="last").sort_index()

    price_panels: list[pd.DataFrame] = []
    # Bound transient Python row-dict memory to eight ticker batches at once.
    for index in range(0, len(batches), 8):
        page_groups = await asyncio.gather(*(fetch_batch(batch) for batch in batches[index:index + 8]))
        price_panels.extend(panel for batch_rows in page_groups if not (panel := to_price_panel(batch_rows)).empty)
    if not price_panels:
        return pd.DataFrame()
    panel = pd.concat(price_panels, axis=1).sort_index()
    return panel.pct_change(fill_method=None).replace([float("inf"), float("-inf")], float("nan")).iloc[1:]


async def run_portfolio_discovery(
    client: Any,
    portfolio: PortfolioConfig,
    *,
    config: DiscoveryConfig | None = None,
    as_of: date | None = None,
    extra_tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Load a three-year saved-price panel and produce portfolio proposals.

    The caller must establish portfolio ownership before invoking this service.
    The default universe uses currently indexed ``stocks`` symbols plus saved
    portfolio tickers; new research candidates are not added to storage here.
    Returns point-in-time performance comparisons and optimizer alternatives.
    """
    settings = config or DiscoveryConfig()
    end = as_of or date.today()
    start = end - timedelta(days=365 * settings.lookback_years + 8)
    # Discovery needs three years of history per candidate. Fetching every
    # indexed symbol in the foreground request can exceed the web host's
    # request deadline, so rank a bounded core universe by measured liquidity.
    # Always retain the user's holdings even when they are outside that set.
    universe: list[str] = []
    try:
        response = await (
            client.table("market_universe_memberships")
            .select("ticker")
            .eq("tier", "core")
            .order("adv_usd", desc=True)
            .range(0, DISCOVERY_UNIVERSE_LIMIT - 1)
            .execute()
        )
        universe = [
            str(row.get("ticker", "")).strip().upper()
            for row in (response.data or [])
            if row.get("ticker")
        ]
    except Exception:
        logger.exception("Could not load the core discovery universe; falling back to indexed stocks")
    if not universe:
        offset = 0
        while True:
            response = await client.table("stocks").select("ticker").range(offset, offset + PRICE_PAGE_SIZE - 1).execute()
            page = response.data or []
            universe.extend(str(row.get("ticker", "")).strip().upper() for row in page if row.get("ticker"))
            if len(page) < PRICE_PAGE_SIZE:
                break
            offset += PRICE_PAGE_SIZE
        universe = sorted(set(universe))[:DISCOVERY_UNIVERSE_LIMIT]
    tickers = list(dict.fromkeys([*portfolio.tickers, *(extra_tickers or []), *universe]))
    if not tickers:
        raise ValueError("No indexed assets are available for discovery")
    price_panel = await _fetch_price_panel(client, tickers, start)
    if price_panel.empty:
        raise ValueError("No historical price data is available for the discovery window")
    proposals = build_portfolio_discovery_proposals(price_panel, portfolio.target_weights, config=settings)
    proposals.update({"portfolio_id": str(portfolio.id), "portfolio_name": portfolio.name,
                      "as_of": end.isoformat(), "snapshot_timestamp": pd.Timestamp.now(tz="UTC").isoformat()})
    return proposals


async def run_discovery_batch_for_owner(client: Any, owner_id: UUID | str, *, limit: int = 100) -> list[dict[str, Any]]:
    """Run the portfolio discovery worker for each portfolio owned by one user.

    This function is suitable for a scheduled task or an explicitly launched
    batch job. It isolates individual portfolio failures in the returned
    status records so one sparse portfolio does not stop the rest of the batch.
    """
    from app.repositories.portfolios import PortfolioRepository

    portfolios = await PortfolioRepository(client).list_for_owner(owner_id, limit=limit)
    semaphore = asyncio.Semaphore(4)

    async def run_one(portfolio: PortfolioConfig) -> dict[str, Any]:
        try:
            async with semaphore:
                proposals = await run_portfolio_discovery(client, portfolio)
            return {"portfolio_id": str(portfolio.id), "status": "complete", "proposals": proposals}
        except Exception as exc:
            logger.exception("Portfolio discovery batch item failed for %s", portfolio.id)
            return {"portfolio_id": str(portfolio.id), "status": "unavailable", "detail": str(exc)}

    return await asyncio.gather(*(run_one(portfolio) for portfolio in portfolios))
