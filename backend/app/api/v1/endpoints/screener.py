"""Cursor-paginated screener backed by a database keyset query."""

from __future__ import annotations

import base64
import json
import logging
import math
from datetime import date, timedelta
from time import perf_counter
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.db.supabase import get_async_supabase_client
from app.qualitative import config as qualitative_config

router = APIRouter()
logger = logging.getLogger(__name__)
SORT_FIELDS = {
    "composite_score", "fundamental_score", "risk_score", "revenue_growth_yoy",
    "market_cap", "pe_trailing", "debt_to_ebitda",
}


async def _page_stock_metadata(client: Any, rows: list[dict[str, Any]]) -> None:
    """Attach each page ticker's successful refresh time and event tally.

    The SQL pagination RPC owns the heavy latest-score/fundamental joins, but
    intentionally returns a compact projection. This adapter adds display-only
    metadata from the same stored tables without changing score calculations.
    The tally follows the legacy contract: 90-day events, excluding low
    confidence, with event_date preferred to created_at.
    """
    tickers = [str(row["ticker"]) for row in rows]
    if not tickers:
        return

    for row in rows:
        row["qualitative_tally"] = None

    try:
        stock_response = await (
            client.table("stocks")
            .select("ticker,updated_at")
            .in_("ticker", tickers)
            .execute()
        )
        stock_updates = {
            str(item["ticker"]): item.get("updated_at")
            for item in (stock_response.data or [])
        }
        for row in rows:
            row["updated_at"] = stock_updates.get(str(row["ticker"]))
    except Exception:
        logger.warning("Could not load last refresh times for screener page", exc_info=True)
        for row in rows:
            row["updated_at"] = None

    try:
        notes_response = await (
            client.table("qualitative_notes")
            .select("ticker,sentiment,confidence,event_date,created_at")
            .in_("ticker", tickers)
            .neq("confidence", qualitative_config.TALLY_EXCLUDE_CONFIDENCE)
            .execute()
        )
        cutoff = (date.today() - timedelta(days=qualitative_config.TALLY_WINDOW_DAYS)).isoformat()
        tallies = {
            ticker: {
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "window_days": qualitative_config.TALLY_WINDOW_DAYS,
            }
            for ticker in tickers
        }
        for note in notes_response.data or []:
            event_ticker = str(note.get("ticker", ""))
            sentiment = note.get("sentiment")
            effective_date = note.get("event_date") or (note.get("created_at") or "")[:10]
            if (
                event_ticker in tallies
                and note.get("confidence") != qualitative_config.TALLY_EXCLUDE_CONFIDENCE
                and effective_date >= cutoff
                and sentiment in qualitative_config.SENTIMENTS
            ):
                tallies[event_ticker][sentiment] += 1
        for row in rows:
            row["qualitative_tally"] = tallies[str(row["ticker"])]
    except Exception:
        logger.warning("Could not load qualitative tallies for screener page", exc_info=True)


def _encode_cursor(value: float | None, ticker: str, sort_by: str, sort_dir: str) -> str:
    """Encode the stable sort-value/ticker keyset position as URL-safe text."""
    payload = json.dumps(
        {"value": value, "ticker": ticker, "sort_by": sort_by, "sort_dir": sort_dir},
        separators=(",", ":"), allow_nan=False,
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_cursor(cursor: str | None, sort_by: str, sort_dir: str) -> tuple[float | None, str | None]:
    """Validate an opaque cursor and reject cursors from a different ordering."""
    if cursor is None:
        return None, None
    if len(cursor) > 2048:
        raise HTTPException(status_code=422, detail="Invalid screener cursor")
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        if not isinstance(payload, dict):
            raise ValueError("cursor payload must be an object")
        value = payload.get("value")
        ticker = payload.get("ticker")
        if payload.get("sort_by") != sort_by or payload.get("sort_dir") != sort_dir:
            raise ValueError("cursor ordering mismatch")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("invalid cursor value")
        if not isinstance(ticker, str) or not ticker or len(ticker) > 20:
            raise ValueError("invalid cursor ticker")
        return (float(value) if value is not None else None), ticker
    except (ValueError, TypeError, json.JSONDecodeError, base64.binascii.Error) as exc:
        raise HTTPException(status_code=422, detail="Invalid screener cursor") from exc


@router.get("/screener")
async def get_screener_page(
    sector: str | None = None,
    min_score: float | None = Query(default=None, ge=0, le=100, allow_inf_nan=False),
    min_growth: float | None = Query(default=None, allow_inf_nan=False),
    min_risk_score: float | None = Query(default=None, ge=0, le=100, allow_inf_nan=False),
    market_cap_min: float | None = Query(default=None, ge=0, allow_inf_nan=False),
    market_cap_max: float | None = Query(default=None, ge=0, allow_inf_nan=False),
    pe_max: float | None = Query(default=None, allow_inf_nan=False),
    debt_to_ebitda_max: float | None = Query(default=None, allow_inf_nan=False),
    include_etfs: bool = False,
    sort_by: str = Query(default="composite_score"),
    sort_dir: str = Query(default="desc", pattern="^(asc|desc)$"),
    limit: int = Query(default=100, ge=1, le=200),
    cursor: str | None = None,
) -> dict[str, Any]:
    """Return one RLS-protected page, seeking after a stable keyset cursor.

    The PostgreSQL RPC applies filters, latest-score joins, ordering and limit
    before returning rows, so API memory and response size are bounded by 201
    records rather than the complete stock universe.
    """
    if sort_by not in SORT_FIELDS:
        raise HTTPException(status_code=422, detail="Unsupported screener sort field")
    cursor_value, cursor_ticker = _decode_cursor(cursor, sort_by, sort_dir)
    params = {
        "p_sector": sector,
        "p_min_score": min_score,
        "p_min_growth": min_growth,
        "p_min_risk_score": min_risk_score,
        "p_market_cap_min": market_cap_min,
        "p_market_cap_max": market_cap_max,
        "p_pe_max": pe_max,
        "p_debt_to_ebitda_max": debt_to_ebitda_max,
        "p_include_etfs": include_etfs,
        "p_sort_by": sort_by,
        "p_sort_dir": sort_dir,
        "p_cursor_value": cursor_value,
        "p_cursor_ticker": cursor_ticker,
        "p_limit": limit + 1,
    }
    started = perf_counter()
    try:
        client = await get_async_supabase_client()
        response = await client.rpc("screener_page", params).execute()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Screener data is temporarily unavailable") from exc
    fetched = list(response.data or [])
    has_more = len(fetched) > limit
    rows = fetched[:limit]
    await _page_stock_metadata(client, rows)
    next_cursor = None
    if has_more and rows:
        last = rows[-1]
        next_cursor = _encode_cursor(last.get(sort_by), str(last["ticker"]), sort_by, sort_dir)
    return {
        "rows": rows,
        "next_cursor": next_cursor,
        "has_more": has_more,
        "page_size": len(rows),
        "execution_time_ms": round((perf_counter() - started) * 1000, 3),
    }
