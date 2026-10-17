"""Chunked Supabase reads used to avoid per-ticker query loops."""

from __future__ import annotations

from typing import Any, Iterable

from app.data.supabase_client import execute_with_retry, get_supabase

_CHUNK_SIZE = 200
_PAGE_SIZE = 1000


def _chunks(values: list[str], size: int = _CHUNK_SIZE) -> Iterable[list[str]]:
    for offset in range(0, len(values), size):
        yield values[offset:offset + size]


def _paged_rows(query: Any, context: str) -> Iterable[dict[str, Any]]:
    offset = 0
    while True:
        page = execute_with_retry(query.range(offset, offset + _PAGE_SIZE - 1), context=context).data
        yield from page
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE


def latest_rows_by_ticker(table: str, tickers: list[str], date_column: str) -> dict[str, dict[str, Any]]:
    """Return newest row by date column for each ticker, batching IN queries."""
    result: dict[str, dict[str, Any]] = {}
    for group in _chunks(list(dict.fromkeys(tickers))):
        query = (get_supabase().table(table).select("*").in_("ticker", group)
                 .order(date_column, desc=True))
        for row in _paged_rows(query, f"{table} batch ({len(group)})"):
            result.setdefault(row["ticker"], row)
    return result


def latest_scores_by_ticker(tickers: list[str]) -> dict[str, dict[str, Any]]:
    return latest_rows_by_ticker("scores", tickers, "computed_at")


def latest_fundamentals_by_ticker(tickers: list[str]) -> dict[str, dict[str, Any]]:
    return latest_rows_by_ticker("fundamentals", tickers, "report_date")


def fundamentals_history_by_ticker(tickers: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Return all point-in-time snapshots in one query per bounded ticker chunk."""
    result: dict[str, list[dict[str, Any]]] = {ticker: [] for ticker in tickers}
    for group in _chunks(list(dict.fromkeys(tickers))):
        query = (get_supabase().table("fundamentals").select("*").in_("ticker", group)
                 .order("ticker").order("know_date"))
        for row in _paged_rows(query, f"fundamentals history batch ({len(group)})"):
            result.setdefault(row["ticker"], []).append(row)
    return result


def stock_metadata_by_ticker(tickers: list[str], columns: str = "ticker, sector") -> dict[str, dict[str, Any]]:
    """Fetch requested stock metadata in bounded, batched reads."""
    result: dict[str, dict[str, Any]] = {}
    for group in _chunks(list(dict.fromkeys(tickers))):
        rows = execute_with_retry(
            get_supabase().table("stocks").select(columns).in_("ticker", group),
            context=f"stocks batch ({len(group)})",
        ).data
        result.update({row["ticker"]: row for row in rows})
    return result
