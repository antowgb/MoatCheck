"""Monthly, idempotency-keyed DCA email digests delivered through Resend.

The worker reads persisted Advisor outputs and account email addresses; the
quantitative calculation itself stays in ``app.advisor.service``.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Any

import httpx

from app.advisor.service import calculate_portfolio_projections
from app.db.supabase import get_async_supabase_client

logger = logging.getLogger(__name__)


def build_monthly_dca_email(
    recipient: str,
    portfolio_name: str,
    recommendation: dict[str, Any],
    *,
    cycle_date: date | None = None,
) -> dict[str, Any]:
    """Build a safe HTML/text Resend payload from the saved long-term plan.

    The digest lists exact buy amounts and the existing Monte Carlo P10/P50/P90
    projections at years 5, 10, and 20 where available. A €500 monthly plan
    split 60/40 becomes €300 and €200 buy instructions. All user and portfolio
    strings are HTML-escaped. Complexity is :math:`O(A+S)` for :math:`A`
    allocation rows and :math:`S` rationale entries.
    """
    cycle = cycle_date or date.today()
    safe_name = html.escape(str(portfolio_name or "Long-term portfolio"))
    dca = recommendation.get("dca_plan") or {}
    currency = html.escape(str(dca.get("currency") or (recommendation.get("valuation") or {}).get("currency") or ""))
    orders = [row for row in dca.get("orders", []) if isinstance(row, dict) and float(row.get("amount", 0) or 0) > 0]
    order_rows = "".join(
        f"<tr><td style='padding:8px;border-bottom:1px solid #263140'>{html.escape(str(row.get('ticker', '')))}</td>"
        f"<td style='padding:8px;border-bottom:1px solid #263140;text-align:right'>{float(row['amount']):,.2f} {currency}</td></tr>"
        for row in orders
    ) or "<tr><td colspan='2' style='padding:8px'>No buy allocation is available. Cash remains unallocated until the plan is updated.</td></tr>"

    valuation = recommendation.get("valuation") or {}
    pnl_fraction = valuation.get("unrealized_pnl_fraction")
    pnl_text = "Not available" if pnl_fraction is None else f"{float(pnl_fraction):+.2%}"
    signal = recommendation.get("trade_signal") or {}
    drift_rows = recommendation.get("tolerance_bands") or []
    health = "Review suggested allocation" if signal.get("triggered") else (
        "Within long-term tolerance bands" if not any(row.get("status") != "BALANCED" for row in drift_rows)
        else "Drift detected; DCA favors underweight assets"
    )

    projection = recommendation.get("projections") or {}
    years = projection.get("years") or []
    available_years = {int(row["year"]): row for row in years if isinstance(row, dict) and "year" in row}
    if not all(year in available_years for year in (5, 10, 20)):
        try:
            years = calculate_portfolio_projections(
                float(projection.get("initial_capital") or 0.0),
                float((dca.get("monthly_amount") or 0.0)),
                float(projection.get("annualized_expected_return") or 0.0),
                float(projection.get("annualized_volatility") or 0.0),
                20,
                seed=42,
            )
            available_years = {int(row["year"]): row for row in years}
        except (ValueError, TypeError, OverflowError):
            available_years = {}
    projection_rows = "".join(
        f"<tr><td style='padding:8px;border-bottom:1px solid #263140'>{year} years</td>"
        + "".join(
            f"<td style='padding:8px;border-bottom:1px solid #263140;text-align:right'>{available_years[year][key]:,.0f} {currency}</td>"
            for key in ("pessimistic_value", "median_value", "optimistic_value")
        )
        + "</tr>"
        for year in (5, 10, 20) if year in available_years
    ) or "<tr><td colspan='4' style='padding:8px'>Long-range scenarios are not available for this portfolio yet.</td></tr>"
    snapshot = str(recommendation.get("snapshot_timestamp") or "Not available")
    cycle_label = cycle.strftime("%B %Y")
    subject = f"MoatCheck monthly DCA plan · {str(portfolio_name)[:80]} · {cycle_label}"
    text_orders = "\n".join(f"BUY {row.get('ticker')}: {float(row['amount']):,.2f} {dca.get('currency', '')}" for row in orders)
    text = (
        f"Monthly contribution plan for {portfolio_name} ({cycle_label})\n"
        f"Health: {health}\nUnrealized PnL: {pnl_text}\n"
        f"Monthly contribution: {float(dca.get('monthly_amount', 0) or 0):,.2f} {dca.get('currency', '')}\n"
        f"Suggested buys:\n{text_orders or 'No buy allocation available.'}\n"
        f"Analysis data as of: {snapshot}\n"
        "These are model estimates for a multi-year horizon, not guaranteed outcomes or personal financial advice. No trades are placed."
    )
    app_url = os.getenv("MOATCHECK_PUBLIC_URL", "https://moatcheck.onrender.com").rstrip("/")
    return {
        "from": os.getenv("ADVISOR_EMAIL_FROM", ""),
        "to": [recipient],
        "subject": subject,
        "text": text,
        "html": (
            "<div style='font-family:Arial,sans-serif;background:#0b0f14;color:#e5edf5;padding:28px;max-width:680px;margin:auto'>"
            "<p style='color:#22c55e;font-weight:700;letter-spacing:2px'>MOATCHECK · MONTHLY ADVISOR</p>"
            f"<h1 style='font-size:24px'>{safe_name}</h1><p>Your DCA plan for <strong>{html.escape(cycle_label)}</strong></p>"
            f"<p style='padding:14px;background:#111720;border-radius:12px'>Portfolio health: <strong>{html.escape(health)}</strong><br/>"
            f"Unrealized PnL: <strong>{html.escape(pnl_text)}</strong><br/>Data snapshot: {html.escape(snapshot)}</p>"
            f"<h2 style='font-size:17px'>Suggested buys · {float(dca.get('monthly_amount', 0) or 0):,.2f} {currency}</h2>"
            f"<table style='width:100%;border-collapse:collapse'>{order_rows}</table>"
            "<h2 style='font-size:17px;margin-top:24px'>Long-term scenarios</h2>"
            "<p style='color:#9ba9ba;font-size:12px'>Monte Carlo estimates, not promises. Values show pessimistic (10th), median (50th), and optimistic (90th) scenarios.</p>"
            f"<table style='width:100%;border-collapse:collapse'><thead><tr><th style='text-align:left;padding:8px'>Horizon</th>"
            "<th style='text-align:right;padding:8px'>10th</th><th style='text-align:right;padding:8px'>50th</th>"
            f"<th style='text-align:right;padding:8px'>90th ({currency})</th></tr></thead><tbody>{projection_rows}</tbody></table>"
            f"<p style='margin-top:24px'><a style='color:#22c55e' href='{html.escape(app_url + '/advisor/', quote=True)}'>Review the plan in MoatCheck</a></p>"
            "<p style='color:#9ba9ba;font-size:12px'>For long-term planning only. Estimates depend on historical inputs, can be wrong, and are not personal financial advice. MoatCheck does not place trades.</p></div>"
        ),
    }


async def send_monthly_dca_email(recipient: str, payload: dict[str, Any], *, idempotency_key: str) -> str:
    """Send one digest through Resend using a stable month/portfolio key.

    Resend's idempotency key prevents duplicate provider delivery when a
    workflow retries the same portfolio and month. Time complexity is one
    bounded HTTP request, :math:`O(1)` local work.
    """
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key or not payload.get("from"):
        return "not_configured"
    request_body = {**payload, "to": [recipient]}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=4.0)) as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {api_key}", "Idempotency-Key": idempotency_key,
                         "Content-Type": "application/json"},
                json=request_body,
            )
            response.raise_for_status()
        return "sent"
    except httpx.HTTPError:
        logger.warning("Monthly DCA digest delivery failed for a portfolio; no recipient or secret logged")
        return "delivery_failed"


async def _execute(query: Any) -> list[dict[str, Any]]:
    result = query.execute()
    if hasattr(result, "__await__"):
        result = await result
    return list(getattr(result, "data", None) or [])


async def run_monthly_dca_digest(
    *, client: Any | None = None, now: datetime | None = None,
) -> dict[str, Any]:
    """Send the monthly digest for each portfolio with a recent saved plan.

    The worker runs on day one only, reads the latest persisted recommendation,
    and emails the verified Supabase Auth account address. Stale plans older
    than 45 days and portfolios without a positive DCA contribution are
    skipped. Provider/database work is :math:`O(P)` requests for :math:`P`
    portfolios; each email is independently reported to allow partial success.
    """
    current = now or datetime.now(timezone.utc)
    if current.day != 1:
        return {"status": "skipped_not_month_start", "sent": 0, "skipped": 0, "failed": 0}
    client = client or await get_async_supabase_client()
    portfolios = await _execute(client.table("portfolios").select("id,owner_id,name"))
    counts = {"sent": 0, "skipped": 0, "failed": 0}
    month_key = current.date().replace(day=1).isoformat()
    for portfolio in portfolios:
        portfolio_id = str(portfolio.get("id", ""))
        try:
            rec_rows = await _execute(
                client.table("advisor_recommendations").select("recommendation,created_at")
                .eq("portfolio_id", portfolio_id).order("created_at", desc=True).limit(1)
            )
            if not rec_rows:
                counts["skipped"] += 1
                continue
            saved = rec_rows[0]
            created = pd_to_datetime_utc(saved.get("created_at"))
            if (current - created).days > 45:
                counts["skipped"] += 1
                continue
            recommendation = saved.get("recommendation") or {}
            dca = recommendation.get("dca_plan") or {}
            if float(dca.get("monthly_amount", 0) or 0) <= 0:
                counts["skipped"] += 1
                continue
            user_response = await client.auth.admin.get_user_by_id(str(portfolio["owner_id"]))
            user = getattr(user_response, "user", None)
            recipient = getattr(user, "email", None)
            if not recipient:
                counts["skipped"] += 1
                continue
            payload = build_monthly_dca_email(
                recipient, str(portfolio.get("name") or "Long-term portfolio"),
                recommendation, cycle_date=current.date(),
            )
            outcome = await send_monthly_dca_email(
                recipient, payload, idempotency_key=f"moatcheck-dca-{portfolio_id}-{month_key}"
            )
            counts["sent" if outcome == "sent" else "failed" if outcome == "delivery_failed" else "skipped"] += 1
        except Exception as exc:
            logger.warning("Monthly DCA digest failed for one portfolio (%s)", type(exc).__name__)
            counts["failed"] += 1
    return {"status": "completed", **counts, "month": month_key, "portfolios": len(portfolios)}


def pd_to_datetime_utc(value: Any) -> datetime:
    """Normalize a PostgREST timestamp without depending on pandas."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("recommendation creation timestamp is missing")
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


async def _async_main() -> int:
    report = await run_monthly_dca_digest()
    logger.info("Monthly DCA digest complete: %s", json.dumps(report))
    return 0 if report.get("failed", 0) == 0 else 1


def main() -> int:
    """CLI entry point used by the monthly GitHub Actions schedule."""
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    return asyncio.run(_async_main())


if __name__ == "__main__":
    raise SystemExit(main())
