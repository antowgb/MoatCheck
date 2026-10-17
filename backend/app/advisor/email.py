"""Optional, post-persistence email delivery for explicitly requested advice."""

from __future__ import annotations

import html
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)


async def send_recommendation_email(
    recipient: str | None,
    recommendation: dict[str, Any],
) -> str:
    """Send a saved recommendation through Resend when configured.

    Delivery is deliberately opt-in at the API request level. A missing recipient
    or mail configuration does not roll back the persisted recommendation; the
    returned status lets the caller explain whether a message was sent.
    """
    if not recipient:
        return "skipped_no_account_email"

    api_key = os.getenv("RESEND_API_KEY")
    sender = os.getenv("ADVISOR_EMAIL_FROM")
    if not api_key or not sender:
        return "not_configured"

    portfolio_name = str(recommendation.get("portfolio_name") or "your portfolio")
    safe_subject_name = re.sub(r"[\r\n]+", " ", portfolio_name).strip()[:100]
    app_url = os.getenv("MOATCHECK_PUBLIC_URL", "https://moatcheck.onrender.com").rstrip("/")
    advisor_url = html.escape(f"{app_url}/advisor/", quote=True)
    horizon = int(recommendation.get("holding_period_days", 0))
    rationale = recommendation.get("rationale") or []
    rationale_html = "".join(
        f"<li><strong>{html.escape(str(step.get('title', '')))}</strong>: "
        f"{html.escape(str(step.get('explanation', '')))}</li>"
        for step in rationale[:6]
        if isinstance(step, dict)
    )
    target_weights = recommendation.get("target_weights") or {}
    allocation_html = "".join(
        f"<li>{html.escape(str(ticker))}: {float(weight) * 100:.1f}%</li>"
        for ticker, weight in target_weights.items()
    )
    text = (
        f"MoatCheck saved a portfolio recommendation for {portfolio_name}.\n"
        f"Investment horizon: {horizon} days.\n"
        "Review the proposal in your MoatCheck account before making any decision. "
        "This is a model estimate, not personal financial advice, and no trades were placed."
    )
    payload = {
        "from": sender,
        "to": [recipient],
        "subject": f"MoatCheck portfolio recommendation · {safe_subject_name}",
        "text": text,
        "html": (
            "<div style=\"font-family:Arial,sans-serif;color:#17202c;max-width:620px;margin:auto\">"
            "<p style=\"color:#10b981;font-weight:700\">MOATCHECK · PORTFOLIO ADVISOR</p>"
            f"<h1 style=\"font-size:22px\">A recommendation is ready for {html.escape(portfolio_name)}</h1>"
            f"<p>Analysis horizon: <strong>{horizon} days</strong>. The proposal was saved to your account for review.</p>"
            f"<h2 style=\"font-size:16px\">Proposed allocation</h2><ul>{allocation_html}</ul>"
            f"<h2 style=\"font-size:16px\">Why this result</h2><ul>{rationale_html}</ul>"
            f"<p><a href=\"{advisor_url}\">Open your MoatCheck Advisor</a> and inspect the assumptions, risk estimates, and execution-cost model. "
            "The tool does not place trades and this message is not personal financial advice.</p>"
            "</div>"
        ),
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=3.0)) as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
        return "sent"
    except (httpx.HTTPError, ValueError, TypeError):
        # Never log recipients, credentials, message bodies, or provider response bodies.
        logger.warning("Advisor notification delivery failed; recommendation remains saved.")
        return "delivery_failed"
