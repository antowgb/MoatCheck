"""Tenant isolation checks for owner-scoped API operations and SQL policies."""

from pathlib import Path
from uuid import uuid4

from app.api.dependencies import AuthenticatedUser, require_current_user
from app.api.v1.endpoints import advisor
from app.main import app
from tests.asgi_client import TestClient


def test_cross_tenant_tracking_read_and_write_return_not_found(monkeypatch):
    owner = AuthenticatedUser(id=uuid4(), email="tenant-a@example.test")
    foreign_portfolio_id = uuid4()
    async def get_client():
        return _client()

    monkeypatch.setattr(advisor, "get_async_supabase_client", get_client)

    async def no_foreign_portfolio(*_args, **_kwargs):
        return None

    monkeypatch.setattr(advisor.PortfolioRepository, "get_for_owner", no_foreign_portfolio)
    app.dependency_overrides[require_current_user] = lambda: owner
    try:
        with TestClient(app) as client:
            read = client.get(f"/api/v1/advisor/portfolios/{foreign_portfolio_id}/tracking")
            write = client.request(
                "PUT",
                f"/api/v1/advisor/portfolios/{foreign_portfolio_id}/holdings",
                json={"holdings": [{"ticker": "AAPL", "quantity": 1, "average_buy_price": 10}]},
            )
        assert read.status_code == 404, read.text
        assert write.status_code == 404, write.text
    finally:
        app.dependency_overrides.pop(require_current_user, None)


def _client():
    """The portfolio ownership check returns before a database query is needed."""
    return object()


def test_owner_policies_and_privileged_rpc_grants_are_explicit():
    root = Path(__file__).resolve().parents[2]
    migration = (root / "supabase_migration_security_hardening.sql").read_text(encoding="utf-8").lower()
    privacy = (root / "supabase_migration_privacy.sql").read_text(encoding="utf-8").lower()
    tracking = (root / "supabase_migration_portfolio_tracking.sql").read_text(encoding="utf-8").lower()

    for table in ("portfolio_holdings", "portfolio_settings", "advisor_recommendations"):
        assert f"on public.{table}" in migration
    assert migration.count("auth.uid() = owner_id") >= 3
    assert "revoke all on function public.delete_owner_quant_data(uuid) from public, anon, authenticated" in migration
    assert "grant execute on function public.delete_owner_quant_data(uuid) to service_role" in migration
    assert "revoke all on function public.replace_portfolio_holdings(uuid, jsonb) from public, anon, authenticated" in tracking
    assert "grant execute on function public.replace_portfolio_holdings(uuid, jsonb) to service_role" in tracking
    assert "grant execute on function public.delete_owner_quant_data(uuid) to service_role" in privacy
