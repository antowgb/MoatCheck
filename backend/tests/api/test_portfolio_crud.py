"""Authenticated portfolio configuration create/update/delete contracts."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.api.dependencies import AuthenticatedUser, require_current_user
from app.api.v1.endpoints import portfolio as portfolio_module
from app.main import app
from app.models.portfolio import PortfolioConfig
from tests.asgi_client import TestClient


OWNER_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
PORTFOLIO_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")


class FakePortfolioRepository:
    item: PortfolioConfig | None = None
    saved: PortfolioConfig | None = None

    def __init__(self, _client):
        pass

    async def get_for_owner(self, portfolio_id, owner_id):
        if (
            self.item is None
            or self.item.id != UUID(str(portfolio_id))
            or self.item.owner_id != UUID(str(owner_id))
        ):
            return None
        return self.item

    async def save(self, item):
        saved = item.model_copy(update={"updated_at": datetime.now(timezone.utc)})
        self.item = saved
        type(self).saved = saved
        return saved

    async def delete_for_owner(self, portfolio_id, owner_id):
        item = await self.get_for_owner(portfolio_id, owner_id)
        if item is None:
            return False
        self.item = None
        return True


def _portfolio() -> PortfolioConfig:
    return PortfolioConfig(
        id=PORTFOLIO_ID,
        owner_id=OWNER_ID,
        name="Core",
        tickers=["AAA", "BBB"],
        target_weights={"AAA": 0.6, "BBB": 0.4},
    )


def test_owner_can_update_portfolio_config_without_changing_owner(monkeypatch):
    FakePortfolioRepository.item = _portfolio()
    FakePortfolioRepository.saved = None

    async def fake_db():
        return object()

    async def no_refresh(_tickers):
        return []

    monkeypatch.setattr(portfolio_module, "get_async_supabase_client", fake_db)
    monkeypatch.setattr(portfolio_module, "PortfolioRepository", FakePortfolioRepository)
    monkeypatch.setattr(portfolio_module, "_refresh_new_tickers", no_refresh)
    app.dependency_overrides[require_current_user] = lambda: AuthenticatedUser(OWNER_ID)
    try:
        with TestClient(app) as client:
            response = client.request("PUT",
                f"/api/v1/portfolio/configs/{PORTFOLIO_ID}",
                json={
                    "name": "Long term",
                    "tickers": ["AAA", "CCC"],
                    "target_weights": {"AAA": 0.35, "CCC": 0.65},
                    "description": "Updated allocation",
                },
            )
    finally:
        app.dependency_overrides.pop(require_current_user, None)

    assert response.status_code == 200, response.text
    assert response.json()["data"]["name"] == "Long term"
    assert response.json()["data"]["owner_id"] == str(OWNER_ID)
    assert FakePortfolioRepository.saved is not None
    assert FakePortfolioRepository.saved.tickers == ["AAA", "CCC"]


def test_owner_cannot_update_another_users_portfolio(monkeypatch):
    FakePortfolioRepository.item = _portfolio()

    async def fake_db():
        return object()

    monkeypatch.setattr(portfolio_module, "get_async_supabase_client", fake_db)
    monkeypatch.setattr(portfolio_module, "PortfolioRepository", FakePortfolioRepository)
    app.dependency_overrides[require_current_user] = lambda: AuthenticatedUser(uuid4())
    try:
        with TestClient(app) as client:
            response = client.request("PUT",
                f"/api/v1/portfolio/configs/{PORTFOLIO_ID}",
                json={
                    "name": "Attempted takeover",
                    "tickers": ["AAA"],
                    "target_weights": {"AAA": 1.0},
                },
            )
    finally:
        app.dependency_overrides.pop(require_current_user, None)

    assert response.status_code == 404
