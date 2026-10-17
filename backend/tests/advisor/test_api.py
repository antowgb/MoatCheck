from datetime import datetime, timezone
from uuid import UUID, uuid4

from tests.asgi_client import TestClient

from app.api.dependencies import AuthenticatedUser, get_optional_current_user, require_current_user
from app.main import app
from app.models.portfolio import PortfolioConfig


def test_inline_advisor_preview_returns_explainability_and_telemetry(market_snapshot):
    app.dependency_overrides[get_optional_current_user] = lambda: None
    try:
        with TestClient(app) as client:
            payload = market_snapshot.model_dump(mode="json")
            response = client.post("/api/v1/advisor/recommendations/preview", json={
                "market": payload,
                "tickers": ["AAA", "BBB"],
                "current_weights": {"AAA": 0.5, "BBB": 0.5},
                "policy": {"holding_period_days": 60, "minimum_holding_period_days": 30},
            })
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["data"]["holding_period_days"] == 60
        assert body["data"]["var_loss_fraction"] >= 0
        assert body["data"]["valuation"]["available"] is False
        assert body["data"]["projections"]["years"]
        assert body["data"]["dca_plan"]["buy_only"] is True
        assert body["data"]["rationale"]
        assert body["meta"]["execution_time_ms"] >= 0
        assert body["meta"]["snapshot_timestamp"] == payload["as_of"]
    finally:
        app.dependency_overrides.pop(get_optional_current_user, None)


def test_inline_advisor_preview_rejects_short_horizon(market_snapshot):
    app.dependency_overrides[get_optional_current_user] = lambda: None
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/advisor/recommendations/preview", json={
                "market": market_snapshot.model_dump(mode="json"),
                "tickers": ["AAA", "BBB"], "current_weights": {"AAA": 0.5, "BBB": 0.5},
                "policy": {"holding_period_days": 5},
            })
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_optional_current_user, None)


def test_saved_portfolio_preview_loads_only_owner_portfolio(monkeypatch, market_snapshot):
    owner_id, portfolio_id = uuid4(), uuid4()
    user = AuthenticatedUser(id=owner_id)
    app.dependency_overrides[get_optional_current_user] = lambda: user

    class FakePortfolios:
        def __init__(self, client): pass
        async def get_for_owner(self, requested_id, requested_owner):
            assert requested_id == portfolio_id and requested_owner == owner_id
            return type("Portfolio", (), {"id": portfolio_id, "tickers": ["AAA", "BBB"], "target_weights": {"AAA": .6, "BBB": .4}, "name": "Saved"})()

    class FakeRecommendations:
        def __init__(self, client): pass
        async def latest_for_portfolio(self, *args): return None

    class FakeTracking:
        def __init__(self, client): pass
        async def list_holdings(self, *args): return []
        async def get_settings(self, portfolio_id):
            return type("Settings", (), {"monthly_dca_amount": 0.0, "target_horizon_years": 5})()

    async def fake_client(): return object()
    monkeypatch.setattr("app.api.v1.endpoints.advisor.get_async_supabase_client", fake_client)
    monkeypatch.setattr("app.api.v1.endpoints.advisor.PortfolioRepository", FakePortfolios)
    monkeypatch.setattr("app.api.v1.endpoints.advisor.AdvisoryRecommendationRepository", FakeRecommendations)
    monkeypatch.setattr("app.api.v1.endpoints.advisor.PortfolioTrackingRepository", FakeTracking)
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/advisor/recommendations/preview", json={
                "market": market_snapshot.model_dump(mode="json"), "portfolio_id": str(portfolio_id),
            })
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["portfolio_name"] == "Saved"
        assert data["current_weights"] == {"AAA": 0.0, "BBB": 0.0, "CASH": 1.0}
        assert data["projections"]["initial_capital"] == 0.0
    finally:
        app.dependency_overrides.pop(get_optional_current_user, None)


def test_persisted_advisor_requires_authentication():
    with TestClient(app) as client:
        response = client.get(f"/api/v1/advisor/portfolios/{uuid4()}/recommendations/latest")
    assert response.status_code == 401


def test_portfolio_selector_collection_is_owner_scoped(monkeypatch):
    owner_id = uuid4()
    app.dependency_overrides[require_current_user] = lambda: AuthenticatedUser(id=owner_id)

    class FakePortfolios:
        def __init__(self, client): pass
        async def list_for_owner(self, requested_owner, limit=100):
            assert requested_owner == owner_id and limit == 100
            return []

    async def fake_client(): return object()
    monkeypatch.setattr("app.api.v1.endpoints.portfolio.get_async_supabase_client", fake_client)
    monkeypatch.setattr("app.api.v1.endpoints.portfolio.PortfolioRepository", FakePortfolios)
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/portfolios")
        assert response.status_code == 200, response.text
        assert response.json()["data"] == []
        assert response.json()["meta"]["execution_time_ms"] >= 0
    finally:
        app.dependency_overrides.pop(require_current_user, None)


def test_acknowledge_uses_owner_scoped_repository(monkeypatch):
    owner_id, recommendation_id = uuid4(), uuid4()
    user = AuthenticatedUser(id=owner_id)
    app.dependency_overrides[require_current_user] = lambda: user

    class FakeRecommendations:
        def __init__(self, client): pass
        async def acknowledge(self, record_id, requested_owner):
            assert record_id == recommendation_id and requested_owner == owner_id
            return type("Saved", (), {
                "model_dump": lambda self, **kwargs: {"id": str(recommendation_id), "status": "acknowledged"},
                "snapshot_timestamp": datetime.now(timezone.utc),
                "rationale": [],
            })()

    async def fake_client(): return object()
    monkeypatch.setattr("app.api.v1.endpoints.advisor.get_async_supabase_client", fake_client)
    monkeypatch.setattr("app.api.v1.endpoints.advisor.AdvisoryRecommendationRepository", FakeRecommendations)
    try:
        with TestClient(app) as client:
            response = client.post(f"/api/v1/advisor/recommendations/{recommendation_id}/acknowledge")
        assert response.status_code == 200, response.text
        assert response.json()["data"]["status"] == "acknowledged"
    finally:
        app.dependency_overrides.pop(require_current_user, None)


def test_formal_recommendation_persists_owner_scoped_result(monkeypatch, market_snapshot):
    owner_id, portfolio_id = uuid4(), uuid4()
    user = AuthenticatedUser(id=owner_id)
    app.dependency_overrides[require_current_user] = lambda: user
    portfolio = PortfolioConfig(
        id=portfolio_id, owner_id=owner_id, name="Persisted",
        tickers=["AAA", "BBB"], target_weights={"AAA": 0.6, "BBB": 0.4},
    )
    saved_records = []

    class FakePortfolios:
        def __init__(self, client): pass
        async def get_for_owner(self, requested_id, requested_owner):
            assert requested_id == portfolio_id and requested_owner == owner_id
            return portfolio

    class FakeRecommendations:
        def __init__(self, client): pass
        async def latest_for_portfolio(self, *args): return None
        async def save(self, item):
            assert item.owner_id == owner_id and item.portfolio_id == portfolio_id
            saved_records.append(item)
            return item

    class FakeTracking:
        def __init__(self, client): pass
        async def list_holdings(self, *args): return []
        async def get_settings(self, portfolio_id):
            return type("Settings", (), {"monthly_dca_amount": 0.0, "target_horizon_years": 5})()

    async def fake_client(): return object()
    monkeypatch.setattr("app.api.v1.endpoints.advisor.get_async_supabase_client", fake_client)
    monkeypatch.setattr("app.api.v1.endpoints.advisor.PortfolioRepository", FakePortfolios)
    monkeypatch.setattr("app.api.v1.endpoints.advisor.AdvisoryRecommendationRepository", FakeRecommendations)
    monkeypatch.setattr("app.api.v1.endpoints.advisor.PortfolioTrackingRepository", FakeTracking)
    try:
        with TestClient(app) as client:
            response = client.post(f"/api/v1/advisor/portfolios/{portfolio_id}/recommendations", json={
                "market": market_snapshot.model_dump(mode="json"),
                "policy": {"holding_period_days": 60, "minimum_holding_period_days": 30},
            })
        assert response.status_code == 200, response.text
        assert len(saved_records) == 1
        assert response.json()["data"]["status"] == "pending_review"
        assert response.json()["data"]["rationale"]
    finally:
        app.dependency_overrides.pop(require_current_user, None)
