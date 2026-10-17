"""Supabase repository contract tests using an in-memory fluent mock."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from app.models.advisory_recommendation import AdvisoryRecommendation
from app.repositories.advisory_recommendations import AdvisoryRecommendationRepository


class Query:
    def __init__(self, rows):
        self.rows = rows
        self.filters = []
        self.payload = None
        self.operation = None

    def select(self, *_): self.operation = "select"; return self
    def insert(self, payload): self.operation = "insert"; self.payload = payload; return self
    def update(self, payload): self.operation = "update"; self.payload = payload; return self
    def eq(self, key, value): self.filters.append((key, value)); return self
    def order(self, *_args, **_kwargs): return self
    def limit(self, *_): return self
    async def execute(self): return SimpleNamespace(error=None, data=self.rows)


class Client:
    def __init__(self, rows): self.rows = rows; self.query = None
    def table(self, _name): self.query = Query(self.rows); return self.query


def recommendation_record(owner_id, portfolio_id):
    item = AdvisoryRecommendation(
        owner_id=owner_id, portfolio_id=portfolio_id, horizon_days=90,
        minimum_holding_period_days=30, snapshot_timestamp="2025-08-15T00:00:00Z",
        recommendation={"target_weights": {"AAA": 1.0}}, rationale=[], telemetry={},
    )
    return item.to_record()


def test_save_uses_database_response_and_latest_query_is_owner_scoped():
    owner_id, portfolio_id = uuid4(), uuid4()
    row = recommendation_record(owner_id, portfolio_id)
    client = Client([row])
    repo = AdvisoryRecommendationRepository(client)
    saved = asyncio.run(repo.save(AdvisoryRecommendation.model_validate(row)))
    assert saved.id == AdvisoryRecommendation.model_validate(row).id
    assert client.query.operation == "insert"

    latest = asyncio.run(repo.latest_for_portfolio(portfolio_id, owner_id))
    assert latest is not None and latest.owner_id == owner_id
    assert ("portfolio_id", str(portfolio_id)) in client.query.filters
    assert ("owner_id", str(owner_id)) in client.query.filters


def test_acknowledgement_scopes_update_to_owner_and_maps_response():
    owner_id, portfolio_id = uuid4(), uuid4()
    row = recommendation_record(owner_id, portfolio_id)
    row["status"] = "acknowledged"
    row["acknowledged_at"] = "2025-08-15T00:00:00Z"
    client = Client([row])
    repo = AdvisoryRecommendationRepository(client)
    result = asyncio.run(repo.acknowledge(row["id"], owner_id))
    assert result is not None and result.status == "acknowledged"
    assert ("id", row["id"]) in client.query.filters
    assert ("owner_id", str(owner_id)) in client.query.filters
    assert client.query.payload["status"] == "acknowledged"
