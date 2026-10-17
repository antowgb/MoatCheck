"""Integration checks for cursor pagination with a mocked database RPC."""

from types import SimpleNamespace
from datetime import date, timedelta

import pytest
from tests.asgi_client import TestClient

from app.api.v1.endpoints import screener as screener_module
from app.main import app


class FakeRPC:
    def __init__(self, response, tables=None):
        self.response = response
        self.calls = []
        self.tables = tables or {}

    def rpc(self, name, params):
        self.calls.append((name, params))
        return self

    async def execute(self):
        return SimpleNamespace(data=self.response)

    def table(self, name):
        return FakeTable(self.tables.get(name, []), name, self.calls)


class FakeTable:
    def __init__(self, response, name, calls):
        self.response = response
        self.name = name
        self.calls = calls

    def select(self, _columns):
        return self

    def in_(self, _column, _values):
        return self

    def neq(self, _column, _value):
        return self

    async def execute(self):
        self.calls.append(("table", self.name))
        return SimpleNamespace(data=self.response)


@pytest.fixture
def client(monkeypatch):
    rpc = FakeRPC([
        {"ticker": "AAA", "composite_score": 92.0},
        {"ticker": "BBB", "composite_score": 89.0},
        {"ticker": "CCC", "composite_score": 85.0},
    ])

    async def get_db():
        return rpc

    monkeypatch.setattr(screener_module, "get_async_supabase_client", get_db)
    with TestClient(app) as test_client:
        yield test_client, rpc


def test_screener_returns_bounded_page_and_cursor(client):
    test_client, rpc = client
    response = test_client.get("/api/v1/screener?limit=2&sort_by=composite_score")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert [row["ticker"] for row in payload["rows"]] == ["AAA", "BBB"]
    assert payload["has_more"] is True
    assert payload["page_size"] == 2
    assert payload["execution_time_ms"] >= 0
    assert rpc.calls[0][0] == "screener_page"
    assert rpc.calls[0][1]["p_limit"] == 3


def test_screener_page_includes_last_refresh_and_recent_qualitative_tally(monkeypatch):
    cutoff_fresh = (date.today() - timedelta(days=20)).isoformat()
    cutoff_old = (date.today() - timedelta(days=120)).isoformat()
    rpc = FakeRPC(
        [{"ticker": "LLY", "composite_score": 91.5}],
        {
            "stocks": [{"ticker": "LLY", "updated_at": "2026-10-16T22:30:00+00:00"}],
            "qualitative_notes": [
                {"ticker": "LLY", "sentiment": "positive", "confidence": "high", "event_date": cutoff_fresh, "created_at": None},
                {"ticker": "LLY", "sentiment": "negative", "confidence": "low", "event_date": cutoff_fresh, "created_at": None},
                {"ticker": "LLY", "sentiment": "negative", "confidence": "medium", "event_date": cutoff_old, "created_at": None},
                {"ticker": "LLY", "sentiment": "neutral", "confidence": "medium", "event_date": None, "created_at": f"{cutoff_fresh}T12:00:00Z"},
            ],
        },
    )

    async def get_db():
        return rpc

    monkeypatch.setattr(screener_module, "get_async_supabase_client", get_db)
    with TestClient(app) as test_client:
        response = test_client.get("/api/v1/screener?limit=1")
    assert response.status_code == 200, response.text
    row = response.json()["rows"][0]
    assert row["updated_at"] == "2026-10-16T22:30:00+00:00"
    assert row["qualitative_tally"] == {"positive": 1, "negative": 0, "neutral": 1, "window_days": 90}


def test_screener_rejects_cursor_reused_with_another_sort(client):
    test_client, _rpc = client
    first = test_client.get("/api/v1/screener?limit=2").json()
    cursor = first["next_cursor"]
    response = test_client.get(f"/api/v1/screener?limit=2&sort_by=market_cap&cursor={cursor}")
    assert response.status_code == 422


def test_maximum_public_page_size_uses_internal_lookahead_row(client):
    test_client, rpc = client
    response = test_client.get("/api/v1/screener?limit=200")
    assert response.status_code == 200, response.text
    screener_call = next(params for name, params in rpc.calls if name == "screener_page")
    assert screener_call["p_limit"] == 201


@pytest.mark.parametrize("payload", [[], None, 1, "cursor", {"value": True, "ticker": "AAA", "sort_by": "composite_score", "sort_dir": "desc"}])
def test_malformed_cursor_payloads_return_422_instead_of_500(client, payload):
    import base64
    import json

    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    test_client, rpc = client
    response = test_client.get("/api/v1/screener", params={"cursor": encoded})
    assert response.status_code == 422, response.text
    assert rpc.calls == []
