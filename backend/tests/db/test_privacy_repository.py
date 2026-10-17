import asyncio
from uuid import uuid4

from app.repositories.privacy import PrivacyRepository


class FakeQuery:
    def __init__(self, response):
        self.response = response
        self.params = None

    async def execute(self):
        return self.response


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.function = None
        self.params = None

    def rpc(self, function, params):
        self.function = function
        self.params = params
        return FakeQuery(self.response)


def test_privacy_repository_calls_atomic_owner_scoped_function():
    owner = uuid4()
    client = FakeClient(type("Response", (), {"data": {
        "portfolios_deleted": 1,
        "backtest_runs_deleted": 2,
        "stress_tests_deleted": 3,
    }, "error": None})())

    deleted = asyncio.run(PrivacyRepository(client).delete_owner_data(owner))

    assert deleted == {
        "portfolios_deleted": 1,
        "backtest_runs_deleted": 2,
        "stress_tests_deleted": 3,
    }
    assert client.function == "delete_owner_quant_data"
    assert client.params == {"p_owner_id": str(owner)}
