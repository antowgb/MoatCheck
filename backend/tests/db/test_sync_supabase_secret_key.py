from types import SimpleNamespace

from app.data import supabase_client


def test_sync_client_supports_new_server_secret_key_without_bearer(monkeypatch):
    client = SimpleNamespace(
        options=SimpleNamespace(headers={"Authorization": "Bearer sb_secret_test", "apikey": "sb_secret_test"}),
        auth=SimpleNamespace(_headers={"Authorization": "Bearer sb_secret_test"}),
    )
    calls = []

    def fake_create_client(url, key, options=None):
        calls.append((url, key, options))
        return client

    monkeypatch.setattr(supabase_client, "create_client", fake_create_client)
    monkeypatch.setenv("SUPABASE_URL", "https://unit-test.invalid")
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_test")
    supabase_client.get_supabase.cache_clear()
    try:
        result = supabase_client.get_supabase()
    finally:
        supabase_client.get_supabase.cache_clear()

    assert result is client
    assert calls[0][0:2] == ("https://unit-test.invalid", "sb_secret_test")
    assert client.options.headers == {"apikey": "sb_secret_test"}
    assert client.auth._headers == {}
