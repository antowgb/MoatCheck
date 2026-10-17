import urllib.request

import pytest

from app.security.network import open_allowlisted_https


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://www.sec.gov/example",
        "https://attacker.invalid/example",
        "https://user:password@www.sec.gov/example",
        "https://www.sec.gov:444/example",
    ],
)
def test_outbound_request_rejects_non_allowlisted_urls(url):
    with pytest.raises(ValueError):
        open_allowlisted_https(
            url,
            allowed_hosts={"www.sec.gov"},
            timeout=1,
        )


def test_outbound_request_uses_allowlisted_https_and_passes_headers(monkeypatch):
    captured = {}
    response = object()

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["headers"] = request.header_items()
        captured["timeout"] = timeout
        return response

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = open_allowlisted_https(
        "https://www.sec.gov/example",
        allowed_hosts={"www.sec.gov"},
        headers={"User-Agent": "MoatCheck test"},
        timeout=3,
    )

    assert result is response
    assert captured == {
        "url": "https://www.sec.gov/example",
        "headers": [("User-agent", "MoatCheck test")],
        "timeout": 3,
    }
