"""Small guard for outbound requests to fixed third-party HTTPS APIs."""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlsplit
import urllib.request


def open_allowlisted_https(
    url: str,
    *,
    allowed_hosts: Iterable[str],
    timeout: float,
    headers: dict[str, str] | None = None,
):
    """Open HTTPS URLs only when their host is explicitly allowlisted.

    Callers must pass service-owned URLs or URLs derived from trusted service
    responses. Restricting the scheme, authority, and port prevents urllib
    from opening local files, custom schemes, or arbitrary destinations.
    """
    parsed = urlsplit(url)
    permitted_hosts = {host.lower() for host in allowed_hosts}
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname is None
        or parsed.hostname.lower() not in permitted_hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.fragment
    ):
        raise ValueError("outbound URL must use HTTPS with an allowlisted host")

    request = urllib.request.Request(url, headers=headers or {})
    # Bandit B310 is addressed by the strict HTTPS/host/port validation above.
    return urllib.request.urlopen(request, timeout=timeout)  # nosec B310
