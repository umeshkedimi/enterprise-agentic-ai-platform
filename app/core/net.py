"""Validation for URLs the platform will fetch on a tenant's instruction.

An MCP server URL is supplied by a tenant and dialled by the platform, which
makes it a server-side request forgery primitive unless something says otherwise.
The platform sits inside a private network with an instance identity: a tenant who
registers `http://169.254.169.254/latest/meta-data/iam/` and grants an agent its
"tools" is asking the platform to read its own cloud credentials out and hand them
over as a tool result.

Blocking by address rather than by hostname is the only version of this that
works. A name resolves to whatever its owner wants, including `127.0.0.1`, so the
check has to happen on what the name resolves *to*, and it has to happen again at
connect time — DNS answers expire, and a host that was public at registration can
be re-pointed afterwards.
"""

import ipaddress
import socket
from urllib.parse import urlparse

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

ALLOWED_SCHEMES = ("https", "http")


class UnsafeUrlError(ValueError):
    """The URL is malformed, or resolves somewhere the platform must not dial."""


def _resolved_addresses(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"host '{host}' could not be resolved") from exc
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def _is_forbidden(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # `is_private` covers loopback, RFC1918, and unique-local. The rest are
    # called out because they are not private but are equally not somewhere a
    # tenant's configuration should be able to send us: link-local is where cloud
    # metadata services live.
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def validate_outbound_url(url: str, settings: Settings) -> None:
    """Raise `UnsafeUrlError` unless the platform may fetch this URL.

    `mcp_allow_private_addresses` exists for local development, where the whole
    point is an MCP server on localhost. It defaults to off, so a deployment has
    to opt into the weaker behaviour explicitly rather than inherit it.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"scheme '{parsed.scheme}' is not supported")
    if not parsed.hostname:
        raise UnsafeUrlError("URL has no host")

    if settings.mcp_allow_private_addresses:
        return

    for address in _resolved_addresses(parsed.hostname):
        if _is_forbidden(address):
            logger.warning(
                "outbound_url_blocked", host=parsed.hostname, address=str(address)
            )
            raise UnsafeUrlError(
                f"host '{parsed.hostname}' resolves to a non-public address and will not be "
                "contacted"
            )
