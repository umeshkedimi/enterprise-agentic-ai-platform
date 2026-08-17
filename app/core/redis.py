"""Process-wide Redis client, opened once and shared.

One client rather than one per caller for the same reason the DB engine and the
checkpointer's connection pool are process-wide: redis-py already pools
connections internally, and building a second client per request would open a
second pool underneath a client nothing asked for.

Every caller that touches Redis treats it as a cache, not a store — a command
that fails degrades to the cache-miss path rather than raising. That handling
stays at the call site (`app/tools/mcp_tools.py`) on purpose, so a reader can
see what "Redis is down" actually costs without following a wrapper that hides
it. This module only hands back a client and reports whether it was reachable
at boot.
"""

import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client: Redis | None = None


def get_redis_client(settings: Settings) -> Redis:
    """The shared client, built on first use.

    Lazy rather than connected here: redis-py's pool makes its first real
    connection on the first command regardless, so importing this module — or
    calling it from a unit test that never issues a command — never touches a
    socket.
    """
    global _client
    if _client is None:
        _client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=settings.redis_socket_timeout,
            socket_timeout=settings.redis_socket_timeout,
        )
    return _client


async def start_redis(settings: Settings) -> None:
    """Log whether Redis is reachable at boot. Never a reason to refuse to boot.

    Purely informational — nothing here is stored for another caller to read.
    Every caller that actually depends on Redis already treats a failed command
    as a cache miss, so this only moves the discovery of an outage from the
    first warning under load to the startup log.
    """
    try:
        await get_redis_client(settings).ping()
    except Exception as exc:  # noqa: BLE001 - connection errors are many types
        logger.warning("redis_unavailable", error=type(exc).__name__)
        return
    logger.info("redis_started")


async def stop_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("redis_stopped")
