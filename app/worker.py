"""Standalone entrypoint for the ingestion worker process.

Deployed as its own process — `docker-compose.yaml`'s `worker` service runs
this module as its command, on the same image the web app uses. Never
imported by `app.main`, and `app.main` is never imported by this: the two
processes share models and services, not a runtime.

    uv run python -m app.worker

Tracing and logging are configured here independently of the FastAPI app's
own lifespan, for the same reason the app configures its own rather than
inheriting anything at import time — a worker process has no lifespan to
hook into, and importing a module must never open a socket as a side effect.
"""

import asyncio

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.tracing import configure_tracing, shutdown_tracing
from app.db.session import async_session_factory, engine
from app.services.ingestion_worker import run_worker

logger = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_tracing(settings)
    configure_logging(settings.log_level)
    logger.info(
        "ingestion_worker_startup",
        poll_interval_seconds=settings.ingestion_poll_interval_seconds,
    )

    try:
        await run_worker(
            async_session_factory,
            poll_interval_seconds=settings.ingestion_poll_interval_seconds,
        )
    finally:
        # Same discipline `app.main`'s lifespan applies on shutdown: an
        # undisposed engine leaks pooled connections across every restart,
        # and a batch span processor that is never flushed drops whatever it
        # was still holding.
        await engine.dispose()
        shutdown_tracing()
        logger.info("ingestion_worker_shutdown")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
