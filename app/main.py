from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.agents.checkpointer import start_checkpointer, stop_checkpointer
from app.api import (
    agents,
    collections,
    documents,
    evaluations,
    health,
    mcp_servers,
    metrics,
    tenants,
)
from app.core import metrics as metrics_registry
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware
from app.core.tracing import configure_tracing, instrument_app, shutdown_tracing
from app.db.session import engine

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Own the lifecycle of process-wide resources.

    Startup work belongs here rather than at import time so that importing the
    app (in tests, in Alembic, in a CLI) never opens sockets as a side effect.
    On shutdown we dispose the engine explicitly: without it, a rolling
    Kubernetes deploy leaks Postgres connections from terminating pods until
    they time out server-side, and a large enough rollout can exhaust
    max_connections while the new pods are still trying to start.

    The graph's checkpointer is started here for the same reason and with the
    same discipline: it owns a connection pool, so it must not be opened by
    importing a module, and it must be closed on the way out.
    """
    settings = get_settings()
    # Tracing before logging, so the processor that stamps trace ids onto log
    # lines has a provider to read from by the time the first line is emitted.
    configure_tracing(settings)
    configure_logging(settings.log_level)
    metrics_registry.set_build_info(settings.app_version, settings.app_env)
    await start_checkpointer(settings)
    logger.info("application_startup", app_env=settings.app_env)

    yield

    await stop_checkpointer()
    await engine.dispose()
    # Last, and after the engine: the spans describing shutdown are worth
    # keeping, and a batch processor that is never flushed drops its final
    # window on every single deploy.
    shutdown_tracing()
    logger.info("application_shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory.

    A factory rather than a module-level singleton so tests can build an app
    with overridden settings, and so nothing is constructed merely by importing
    this module.
    """
    settings = settings or get_settings()
    is_production = settings.app_env == "production"

    app = FastAPI(
        title="Enterprise Agentic AI Platform",
        description="Platform for configuring, running, and operating AI agents across teams.",
        version=settings.app_version,
        lifespan=lifespan,
        # Interactive docs are invaluable locally and an information-disclosure
        # liability in production, where they hand an attacker a complete map of
        # the API surface. Off by default there; expose deliberately if needed.
        docs_url=None if is_production else "/docs",
        redoc_url=None if is_production else "/redoc",
        openapi_url=None if is_production else "/openapi.json",
    )

    app.add_middleware(RequestContextMiddleware)
    # Before the first request and not in the lifespan: this patches
    # `build_middleware_stack`, which Starlette has already called by the time a
    # lifespan handler runs. Instrumenting there appears to work and silently
    # produces orphan traces with no request span above them.
    instrument_app(app, engine)

    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(tenants.router)
    app.include_router(collections.router)
    app.include_router(agents.router)
    app.include_router(documents.router)
    app.include_router(mcp_servers.router)
    app.include_router(evaluations.router)

    _register_exception_handlers(app)

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # FastAPI's default 422 body is fine, but it bypasses our logging, so a
        # client integrating against the API sees failures we never record.
        logger.info("request_validation_failed", path=request.url.path, errors=exc.errors())
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Anything reaching here is a bug. Log the full traceback server-side,
        # return an opaque body: stack traces and driver errors leak schema,
        # file paths, and library versions to whoever is probing the API.
        logger.exception("unhandled_exception", path=request.url.path)
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
        )


app = create_app()
