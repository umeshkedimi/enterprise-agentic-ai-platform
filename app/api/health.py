from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.session import get_db_session

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def liveness() -> dict[str, str]:
    """Liveness probe: is this process alive and able to serve?

    Deliberately checks no downstream dependency. A liveness failure tells the
    orchestrator to *restart the pod*, so wiring it to Postgres would turn a
    brief database blip into a rolling restart of every replica — losing the
    in-flight requests that were still fine, and hammering the database with
    reconnects exactly when it is least able to cope.
    """
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(
    response: Response,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, object]:
    """Readiness probe: can this process serve traffic *right now*?

    A readiness failure only pulls the pod out of the load-balancer rotation and
    lets it recover in place, so checking dependencies here is safe and useful.
    """
    checks: dict[str, str] = {}

    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001 - probe reports, never raises
        checks["database"] = "error"
        logger.warning("readiness_check_failed", dependency="database", error=str(exc))

    ready = all(value == "ok" for value in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {"status": "ready" if ready else "not_ready", "checks": checks}
