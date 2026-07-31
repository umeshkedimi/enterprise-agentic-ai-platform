from fastapi import APIRouter, HTTPException, Response, status

from app.core import metrics
from app.core.config import get_settings

router = APIRouter(tags=["operations"])


@router.get("/metrics", include_in_schema=False)
async def scrape() -> Response:
    """The Prometheus scrape endpoint.

    Unauthenticated, and that is a consequence of a decision made in
    `app/core/metrics.py` rather than an oversight: no metric here carries a
    tenant, agent, or conversation label, so the body describes the platform's
    health and nothing about who is using it. Had tenant labels been allowed,
    this endpoint would have needed authentication that Prometheus is awkward at
    supplying — the cardinality rule and the exposure rule are the same rule.

    It stays off the OpenAPI schema because it is an operator surface, not part
    of the API a team owner integrates against, and it can be switched off
    entirely for a deployment that collects metrics some other way.
    """
    if not get_settings().metrics_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")

    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)
