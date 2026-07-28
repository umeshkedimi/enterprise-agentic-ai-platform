import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_tenant, require_platform_admin
from app.db.session import get_db_session
from app.models.schemas import (
    ApiKeyCreate,
    ApiKeyIssuedResponse,
    TenantCreate,
    TenantResponse,
)
from app.models.tenant import Tenant
from app.services import tenant_service
from app.services.tenant_service import SlugAlreadyExistsError

# Provisioning routes are admin-gated at the router level; identity routes below
# authenticate as a tenant instead. The split mirrors the control-plane vs.
# data-plane boundary: creating a tenant is an operator action, using one is not.
router = APIRouter(prefix="/tenants", tags=["tenants"])


@router.post(
    "",
    response_model=TenantResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_platform_admin)],
)
async def create_tenant(
    body: TenantCreate,
    session: AsyncSession = Depends(get_db_session),
) -> TenantResponse:
    try:
        tenant = await tenant_service.create_tenant(session, slug=body.slug, name=body.name)
    except SlugAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A tenant with slug '{body.slug}' already exists.",
        ) from exc
    return TenantResponse(
        id=tenant.id, slug=tenant.slug, name=tenant.name, created_at=tenant.created_at
    )


@router.post(
    "/{tenant_id}/keys",
    response_model=ApiKeyIssuedResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_platform_admin)],
)
async def issue_api_key(
    tenant_id: uuid.UUID,
    body: ApiKeyCreate,
    session: AsyncSession = Depends(get_db_session),
) -> ApiKeyIssuedResponse:
    if await session.get(Tenant, tenant_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found.")

    record, plaintext = await tenant_service.issue_api_key(
        session, tenant_id=tenant_id, name=body.name
    )
    # The only moment the plaintext key exists outside the caller's request.
    return ApiKeyIssuedResponse(
        id=record.id,
        name=record.name,
        key_prefix=record.key_prefix,
        api_key=plaintext,
        created_at=record.created_at,
    )


@router.get("/me", response_model=TenantResponse)
async def whoami(tenant: Tenant = Depends(get_current_tenant)) -> TenantResponse:
    """Echo the tenant resolved from the caller's API key — the simplest proof
    that a key authenticates, and a health check for the auth path itself."""
    return TenantResponse(
        id=tenant.id, slug=tenant.slug, name=tenant.name, created_at=tenant.created_at
    )
