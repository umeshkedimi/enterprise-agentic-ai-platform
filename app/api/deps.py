import secrets

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db_session
from app.models.tenant import Tenant
from app.services import tenant_service

# Sent by callers as `Authorization: Bearer <key>`. 401 (not 403) on failure:
# the caller has not proven who they are, so the correct signal is
# "authenticate", not "you are known but forbidden".
_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Missing or invalid API key.",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_tenant(
    authorization: str | None = Header(default=None),
    session: AsyncSession = Depends(get_db_session),
) -> Tenant:
    """Resolve the calling tenant from its bearer API key, or reject with 401.

    Every tenant-scoped endpoint depends on this: it is the single place a
    request acquires its identity, replacing the former hardcoded "default"
    tenant. Swapping the bearer scheme for OIDC/SSO later is a change here
    alone, not at every call site.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise _UNAUTHENTICATED

    presented_key = authorization[len("bearer ") :].strip()
    tenant = await tenant_service.authenticate(session, presented_key=presented_key)
    if tenant is None:
        raise _UNAUTHENTICATED
    return tenant


def require_platform_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Gate control-plane provisioning behind the platform admin token.

    Fails closed: if no admin token is configured, every admin call is refused
    rather than allowed. Comparison is constant-time so a timing side channel
    cannot be used to recover the token character by character.
    """
    configured = get_settings().platform_admin_token
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Platform admin operations are disabled (no admin token configured).",
        )
    if not x_admin_token or not secrets.compare_digest(x_admin_token, configured):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid platform admin token.",
        )
