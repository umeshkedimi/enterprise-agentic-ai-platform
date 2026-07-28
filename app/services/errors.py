"""Domain errors raised by the service layer.

Services must stay callable from an HTTP handler, a worker, or a test without
importing the web framework, so they raise these instead of HTTPException. The
API layer maps each to a status code.
"""


class DomainError(Exception):
    """Base for all service-layer domain errors."""


class SlugAlreadyExistsError(DomainError):
    """A slug that must be unique within its scope already exists (→ 409)."""


class NotFoundError(DomainError):
    """A referenced resource does not exist in the caller's tenant (→ 404).

    Cross-tenant references are reported as not-found rather than forbidden: a
    tenant must not be able to probe for the existence of another tenant's
    resources by watching for 403 vs 404.
    """
