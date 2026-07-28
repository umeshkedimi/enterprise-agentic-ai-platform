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


class AgentDisabledError(DomainError):
    """The agent exists but is switched off, so it refuses to run (→ 409)."""


class ModelConfigurationError(DomainError):
    """The agent's `model` string does not route to a known provider (→ 400).

    This is the tenant's configuration mistake, not a platform fault: the model
    is stored as free text so a new model needs no code change, which means a
    typo is only detectable at invocation time.
    """


class ProviderNotConfiguredError(DomainError):
    """The model routes to a provider the platform has no credentials for (→ 503).

    Distinct from ModelConfigurationError: the agent is configured correctly and
    the operator has not finished wiring the platform, so the tenant can do
    nothing about it.
    """


class ModelInvocationError(DomainError):
    """The upstream provider failed to complete the request (→ 502)."""
