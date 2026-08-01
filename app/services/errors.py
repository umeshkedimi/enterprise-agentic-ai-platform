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


class UnsafeServerUrlError(DomainError):
    """A tenant-supplied URL the platform will not dial (→ 400).

    The tenant's configuration mistake, or their attempt at one: the platform
    fetches this URL itself, from inside its own network, so a host resolving to
    a loopback or link-local address is refused rather than tried.
    """


class CredentialStorageUnavailableError(DomainError):
    """No encryption key is configured, so a credential cannot be stored (→ 503).

    The operator's unfinished wiring, like `ProviderNotConfiguredError`. The
    request is refused rather than downgraded to storing the secret in plaintext.
    """


class EvaluationNotApplicableError(DomainError):
    """This turn cannot be judged for groundedness (→ 409).

    A user turn has no claims to check, and an agent with no knowledge scope was
    never grounded in documents to begin with — its answer may be perfectly good
    and this rubric still has nothing to say about it. Refused rather than scored
    zero: a fabricated failing score in the audit trail is worse than a gap,
    because someone will chart it.
    """


class EvaluationFailedError(DomainError):
    """The judge ran but its verdict could not be read (→ 502).

    Kept apart from `ModelInvocationError` because the provider did its job. What
    failed is the platform's contract with the judge — a response that is not the
    documented JSON — and the fix is a prompt or a model, not a retry against a
    sick upstream.
    """


class RetrievalError(DomainError):
    """Evidence could not be gathered for a grounded answer (→ 502).

    Almost always the embedding provider: a question has to be embedded before
    it can be searched for. Distinct from an empty result, which is a legitimate
    outcome the agent answers from — this is the search never having happened,
    and answering anyway would silently downgrade a grounded assistant into an
    ungrounded one at exactly the moment nobody is watching.
    """
