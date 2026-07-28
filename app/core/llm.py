"""Model routing and provider credentials.

Two distinct paths live here on purpose:

* **Embeddings** use the OpenAI SDK directly. The embedding provider is pinned
  platform-wide — changing it invalidates every stored vector — so there is
  nothing to route.
* **Chat** goes through LiteLLM, because the model is per-agent configuration
  and may resolve to a different provider on every request.

This module answers "where does this model string go, with what credentials,
and what will that model actually accept?" It deliberately raises no domain
errors and performs no I/O: `app/services` owns that translation.
"""

from dataclasses import dataclass
from functools import lru_cache

import litellm
from openai import AsyncAzureOpenAI, AsyncOpenAI

from app.core.config import Settings, get_settings

# LiteLLM phones home and prints provider banners by default. Both are noise in
# a service that emits structured logs, and the telemetry call adds latency to
# the first completion of every process.
litellm.telemetry = False
litellm.suppress_debug_info = True

# Per-provider credential lookup. Adding a provider is an entry here plus a
# settings field — no change to the agent model, the service, or the router.
_PROVIDER_CREDENTIALS: dict[str, str] = {
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
    "azure": "azure_openai_api_key",
}


class UnknownModelError(ValueError):
    """LiteLLM cannot map the model string to a provider."""


@dataclass(frozen=True)
class ModelRoute:
    """Everything needed to invoke one model, resolved from its name alone."""

    model: str
    provider: str
    # None when the platform holds no credential for this provider. Callers
    # decide whether that is fatal; resolving a route stays side-effect free.
    api_key: str | None
    # Current-generation Anthropic models (Sonnet 5, Opus 5/4.8, Fable 5)
    # removed sampling parameters: sending `temperature` returns a 400. An
    # agent's stored temperature therefore cannot be forwarded unconditionally.
    supports_sampling_params: bool


def _supports_sampling_params(model: str) -> bool:
    # LiteLLM's model map carries this per model. Reading the flag rather than
    # keeping our own list means a newly-released model is a dependency bump,
    # not a code change. Absent flag means the model still accepts sampling.
    flag = litellm.model_cost.get(model, {}).get("supports_sampling_params")
    return True if flag is None else bool(flag)


def resolve_model_route(model: str, settings: Settings | None = None) -> ModelRoute:
    """Map an agent's `model` string to a provider and its credentials.

    Raises UnknownModelError if the string routes nowhere — the model is stored
    as free text so that onboarding a new model stays a configuration change,
    which means validation can only happen here, at invocation time.
    """
    settings = settings or get_settings()
    try:
        _, provider, *_ = litellm.get_llm_provider(model=model)
    except Exception as exc:  # noqa: BLE001 - LiteLLM raises several unrelated types
        raise UnknownModelError(model) from exc

    credential_field = _PROVIDER_CREDENTIALS.get(provider)
    api_key = getattr(settings, credential_field, "") if credential_field else ""

    return ModelRoute(
        model=model,
        provider=provider,
        api_key=api_key or None,
        supports_sampling_params=_supports_sampling_params(model),
    )


@lru_cache
def get_llm_client() -> AsyncOpenAI | AsyncAzureOpenAI:
    settings = get_settings()
    if settings.llm_provider == "azure":
        return AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
    return AsyncOpenAI(api_key=settings.openai_api_key)


def get_chat_model(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if settings.llm_provider == "azure":
        return settings.azure_openai_chat_deployment
    return settings.openai_chat_model


def get_embedding_model(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if settings.llm_provider == "azure":
        return settings.azure_openai_embedding_deployment
    return settings.openai_embedding_model
