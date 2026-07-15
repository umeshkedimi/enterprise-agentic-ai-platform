from functools import lru_cache

from openai import AsyncAzureOpenAI, AsyncOpenAI

from app.core.config import Settings, get_settings


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
