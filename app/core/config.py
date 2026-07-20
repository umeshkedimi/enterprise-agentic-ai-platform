from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    app_port: int = 8000
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://eaap:eaap@localhost:5432/eaap"
    database_url_sync: str = "postgresql+psycopg://eaap:eaap@localhost:5432/eaap"

    redis_url: str = "redis://localhost:6379/0"

    llm_provider: Literal["openai", "azure"] = "openai"

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = ""
    azure_openai_embedding_deployment: str = ""

    history_window: int = 20
    history_ttl_seconds: int = 86400

    agent_max_retries: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
