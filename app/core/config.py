from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    app_port: int = 8000
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # --- Observability ---
    # `/metrics` is on by default because a platform that cannot be scraped is
    # not operable; it is switchable for a deployment that collects some other
    # way. Nothing it exports is tenant-scoped, which is what lets it stay
    # unauthenticated — see the label rules in `app/core/metrics.py`.
    metrics_enabled: bool = True

    # Tracing is off until an operator names a collector. The OpenTelemetry API
    # is safe to call with no SDK configured — spans become non-recording
    # objects — so instrumentation stays in the code paths unconditionally and
    # costs almost nothing when there is nowhere to send it. A checkout with no
    # collector running must still boot.
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "enterprise-agentic-ai-platform"
    # Head sampling, applied per trace. One turn is one trace, and a trace that
    # is sampled keeps all of its spans — so this is a knob on cost, not on
    # completeness.
    otel_traces_sample_ratio: float = 1.0

    database_url: str = "postgresql+asyncpg://eaap:eaap@localhost:5432/eaap"
    database_url_sync: str = "postgresql+psycopg://eaap:eaap@localhost:5432/eaap"

    redis_url: str = "redis://localhost:6379/0"

    # Gates tenant provisioning (creating tenants, minting API keys) — a
    # privileged control-plane operation, distinct from a tenant using the
    # platform. Empty by default so admin endpoints fail closed until an
    # operator sets it: an unset secret must never mean "no auth required".
    platform_admin_token: str = ""

    # Fernet key for credentials the platform has to replay rather than verify —
    # today, MCP server tokens. Empty by default and fails closed: registering a
    # server with a credential is refused until an operator sets it, rather than
    # the token being written in the clear. Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    credential_encryption_key: str = ""

    # Whether a tenant-registered MCP URL may resolve to a private, loopback, or
    # link-local address. Off by default because the platform dials these URLs
    # itself, from inside the network, holding its own cloud identity. Local
    # development turns it on to reach an MCP server on localhost.
    mcp_allow_private_addresses: bool = False
    # How long a discovered tool list is reused before the server is asked again.
    # Discovery is a round trip per server per model call, and a tool list is not
    # something that changes between two turns of one conversation.
    mcp_tool_cache_ttl_seconds: int = 300

    llm_provider: Literal["openai", "azure"] = "openai"

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Chat credentials are per-provider because the chat model is per-agent
    # config: two agents in the same tenant may resolve to different providers
    # on the same request path. Embeddings deliberately do not read this — the
    # embedding provider is pinned to OpenAI platform-wide.
    anthropic_api_key: str = ""

    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = ""
    azure_openai_embedding_deployment: str = ""

    # How many messages of a thread are replayed to the model. Bounds cost and
    # context length; it does not bound what is stored — the transcript keeps
    # every turn and the API serves all of it.
    history_window: int = 20
    history_ttl_seconds: int = 86400

    # The graph checkpointer's own connection pool. Separate from the SQLAlchemy
    # engine because LangGraph's saver speaks psycopg directly, so the two cannot
    # share connections even though they share a database.
    checkpointer_pool_size: int = 10
    checkpointer_open_timeout: float = 10.0

    agent_max_retries: int = 2

    # --- Evaluation ---
    # The judge is platform config, never per-agent: a groundedness score is only
    # comparable across agents if every agent was measured by the same judge, and
    # letting a tenant choose its own would let it choose a lenient one.
    #
    # Defaults onto OpenAI rather than onto the platform's default *chat* model,
    # which is Anthropic's. The embedding provider is pinned to OpenAI, so an
    # OpenAI key is the one credential a running platform is guaranteed to hold;
    # an Anthropic default would leave the evaluation harness returning 503 on a
    # perfectly valid deployment that never configured a second provider.
    #
    # A small model is a defensible judge here specifically because the rubric
    # asks it for binary judgements — does this sentence follow from that
    # paragraph — and never for a calibrated score. That was a design choice
    # about model reliability; this is the payoff. Point it at a larger model if
    # the audit trail warrants one.
    evaluation_judge_model: str = "gpt-4o-mini"
    # Room for a claim-by-claim breakdown of a long answer. A judgement truncated
    # mid-JSON is unparseable, which costs the whole evaluation.
    evaluation_judge_max_output_tokens: int = 2048
    # How many claims one answer may be broken into before the platform stops
    # reading them. A bound on what a judge can write into a row, not a rubric
    # rule — the judge is told nothing about it.
    evaluation_max_claims: int = 50
    evaluation_rationale_max_chars: int = 2000


@lru_cache
def get_settings() -> Settings:
    return Settings()
