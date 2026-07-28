# Enterprise Agentic AI Platform

An internal platform for building, configuring, and operating AI agents across an organisation.
Rather than each team hand-rolling its own assistant, a team registers an agent configuration —
prompt, model, tool allowlist, knowledge scope — and the platform supplies the shared runtime:
retrieval, orchestration, conversation state, streaming, and observability.

Built on FastAPI, PostgreSQL/pgvector, and Redis.

## Status

Under active development. This README documents **only what is implemented**; the roadmap below is
explicitly marked as not-yet-built.

**Implemented**

- Multi-tenancy & identity — tenants as a first-class boundary; callers authenticate with an API
  key (bearer token, stored as a SHA-256 hash) that resolves to their tenant. Tenant provisioning
  and key minting are gated behind a platform admin token that fails closed.
- Agent configuration domain — agents and knowledge collections as first-class, tenant-scoped
  database entities. An agent is a row (system prompt, model, tool allowlist, collection scope, and
  execution policy: temperature, max output tokens, retrieval top-k), so onboarding an assistant is
  an authenticated `POST`, not a code change.
- Document ingestion pipeline — upload into a collection, text extraction (PDF/txt/markdown),
  token-based chunking, batched embedding with retry, persistence to pgvector.
- Semantic retrieval — cosine similarity search over chunks, scoped by collection and document
  status, so one team's knowledge never surfaces in another's results.
- FastAPI application — app factory, lifespan-managed resources, request correlation IDs,
  structured JSON logging, liveness/readiness probes.
- Postgres + pgvector and Redis via Docker Compose; Alembic migrations; multi-stage app image.

**Roadmap (not yet implemented)**

LangGraph orchestration · tool registry and MCP integration · conversation persistence and memory ·
SSE streaming · LiteLLM multi-provider routing · federated auth (OIDC/SSO) · async ingestion via
Celery/RabbitMQ · OpenTelemetry tracing and Prometheus metrics · evaluation (groundedness,
confidence calibration) · Kubernetes manifests · CI/CD.

## Architecture

| Layer | Package | Responsibility |
|---|---|---|
| API | `app/api` | FastAPI routers, request/response DTOs, HTTP error mapping |
| Services | `app/services` | Tenancy, agent/collection config, ingestion, retrieval — framework-agnostic |
| Models | `app/models` | SQLModel tables (persistence) + Pydantic DTOs (transport) |
| Core | `app/core` | Settings, structured logging, middleware, LLM client factory |
| DB | `app/db` | Engine, session factory, Alembic metadata target |
| Agents | `app/agents` | *(empty — LangGraph orchestration, roadmap)* |
| Tools | `app/tools` | *(empty — tool registry and MCP, roadmap)* |

Services never import from `app/api`, so the same logic is reachable from an HTTP handler, a
background worker, or a test without dragging in the web framework.

## Requirements

- Python 3.12 (managed via [uv](https://docs.astral.sh/uv/))
- Docker + Docker Compose
- An OpenAI API key (or Azure OpenAI credentials)

## Setup

```bash
cp .env.example .env   # fill in OPENAI_API_KEY
uv sync

docker compose up -d postgres redis
uv run alembic upgrade head

uv run uvicorn app.main:app --reload
```

Interactive API docs at `http://localhost:8000/docs` (disabled when `APP_ENV=production`).

## Testing

```bash
uv run pytest tests/unit -v
docker compose up -d postgres redis
uv run pytest tests/integration -v
```

## API

Tenant-scoped routes authenticate with `Authorization: Bearer <api-key>`; admin routes require the
`X-Admin-Token` header.

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | — | Liveness — process is up; checks no dependencies |
| GET | `/health/ready` | — | Readiness — verifies database connectivity |
| POST | `/tenants` | admin | Provision a tenant |
| POST | `/tenants/{id}/keys` | admin | Mint an API key (plaintext returned once) |
| GET | `/tenants/me` | tenant | Resolve the calling tenant from its key |
| POST | `/collections` | tenant | Create a knowledge collection |
| GET | `/collections` | tenant | List the tenant's collections |
| GET/DELETE | `/collections/{id}` | tenant | Fetch or delete a collection |
| POST | `/agents` | tenant | Register an agent (config-as-data) |
| GET | `/agents` | tenant | List the tenant's agents |
| GET/PATCH/DELETE | `/agents/{id}` | tenant | Fetch, update, or delete an agent |
| POST | `/collections/{id}/documents` | tenant | Upload a pdf/txt/markdown document |
| GET | `/collections/{id}/documents` | tenant | List documents with chunk counts |
| DELETE | `/documents/{id}` | tenant | Delete a document and its chunks |

Request/response contracts: `app/models/schemas.py`.
