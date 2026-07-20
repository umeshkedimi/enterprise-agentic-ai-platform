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

- Document ingestion pipeline — upload, text extraction (PDF/txt/markdown), token-based chunking,
  batched embedding with retry, persistence to pgvector.
- Semantic retrieval — cosine similarity search over chunks, scoped by tenant and document status.
- FastAPI application — app factory, lifespan-managed resources, request correlation IDs,
  structured JSON logging, liveness/readiness probes, document CRUD endpoints.
- Postgres + pgvector and Redis via Docker Compose; Alembic migrations; multi-stage app image.

**Roadmap (not yet implemented)**

Agent configuration domain (agents as first-class, tenant-scoped database entities) · knowledge
collections · LangGraph orchestration · tool registry and MCP integration · conversation
persistence and memory · SSE streaming · LiteLLM multi-provider routing · authentication ·
async ingestion via Celery/RabbitMQ · OpenTelemetry tracing · Kubernetes manifests · CI/CD.

## Architecture

| Layer | Package | Responsibility |
|---|---|---|
| API | `app/api` | FastAPI routers, request/response DTOs, HTTP error mapping |
| Services | `app/services` | Ingestion, chunking, embedding, retrieval — framework-agnostic |
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

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness — process is up; checks no dependencies |
| GET | `/health/ready` | Readiness — verifies database connectivity |
| POST | `/documents/upload` | Upload a pdf/txt/markdown document for retrieval |
| GET | `/documents` | List uploaded documents with chunk counts |
| DELETE | `/documents/{id}` | Delete a document and its chunks |

Request/response contracts: `app/models/schemas.py`.
