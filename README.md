# Enterprise Knowledge Agent Platform

Production-grade internal enterprise knowledge assistant: retrieval-augmented chat, tool calling,
and a multi-agent LangGraph workflow, built on FastAPI, PostgreSQL/pgvector, and Redis.

> **Status**: Phase 1 (working core) in progress. See `docs/` / commit history for phase scope.

## Architecture

- **API layer** (`app/api`) — FastAPI routers, request/response DTOs.
- **Agent layer** (`app/agents`) — LangGraph state graph: Planner → Decision → Retrieval/Tool/Direct
  → Execution → Critic → Response, implemented as five cooperating agents communicating through
  shared graph state (`AgentState`).
- **Tools** (`app/tools`) — schema-validated, retryable, timeout-bounded tool implementations
  (knowledge search, calculator, current time; more in later phases).
- **Services** (`app/services`) — document ingestion/chunking/embedding, semantic retrieval,
  Redis-backed conversation memory.
- **Models** (`app/models`) — SQLModel tables + Pydantic DTOs.
- **Core** (`app/core`) — settings, structured logging, LLM provider client factory.

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

## Testing

```bash
uv run pytest tests/unit -v
docker compose up -d postgres redis
uv run pytest tests/integration -v
```

## API

| Method | Path                    | Description                          |
|--------|--------------------------|---------------------------------------|
| POST   | `/chat`                 | Run the agent graph on a user message |
| POST   | `/documents/upload`     | Upload a pdf/txt/md document for RAG  |
| GET    | `/documents`            | List uploaded documents               |
| DELETE | `/documents/{id}`       | Delete a document and its chunks      |

Full request/response contracts: see `app/models/schemas.py`.
