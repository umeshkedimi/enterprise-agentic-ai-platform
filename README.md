# Enterprise Agentic AI Platform

An internal platform for building, configuring, and operating AI agents across an organisation.
Rather than each team hand-rolling its own assistant, a team registers an agent configuration —
prompt, model, tool allowlist, knowledge scope — and the platform owns the shared runtime once.

**Onboarding a new assistant is a configuration change, not a deployment.** Two agents diverge in
behaviour entirely through the values in their rows: no branch in the code, no redeploy.

Built on FastAPI, PostgreSQL/pgvector, and Redis. Today the shared runtime covers tenant isolation,
knowledge scoping, ingestion, retrieval, and multi-provider model routing; orchestration,
conversation state, and streaming are on the roadmap below.

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
- Multi-model routing — an agent's `model` string is resolved to a provider at request time
  (OpenAI or Anthropic) through one LiteLLM-backed interface, with credentials held per provider.
  The platform adapts the agent's execution policy to what each model actually accepts: current
  Anthropic models reject `temperature`, so it is withheld and logged rather than 400-ing the
  request. Token usage is normalised across providers and returned with every completion.
- Document ingestion pipeline — upload into a collection, text extraction (PDF/txt/markdown),
  token-based chunking, batched embedding with retry, persistence to pgvector.
- Semantic retrieval — cosine similarity search over chunks, scoped by collection and document
  status, so one team's knowledge never surfaces in another's results.
- FastAPI application — app factory, lifespan-managed resources, request correlation IDs,
  structured JSON logging, liveness/readiness probes.
- Postgres + pgvector and Redis via Docker Compose; Alembic migrations; multi-stage app image.

**Roadmap (not yet implemented)**

LangGraph orchestration · tool registry and MCP integration · conversation persistence and memory ·
SSE streaming · retrieval-grounded chat endpoint · federated auth (OIDC/SSO) · async ingestion via
Celery/RabbitMQ · OpenTelemetry tracing and Prometheus metrics · evaluation (groundedness,
confidence calibration) · Kubernetes manifests · CI/CD.

## Architecture

| Layer | Package | Responsibility |
|---|---|---|
| API | `app/api` | FastAPI routers, request/response DTOs, HTTP error mapping |
| Services | `app/services` | Tenancy, agent/collection config, ingestion, retrieval, completions — framework-agnostic |
| Models | `app/models` | SQLModel tables (persistence) + Pydantic DTOs (transport) |
| Core | `app/core` | Settings, structured logging, middleware, model routing and provider credentials |
| DB | `app/db` | Engine, session factory, Alembic metadata target |
| Agents | `app/agents` | *(empty — LangGraph orchestration, roadmap)* |
| Tools | `app/tools` | *(empty — tool registry and MCP, roadmap)* |

Services never import from `app/api`, so the same logic is reachable from an HTTP handler, a
background worker, or a test without dragging in the web framework.

## Requirements

- Python 3.12 (managed via [uv](https://docs.astral.sh/uv/))
- Docker + Docker Compose
- An OpenAI API key (or Azure OpenAI credentials) — required: embeddings are pinned to OpenAI
- Optionally an Anthropic API key, to run agents configured with Claude models

## Setup

```bash
cp .env.example .env   # set OPENAI_API_KEY; ANTHROPIC_API_KEY to run Claude agents;
                       # PLATFORM_ADMIN_TOKEN to enable tenant provisioning
uv sync

docker compose up -d postgres redis
uv run alembic upgrade head

uv run uvicorn app.main:app --reload
```

Interactive API docs at `http://localhost:8000/docs` (disabled when `APP_ENV=production`).

## Walkthrough

Two flows, mirroring the two roles. A **team owner** configures an agent once; **end users** then
run it. Neither touches code.

**Flow A — configure (once).** The platform operator provisions a tenant; everything after that is
the team's own, authenticated with its key.

```bash
BASE=http://localhost:8000
ADMIN="X-Admin-Token: $PLATFORM_ADMIN_TOKEN"
JSON="content-type: application/json"

TENANT=$(curl -sX POST $BASE/tenants -H "$ADMIN" -H "$JSON" \
  -d '{"slug":"people-ops","name":"People Ops"}' | jq -r .id)

# The plaintext key is returned exactly once — only its hash is stored.
KEY=$(curl -sX POST $BASE/tenants/$TENANT/keys -H "$ADMIN" -H "$JSON" \
  -d '{"name":"local-dev"}' | jq -r .api_key)
AUTH="Authorization: Bearer $KEY"

# A collection is the knowledge boundary: an agent searches its own, never another's.
COLLECTION=$(curl -sX POST $BASE/collections -H "$AUTH" -H "$JSON" \
  -d '{"slug":"hr-policies","name":"HR Policies"}' | jq -r .id)

curl -sX POST $BASE/collections/$COLLECTION/documents -H "$AUTH" -F file=@handbook.pdf

# The agent is a row. This POST is the entire onboarding.
AGENT=$(curl -sX POST $BASE/agents -H "$AUTH" -H "$JSON" -d "{
  \"slug\": \"hr-assistant\",
  \"name\": \"HR Assistant\",
  \"system_prompt\": \"You answer HR policy questions. Cite the policy you rely on.\",
  \"model\": \"claude-sonnet-5\",
  \"collection_id\": \"$COLLECTION\",
  \"temperature\": 0.2,
  \"max_output_tokens\": 1024
}" | jq -r .id)
```

**Flow B — run.**

```bash
curl -sX POST $BASE/agents/$AGENT/complete -H "$AUTH" -H "$JSON" \
  -d '{"turns":[{"role":"user","content":"How much paid leave do I get?"}]}'
```

```json
{
  "text": "...",
  "model": "claude-sonnet-5",
  "provider": "anthropic",
  "usage": { "prompt_tokens": 412, "completion_tokens": 96, "total_tokens": 508 },
  "latency_ms": 1183
}
```

Switching providers is a `PATCH`, not a deployment — same endpoint, same code path, different
vendor. The platform reconciles the execution policy with the new model's capabilities on the next
request:

```bash
curl -sX PATCH $BASE/agents/$AGENT -H "$AUTH" -H "$JSON" -d '{"model":"gpt-4o-mini"}'
```

> **Scope note.** `/agents/{id}/complete` is stateless and does **not** yet consult the agent's
> collection — it exists to prove that stored configuration selects the model and provider at
> runtime. Retrieval is implemented and tested at the service layer; wiring it into a
> conversational, tool-capable endpoint is the next chunk of work (see the roadmap).

## Testing

```bash
uv run pytest tests/unit -v          # no external dependencies
docker compose up -d postgres redis
uv run pytest tests/integration -v   # requires Postgres
```

The suite never calls a model provider: completions are faked at the LiteLLM boundary and the
ingestion tests avoid the embedding API, so tests run offline and need no vendor credentials.

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
| POST | `/agents/{id}/complete` | tenant | Run one stateless turn on the agent's configured model |
| POST | `/collections/{id}/documents` | tenant | Upload a pdf/txt/markdown document |
| GET | `/collections/{id}/documents` | tenant | List documents with chunk counts |
| DELETE | `/documents/{id}` | tenant | Delete a document and its chunks |

Request/response contracts: `app/models/schemas.py`.

## Design decisions

- **Tenancy lives in foreign keys and queries, never in a prompt.** Every query touching
  tenant-owned data filters on `tenant_id`, and an agent can only reference a collection its own
  tenant owns. Asking a model to respect a boundary is not isolation.
- **Cross-tenant references return 404, not 403.** A 403 confirms that someone else's resource
  exists; not-found leaks nothing.
- **API keys are stored as SHA-256 hashes.** The plaintext is shown once at mint time and is
  unrecoverable afterwards; a short `key_prefix` is kept so a key can be identified without
  storing anything usable. A database dump yields no working credential.
- **The admin token fails closed.** Unset means tenant provisioning is disabled, never
  "authentication not required".
- **The embedding provider is pinned; the chat provider is not.** Changing the embedding model
  invalidates every stored vector and forces a full re-index, so it is a platform-wide constant.
  Chat models are per-agent config and vary freely.
- **Model capabilities are read from the provider map, never hardcoded.** Current Anthropic models
  removed sampling parameters, so sending `temperature` is a 400 — and the default agent sets one.
  The routing layer reads the per-model capability flag, withholds the parameter, and logs the
  omission. A newly released model becomes a dependency bump rather than a code change.
- **Incompatible model/policy combinations adapt at runtime rather than failing at create time.**
  Capabilities shift underneath a stored config; keeping a configured agent runnable is the
  platform's job.
- **The system prompt comes from configuration, never the request.** Callers may only supply
  `user` and `assistant` turns — enforced in the schema, so it is a boundary rather than a
  convention.
- **The tool allowlist grants nothing by default.** Capability is named explicitly per agent
  instead of being inherited from whatever the platform happens to support.
- **Liveness checks nothing; readiness checks Postgres.** A liveness probe wired to the database
  turns a brief blip into a rolling restart of every replica.
- **Services never import from `app/api`.** They raise domain errors and the router maps them to
  status codes, which is what keeps the same logic callable from a worker or a test.
