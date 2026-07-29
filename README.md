# Enterprise Agentic AI Platform

An internal platform for building, configuring, and operating AI agents across an organisation.
Rather than each team hand-rolling its own assistant, a team registers an agent configuration —
prompt, model, tool allowlist, knowledge scope — and the platform owns the shared runtime once.

**Onboarding a new assistant is a configuration change, not a deployment.** Two agents diverge in
behaviour entirely through the values in their rows: no branch in the code, no redeploy.

Built on FastAPI, PostgreSQL/pgvector, and Redis. Today the shared runtime covers tenant isolation,
knowledge scoping, ingestion, retrieval, multi-provider model routing, orchestration, conversation
memory, and streaming; MCP, observability, and evaluation are on the roadmap below.

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
- Retrieval-grounded chat — one LangGraph workflow serves every agent, branching on configuration
  rather than identity: an agent with a collection routes through retrieval, one without goes
  straight to generation. Answers come back with citations to the chunks they were grounded in.
  Retrieved text is placed in the user turn, never the system prompt, so an uploaded document
  cannot issue instructions; a failed retrieval fails the turn instead of silently answering
  ungrounded.
- Tool execution — a registry of platform tools (`search_knowledge_base`, `list_documents`), each
  granted per agent through its `tool_allowlist`, which defaults to empty. The model can search
  again with better terms once it has seen the first result, and what it finds is citable. The
  loop is bounded, the allowlist is enforced at execution rather than only by omitting the schema,
  and tool arguments are filtered to the declared parameters — so a tool's scope comes from the
  agent row and there is no argument through which a prompt injection can widen it.
- Conversation memory — threads are stored server-side and scoped to a tenant and an agent. A
  request carries a `conversation_id` and nothing else about the past; the platform replays the
  thread itself, so a caller cannot fabricate an assistant turn it never received. The transcript
  keeps every turn with the provenance of the answer — citations, tools, model, tokens, latency —
  while a configurable window bounds how much of it is replayed to the model. LangGraph's Postgres
  checkpointer persists graph execution state alongside it, keyed by conversation.
- SSE streaming — `POST /agents/{id}/chat/stream` narrates the same turn as it happens: the
  conversation id, the answer in fragments, an event for each tool call as it starts, then a
  terminal frame carrying citations and token usage. Anything checkable before the first byte is
  still a real status code; only failures after the headers travel in band as an `error` frame.
- FastAPI application — app factory, lifespan-managed resources, request correlation IDs,
  structured JSON logging, liveness/readiness probes.
- Postgres + pgvector and Redis via Docker Compose; Alembic migrations; multi-stage app image.

**Roadmap (not yet implemented)**

MCP integration · federated auth (OIDC/SSO) · async ingestion via Celery/RabbitMQ · OpenTelemetry
tracing and Prometheus metrics · evaluation (groundedness, confidence calibration) · Kubernetes
manifests · CI/CD.

## Architecture

| Layer | Package | Responsibility |
|---|---|---|
| API | `app/api` | FastAPI routers, request/response DTOs, HTTP error mapping |
| Agents | `app/agents` | The LangGraph workflow — state/context split, nodes, grounding prompts, checkpointer |
| Tools | `app/tools` | Tool registry, per-agent capability resolution, built-in tools |
| Services | `app/services` | Tenancy, agent/collection config, ingestion, retrieval, conversations, completions — framework-agnostic |
| Models | `app/models` | SQLModel tables (persistence) + Pydantic DTOs (transport) |
| Core | `app/core` | Settings, structured logging, middleware, model routing and provider credentials |
| DB | `app/db` | Engine, session factory, Alembic metadata target |

Dependencies run one way: `api → agents → {services, tools} → models/core/db`. Neither `app/agents` nor
`app/services` imports from `app/api`, so the same logic is reachable from an HTTP handler, a
background worker, or a test without dragging in the web framework.

Within the graph, **state is what gets persisted and context is what does not**. State holds
serializable conversation data; live handles — the database session, resolved settings, the agent
row — travel in per-invocation context. With the checkpointer attached that split is load-bearing
rather than theoretical: state round-trips through Postgres between turns, so a live handle stored
next to it would come back dead, and a resumed conversation reads the agent's *current*
configuration rather than a copy frozen at its start.

Conversation data lives in two stores, and they are not redundant. `conversation_messages` is the
product record — tenant-scoped, queryable, and the only thing history is ever read from. The
checkpointer is the execution record, keyed by an opaque thread id, and is what makes a turn
resumable rather than restartable. Neither can do the other's job: a checkpoint cannot say which
tenant owns it, and a transcript cannot resume a half-finished tool loop.

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
  \"tool_allowlist\": [\"search_knowledge_base\", \"list_documents\"],
  \"temperature\": 0.2,
  \"max_output_tokens\": 1024
}" | jq -r .id)
```

Capability is granted, never inherited: omit `tool_allowlist` and the agent can call nothing, no
matter how many tools the platform registers later.

**Flow B — run.**

```bash
curl -sX POST $BASE/agents/$AGENT/chat -H "$AUTH" -H "$JSON" \
  -d '{"message":"How much paid leave do I get?"}'
```

```json
{
  "conversation_id": "…",
  "answer": "Full-time employees accrue 25 days of paid annual leave [1].",
  "citations": [
    { "document_id": "…", "chunk_id": "…", "snippet": "…accrue twenty-five days…", "score": 0.83 }
  ],
  "tools_used": ["search_knowledge_base"],
  "model": "claude-sonnet-5",
  "provider": "anthropic",
  "usage": { "prompt_tokens": 412, "completion_tokens": 96, "total_tokens": 508 },
  "latency_ms": 1183
}
```

Note what the request body cannot say: which collection to search, which model to answer with, what
the system prompt is. All of it is read from the agent row per request. So switching providers is a
`PATCH`, not a deployment — same endpoint, same code path, different vendor — and every caller picks
it up on their next question:

```bash
curl -sX PATCH $BASE/agents/$AGENT -H "$AUTH" -H "$JSON" -d '{"model":"gpt-4o-mini"}'
```

A follow-up passes the `conversation_id` back and nothing else. The platform replays the thread it
stored — there is no request shape that can supply history, which is what stops a caller inventing
an assistant turn the agent never gave:

```bash
CONVERSATION=$(curl -sX POST $BASE/agents/$AGENT/chat -H "$AUTH" -H "$JSON" \
  -d '{"message":"How much paid leave do I get?"}' | jq -r .conversation_id)

curl -sX POST $BASE/agents/$AGENT/chat -H "$AUTH" -H "$JSON" \
  -d "{\"message\":\"And for part-timers?\",\"conversation_id\":\"$CONVERSATION\"}"

curl -s $BASE/agents/$AGENT/conversations/$CONVERSATION/messages -H "$AUTH"
```

The same turn, streamed:

```bash
curl -N -sX POST $BASE/agents/$AGENT/chat/stream -H "$AUTH" -H "$JSON" \
  -d '{"message":"How much paid leave do I get?"}'
```

```
event: conversation
data: {"conversation_id": "…"}

event: token
data: {"text": "Full-time "}

event: tool
data: {"name": "search_knowledge_base"}

event: done
data: {"answer": "…", "citations": [...], "usage": {...}, "latency_ms": 1183}
```

> **Scope note.** `/agents/{id}/complete` remains as the unorchestrated path — one turn, the
> agent's model, no retrieval, no memory. It is useful for verifying a model or prompt change in
> isolation from retrieval quality; `/chat` is the runtime the platform is actually for.

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
| POST | `/agents/{id}/chat` | tenant | Run an orchestrated turn — retrieve, then answer with citations |
| POST | `/agents/{id}/chat/stream` | tenant | The same turn, streamed as server-sent events |
| POST | `/agents/{id}/conversations` | tenant | Open a thread explicitly (`/chat` opens one on demand) |
| GET | `/agents/{id}/conversations` | tenant | List the agent's threads, most recently active first |
| GET | `/agents/{id}/conversations/{cid}/messages` | tenant | The full transcript, with per-turn citations and usage |
| POST | `/agents/{id}/complete` | tenant | Run one turn on the agent's model, without retrieval |
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
- **The system prompt comes from configuration, never the request — and now neither is the
  history.** A client that supplies its own prior turns can fabricate assistant ones, and a forged
  turn is indistinguishable from a real one by the time it reaches the model. History is read from
  rows the platform wrote; unknown request fields are rejected rather than ignored, so a client
  still sending `history` is told rather than silently having it dropped.
- **A thread belongs to the agent that produced it.** Continuing one under a different agent is a
  404. That is a retrieval boundary, not tidiness: an HR agent's history holds passages an HR agent
  was shown, and replaying it into a finance agent's context moves that text across the collection
  boundary without any query ever crossing it.
- **The tool allowlist grants nothing by default.** Capability is named explicitly per agent
  instead of being inherited from whatever the platform happens to support. It is enforced again
  when a call is executed, not only by omitting the schema from the request — a model can emit a
  call it was never offered.
- **A tool's scope comes from the agent row, never from an argument.** Arguments are filtered to
  the tool's declared parameters, so there is no `collection_id` for an injected instruction to
  supply. Isolation cannot be argued out of.
- **Retrieved text goes in the user turn, never the system prompt.** Uploaded documents are data,
  not instructions; a chunk reading "ignore your previous instructions" must arrive as a quoted
  string rather than an elevated directive.
- **A failed retrieval fails the turn.** Answering anyway would quietly downgrade a grounded
  assistant to an ungrounded one at the moment nobody is watching.
- **Graph state is what gets persisted; context is what does not.** Live handles — session,
  settings, the agent row — travel in per-invocation context, so a checkpointed turn cannot
  resurrect a dead connection, and a resumed conversation reads the agent's current configuration
  rather than a copy frozen at its start.
- **Graph state is scratch for one turn.** With a checkpointer attached, an invocation *merges*
  into whatever the thread already holds, so every turn-scoped field is written explicitly at the
  start of a turn. Left to default, turn two would cite turn one's passages and start with its tool
  budget already spent.
- **Conversation memory degrades rather than blocks.** If the checkpointer cannot be reached,
  startup logs and continues: turns still run and history still replays from the transcript.
  Refusing to serve traffic because memory is unavailable trades a feature for an outage.
- **Streaming is an argument, not a second code path.** The same `complete()` call streams or does
  not; retries, tool-call extraction, usage accounting, and error mapping are shared. A parallel
  implementation is the one that quietly stops counting tokens.
- **Liveness checks nothing; readiness checks Postgres.** A liveness probe wired to the database
  turns a brief blip into a rolling restart of every replica.
- **Neither `app/agents` nor `app/services` imports from `app/api`.** They raise domain errors and
  the router maps them to status codes, which is what keeps the same logic callable from a worker
  or a test.
