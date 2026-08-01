# Enterprise Agentic AI Platform

An internal platform for building, configuring, and operating AI agents across an organisation.
Rather than each team hand-rolling its own assistant, a team registers an agent configuration —
prompt, model, tool allowlist, knowledge scope — and the platform owns the shared runtime once.

**Onboarding a new assistant is a configuration change, not a deployment.** Two agents diverge in
behaviour entirely through the values in their rows: no branch in the code, no redeploy.

Built on FastAPI, PostgreSQL/pgvector, and Redis. Today the shared runtime covers tenant isolation,
knowledge scoping, ingestion, retrieval, multi-provider model routing, orchestration, conversation
memory, streaming, MCP tool integration, and observability; evaluation is on the roadmap below.

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
- MCP integrations — a team registers a remote MCP server (`POST /mcp-servers`) and its tools become
  grantable to that team's agents by name. Nothing in this repository knows what those tools do, so
  adding a capability is a `POST` rather than a release. Discovered tools become ordinary platform
  tools: the same allowlist, the same argument filtering, the same bounded loop. Remote HTTP
  transport only — no subprocess launching — tenant-supplied URLs are checked against an SSRF guard
  before every connection, and server credentials are encrypted at rest. A server that is
  unreachable costs its tools, not the turn.
- Observability — Prometheus metrics at `/metrics` and OpenTelemetry traces over OTLP, covering the
  units an operator actually reasons about: the turn, the model call, the vector search, the tool,
  the MCP round trip. Metric labels are restricted to values the platform itself chose, so no tenant
  can mint a time series and the scrape body carries nothing tenant-specific; the high-cardinality
  detail — which tenant, which agent, which remote tool — lives on spans and in logs, which are
  sampled and expire. Every log line carries the request id and, when tracing is on, the trace and
  span ids, so a slow turn in a dashboard leads to a trace and a trace leads to its logs. Tracing is
  off until an operator names a collector; the platform runs without one.
- Evaluation & audit trail — served answers are judged for groundedness against the evidence they
  were actually shown. The judge is never asked for a score: it enumerates the claims in an answer
  and marks each supported or not, and the platform does the arithmetic, so every number is
  reproducible from the stored breakdown and a team owner who disagrees with a 0.67 can see which
  third failed. Evaluation runs after the fact over the transcript, never inside a turn — an
  assistant whose judge is down still answers, and a scoring bug cannot take answers with it. An
  answer that correctly declines for lack of evidence is recorded as an abstention with no score,
  rather than as a perfect one, so a retriever that finds nothing cannot report flawless grounding.
  A calibration report then buckets retrieval scores against judged groundedness and will propose a
  relevance floor — including what that floor would cost in abstentions — but declines to propose
  one at all until there is enough data to mean it.
- FastAPI application — app factory, lifespan-managed resources, request correlation IDs,
  structured JSON logging, liveness/readiness probes.
- Postgres + pgvector and Redis via Docker Compose, with Prometheus, Grafana, and Jaeger behind an
  `observability` profile; Alembic migrations; multi-stage app image.

**Roadmap (not yet implemented)**

Federated auth (OIDC/SSO) · async ingestion via Celery/RabbitMQ · queued/scheduled evaluation runs
(the harness is synchronous today) · Kubernetes manifests · CI/CD.

## Architecture

| Layer | Package | Responsibility |
|---|---|---|
| API | `app/api` | FastAPI routers, request/response DTOs, HTTP error mapping |
| Agents | `app/agents` | The LangGraph workflow — state/context split, nodes, grounding prompts, checkpointer |
| Tools | `app/tools` | Tool registry, per-agent capability resolution, built-in tools, MCP client |
| Services | `app/services` | Tenancy, agent/collection config, ingestion, retrieval, conversations, completions, evaluation — framework-agnostic |
| Models | `app/models` | SQLModel tables (persistence) + Pydantic DTOs (transport) |
| Core | `app/core` | Settings, structured logging, middleware, metrics, tracing, model routing, credential encryption, outbound-URL safety |
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

### Observability

Three signals, with a strict division of labour between them.

**Metrics** (`/metrics`, Prometheus) answer *is the fleet healthy*. Every label is a value the
platform chose: a route template rather than a path, a provider and a model that LiteLLM's model map
has already vouched for, a domain-error class name, and `mcp` in place of any remote tool's name. No
metric is labelled by tenant, agent, conversation, or collection — those are strings somebody else
supplies, and each distinct one would be a time series that never goes away. That restraint is also
why `/metrics` needs no authentication: a body that has aggregated its subjects away has nothing
tenant-specific to leak.

**Traces** (OTLP → Jaeger) answer *where did this turn's time go*. A turn is one trace:
`agent.turn → agent.retrieve → agent.generate → agent.tools → agent.generate`, with the provider
call and every tool as children, plus SQLAlchemy statement spans underneath. Spans carry exactly the
detail metrics may not — tenant id, agent slug, conversation id, the real MCP tool name — because a
span is sampled and expires. What they never carry is prompt text, retrieved passages, or answers:
spans leave the process, and one that quoted the prompt would export a tenant's documents to a
third-party backend on every request.

**Logs** (structured JSON) answer *what exactly happened*. Every line carries the request id, and
when tracing is on, the trace and span ids — so a p99 spike leads to a trace and a trace leads to its
own log lines.

The stack runs locally behind a compose profile:

```bash
docker compose --profile observability up -d
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 uv run uvicorn app.main:app --reload
```

Grafana at `:3000` (anonymous, with the platform dashboard provisioned), Prometheus at `:9090`,
Jaeger at `:16686`.

One panel on that dashboard exists to settle an argument rather than to raise an alert. Retrieval
still has no relevance-score floor, because the right cosine threshold depends on the embedding model
and a badly-chosen one silently breaks retrieval — so the decision was deferred until it could be
measured. `eaap_retrieval_top_score` is the measurement: the distribution of best-match scores across
real traffic, bucketed densely exactly where a floor would plausibly sit.

### Evaluation

Answers are audited after the fact, over the transcript, never inside the turn that produced them.
A judge on the request path would roughly double the cost and latency of every answer, but the
reason it is not there is stronger than that: the two have to be able to fail independently. An
assistant whose evaluator is down still answers, and a scoring bug cannot take answers with it.

**The judge is never asked for a score.** A model asked to rate an answer out of ten returns a
number with no defensible relationship to anything, and asked whether an answer is good it returns
"yes". So it is asked the concrete question models are reliable on — does this sentence follow from
that paragraph — once per claim, and the platform divides. The consequences are worth the detour:
every score is reproducible from the stored claims, a reader who disagrees with a 0.67 can see which
third failed, and a small, cheap model becomes a defensible judge, which is what makes judging every
turn affordable at all.

**Evidence is re-read, not quoted from the citation.** A citation snippet is a 240-character prefix
of a chunk running to roughly 400 tokens, so a judge fed snippets would fail supported claims for
lack of evidence the answering model actually had. Full chunk text is recovered by `chunk_id`;
citations whose document has since been deleted fall back to the frozen snippet, which is the job
that snippet was stored for.

**An abstention is not a failure.** An agent shown nothing relevant that says so has done exactly
what a grounded assistant should, and it is recorded with a null score rather than a zero or a one.
A stand-in 1.0 would let a retriever that finds nothing report a flawless platform; a 0.0 would
train the platform's own metrics to punish the behaviour they exist to encourage.

```bash
# Judge one turn, or a whole thread. Repeating either is free — an evaluation
# already on file is returned rather than paid for again.
curl -X POST .../agents/$AGENT/conversations/$CONV/messages/$MSG/evaluations -H "$AUTH"
curl -X POST .../agents/$AGENT/conversations/$CONV/evaluations -H "$AUTH"

# What a retrieval score turned out to be worth.
curl .../evaluations/calibration -H "$AUTH"
```

That last endpoint is the other half of the deferred floor decision. Every evaluation row holds the
platform's own confidence in the best passage it retrieved next to an independent judgement of the
answer built on it; bucketed against each other, they answer *given that the best match scored 0.62,
how often was the answer grounded?* It will propose a floor — and report how many turns would have
become abstentions under it, which is the number a threshold read off a chart never comes with — but
it refuses to propose one from thin data, since a guess dressed as an analysis is worse than the
honest deferral it replaced.

## Requirements

- Python 3.12 (managed via [uv](https://docs.astral.sh/uv/))
- Docker + Docker Compose
- An OpenAI API key (or Azure OpenAI credentials) — required: embeddings are pinned to OpenAI
- Optionally an Anthropic API key, to run agents configured with Claude models
- Optionally a Fernet key (`CREDENTIAL_ENCRYPTION_KEY`), to register MCP servers that need a token

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

**Flow A′ — extend what the platform can do (optional).** A team can widen the set of grantable
tools without waiting on this repository, by pointing the platform at an MCP server:

```bash
SERVER=$(curl -sX POST $BASE/mcp-servers -H "$AUTH" -H "$JSON" -d '{
  "slug": "jira",
  "name": "Jira MCP",
  "url": "https://mcp.internal.example.com/jira",
  "auth_token": "…"
}' | jq -r .id)

# Ask the server what it offers, under the names an allowlist has to use.
curl -s $BASE/mcp-servers/$SERVER/tools -H "$AUTH" | jq '.tools[].name'
# → "jira__search_issues"
# → "jira__create_issue"

# Granting one is the same PATCH as granting a built-in.
curl -sX PATCH $BASE/agents/$AGENT -H "$AUTH" -H "$JSON" \
  -d '{"tool_allowlist":["search_knowledge_base","jira__search_issues"]}'
```

Tool names are namespaced by server slug, but that is legibility rather than security: the servers
consulted are the ones the *agent's own tenant* registered, so an allowlist entry naming another
tenant's server resolves to nothing. The `auth_token` is write-only — stored encrypted, replayed to
that server, and never returned by the API.

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
| GET | `/metrics` | — | Prometheus scrape (operator surface; off the OpenAPI schema) |
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
| POST | `/mcp-servers` | tenant | Register a remote MCP server (config-as-data) |
| GET | `/mcp-servers` | tenant | List the tenant's MCP servers |
| GET | `/mcp-servers/{id}/tools` | tenant | Discover its tools, under the names an allowlist uses |
| GET/PATCH/DELETE | `/mcp-servers/{id}` | tenant | Fetch, update, or remove a server |
| POST | `/agents/{id}/conversations/{cid}/messages/{mid}/evaluations` | tenant | Judge one turn for groundedness (idempotent; `refresh` re-judges) |
| GET | `/agents/{id}/conversations/{cid}/messages/{mid}/evaluations` | tenant | Judgements on file for a turn, with the claim-by-claim breakdown |
| POST | `/agents/{id}/conversations/{cid}/evaluations` | tenant | Judge every assistant turn in a thread |
| GET | `/agents/{id}/calibration` | tenant | What this agent's retrieval scores were worth |
| GET | `/evaluations/calibration` | tenant | The same reading across the tenant, with a floor recommendation |
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
- **An MCP tool is an ordinary tool.** Discovery produces the same `Tool` objects the built-ins
  register, so the allowlist, the argument filter, the step cap, and the citation logic are
  unchanged and the tool loop cannot tell a remote tool from a local one. MCP is a *source* of
  tools, not a second kind of tool.
- **Remote tool names are namespaced; tenancy is what makes that safe.** `jira__search_issues` is
  legible, but an allowlist entry naming another tenant's server resolves to nothing, because the
  servers consulted are the ones the agent's own tenant registered. The `__` separator does buy one
  guarantee outright: no built-in name contains it, so a hostile server cannot shadow
  `search_knowledge_base`.
- **A tool name a provider would reject is dropped, not sanitised.** MCP permits `weather.forecast`;
  OpenAI does not, and an illegal name fails the *whole* request rather than that one tool. The
  64-character ceiling is the strictest provider's, not the current one's — the model is a `PATCH`
  away from changing.
- **No stdio MCP transport.** Launching a server as a subprocess would make a tenant-editable
  command string arbitrary code execution on the platform's hosts. Remote HTTP only, so there is no
  field through which to ask.
- **Tenant-supplied URLs are checked by address, before every connection.** The platform dials these
  itself, from inside its own network, with its own cloud identity — so private, loopback,
  link-local, and reserved addresses are refused. Checking hostnames would be theatre: a name
  resolves to whatever its owner wants, and a host that was public at registration can be
  re-pointed afterwards.
- **Third-party credentials are encrypted, not hashed.** An API key the platform issues is only ever
  verified, so it is hashed. A credential for somebody else's server has to be replayed on every
  call, so it cannot be — it is encrypted under a key held outside the database, and with no key
  configured the platform refuses to store one rather than writing it in the clear.
- **A broken integration costs its tools, not the turn.** An unreachable MCP server contributes no
  tools and a logged reason; failures are cached like successes, so one outage does not add its
  timeout to every model call in the tenant.
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
- **A metric label may only take values the platform chose.** Tenant ids, agent ids, conversation
  ids, and MCP tool names are all typed in by somebody else; each distinct value would be a time
  series that outlives whoever created it, so a single tenant could degrade monitoring for everyone.
  Route templates, providers, models, and domain-error classes are bounded by the code, and those are
  the only things labelled on.
- **The cardinality rule and the exposure rule are the same rule.** Because no metric names a tenant,
  the scrape body describes the platform's health and nothing about who is using it — which is what
  lets `/metrics` stay unauthenticated, rather than needing credentials Prometheus is awkward at
  carrying.
- **Spans carry what metrics may not, and never carry prompts.** Tenant, agent, conversation, and the
  real remote tool name belong on spans, which are sampled and expire. Prompt text, retrieved
  passages, and answers belong on neither: a span leaves the process, so one quoting the prompt would
  ship a tenant's documents to a third-party backend on every request.
- **A streaming response is timed at its last byte.** A middleware that measures when the handler
  returns records the SSE chat endpoint — the slowest thing the platform does — as taking roughly
  zero milliseconds, because the body runs after the response object is returned. The request
  middleware is raw ASGI for this reason.
- **Tracing is off until an operator names a collector.** The OpenTelemetry API is safe to call with
  no SDK configured, so instrumentation lives unconditionally in the hot paths and costs nothing when
  there is nowhere to send it. An observability stack is an operational dependency, and the platform
  must not require one to boot — the same discipline the checkpointer follows.
- **Evaluation is downstream of serving and can fail on its own.** The judge reads the transcript
  after the fact and nothing on the request path waits for it, so an evaluator outage costs the
  platform its scores and never its answers. It is also why the transcript is the product record and
  the checkpointer is not: judging happens later, twice, or under a better rubric.
- **The judge counts claims; the platform does the arithmetic.** Models are unreliable at producing
  calibrated scores and reliable at deciding whether a sentence follows from a paragraph, so the
  rubric only ever asks the second question. Every score is recomputable from the stored claims, and
  a cheap model becomes a defensible judge — which is what makes judging every turn affordable.
- **The judge model is platform config, never per-agent.** A groundedness score is only comparable
  across agents if the same judge produced it, and a tenant allowed to choose its own would choose a
  lenient one.
- **An unreadable verdict raises rather than defaults.** A fabricated score in an audit trail is
  worse than a missing one, because somebody will chart it and act on it.
- **An abstention has no groundedness.** It is stored as a null score and counted by verdict, not
  folded into the average as a 1.0 — which would let a retriever that finds nothing report a
  flawless platform.
- **Serving and evaluation are separate workloads on the metrics.** The judge deliberately shares
  `complete()` so it cannot become a workload nobody counts, which leaves latency the one thing that
  must be told apart: its long, unhurried calls would otherwise move the p95 an operator pages on
  without a single answer getting slower.
- **A calibration report declines to recommend a floor from thin data.** The relevance-floor decision
  was deferred precisely to stop it being a guess, and an analysis confident on the strength of nine
  samples is the same guess wearing a chart.
- **Liveness checks nothing; readiness checks Postgres.** A liveness probe wired to the database
  turns a brief blip into a rolling restart of every replica.
- **Neither `app/agents` nor `app/services` imports from `app/api`.** They raise domain errors and
  the router maps them to status codes, which is what keeps the same logic callable from a worker
  or a test.
