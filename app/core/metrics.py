"""Every number this platform exports, and the labels it refuses to export.

Metrics answer a different question from logs and traces, and the difference is
what decides the label sets here. A log line or a span describes *one* request
and can afford to name the tenant, the agent, the conversation, and the tool —
it is indexed, it expires, and nobody queries across all of them. A metric is a
time series that lives in the TSDB forever, and its cost is the cross-product of
its labels. So the rule below is applied without exception:

    **A label may only take values the platform itself chose.**

Tenant ids, agent ids, conversation ids, and MCP tool names are all typed in by
somebody else, which makes each of them an unbounded label and therefore a way
for one tenant to degrade monitoring for every other. They belong in the logs
and the traces, which already carry them.

That restraint pays a second dividend, and the two decisions are really one: a
`/metrics` body containing no tenant identifiers holds nothing tenant-specific
to leak. An operator surface that aggregates away its subjects can be scraped
without becoming an authorization problem.
"""

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
)
from prometheus_client.gc_collector import GCCollector
from prometheus_client.platform_collector import PlatformCollector
from prometheus_client.process_collector import ProcessCollector

# A registry of our own rather than the process-wide default. Any library in the
# dependency tree can register a collector on the default one, and the first
# time that happens the platform starts exporting a series it never chose and
# cannot explain. The three standard collectors below are attached explicitly
# for the same reason: what this process exports is a decision, not a default.
REGISTRY = CollectorRegistry()
ProcessCollector(registry=REGISTRY)
PlatformCollector(registry=REGISTRY)
GCCollector(registry=REGISTRY)

APP_INFO = Info(
    "eaap_build",
    "Static facts about the running build, for joining onto other series.",
    registry=REGISTRY,
)

# --- HTTP -------------------------------------------------------------------

# `route` is the *route template* (`/agents/{agent_id}/chat`), never the request
# path. Paths in this API carry UUIDs, so labelling by path would mint a new
# time series per agent, per conversation, per document — the classic way to
# take down a Prometheus with a single well-behaved client.
HTTP_REQUESTS = Counter(
    "eaap_http_requests_total",
    "HTTP requests completed, by route template and status.",
    ["method", "route", "status"],
    registry=REGISTRY,
)

HTTP_DURATION = Histogram(
    "eaap_http_request_duration_seconds",
    "Wall-clock time from request received to response fully written.",
    ["method", "route"],
    buckets=(0.005, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
    registry=REGISTRY,
)

# Counted from the moment the status line is sent to the moment the last body
# byte is, which for an ordinary JSON response is close to instant and for the
# SSE chat endpoint is the whole answer. So this is, in practice, the number of
# open streams — the one number that says whether a deploy is about to cut off
# conversations mid-sentence.
HTTP_IN_FLIGHT = Gauge(
    "eaap_http_responses_in_flight",
    "Responses whose body is still being written.",
    ["method", "route"],
    registry=REGISTRY,
)

# --- Agent turns ------------------------------------------------------------

# Buckets stretch to two minutes because an agent turn is not a web request: it
# is one or more model calls, a vector search, and possibly a round trip to a
# third party's server. A histogram topping out at ten seconds would put every
# interesting turn in `+Inf` and report a p99 of "at least ten seconds".
_TURN_BUCKETS = (0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60, 120)

AGENT_TURNS = Counter(
    "eaap_agent_turns_total",
    "Orchestrated turns run, by outcome and transport.",
    ["outcome", "streamed"],
    registry=REGISTRY,
)

AGENT_TURN_DURATION = Histogram(
    "eaap_agent_turn_duration_seconds",
    "Wall-clock time for a whole turn: retrieval, every model call, every tool.",
    ["streamed"],
    buckets=_TURN_BUCKETS,
    registry=REGISTRY,
)

# The error label is a domain-error class name — a closed set defined in
# `app/services/errors.py`, which is what makes it safe to label on. It is the
# one breakdown that tells an operator whether the platform is broken (503, 502)
# or a tenant's configuration is (400, 409).
AGENT_TURN_ERRORS = Counter(
    "eaap_agent_turn_errors_total",
    "Turns that failed, by domain error class.",
    ["error"],
    registry=REGISTRY,
)

# --- Model calls ------------------------------------------------------------

# `model` is per-agent configuration, which sounds tenant-supplied and unbounded
# — but a model string only reaches these metrics after `resolve_model_route`
# has matched it against LiteLLM's model map. A typo raises before it can become
# a label. Failures earlier than that are counted as `ModelConfigurationError`
# on the turn instead, deliberately without the string that caused them.
LLM_REQUESTS = Counter(
    "eaap_llm_requests_total",
    "Provider calls, by outcome. One turn can make several.",
    ["provider", "model", "workload", "outcome"],
    registry=REGISTRY,
)

LLM_DURATION = Histogram(
    "eaap_llm_request_duration_seconds",
    "Time spent waiting on the provider for one call.",
    ["provider", "model", "workload", "streamed"],
    buckets=_TURN_BUCKETS,
    registry=REGISTRY,
)

# Split prompt from completion because they price differently and move for
# different reasons: prompt tokens grow when retrieval or history grows, which
# is a platform decision; completion tokens grow when answers get longer, which
# is a model decision. A single total would hide both.
LLM_TOKENS = Counter(
    "eaap_llm_tokens_total",
    "Tokens billed, by direction.",
    ["provider", "model", "workload", "kind"],
    registry=REGISTRY,
)

# The three things the platform calls a model for. All three are platform
# constants — nobody outside chooses between them — and separating them is not
# bookkeeping: the evaluation judge issues long, cheap, unhurried calls that
# nobody is waiting on, and folding them into the same series as the serving
# path would move the p95 an operator pages on without a single served turn
# getting slower. Summarization sits on the request path (it runs before
# `generate` on the turn that crosses the history window) but is still its own
# workload, because it is not the model call the caller is waiting to read —
# conflating the two would report a turn's answer as slower than the answer
# itself took.
WORKLOAD_SERVING = "serving"
WORKLOAD_EVALUATION = "evaluation"
WORKLOAD_SUMMARIZATION = "summarization"

# --- Retrieval --------------------------------------------------------------

RETRIEVAL_REQUESTS = Counter(
    "eaap_retrieval_requests_total",
    "Vector searches run, by outcome.",
    ["outcome"],
    registry=REGISTRY,
)

RETRIEVAL_DURATION = Histogram(
    "eaap_retrieval_duration_seconds",
    "Embedding call plus vector scan for one search.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
    registry=REGISTRY,
)

RETRIEVAL_CHUNKS = Histogram(
    "eaap_retrieval_chunks",
    "How many chunks a search returned. Zero means the agent answered ungrounded.",
    buckets=(0, 1, 2, 3, 4, 5, 8, 10, 20),
    registry=REGISTRY,
)

# The instrument that exists to settle an open question rather than to raise an
# alert. Retrieval has no relevance floor, because a badly-chosen cosine
# threshold silently breaks retrieval and the right value is embedding-model
# dependent — so the decision was deferred until it could be measured instead of
# guessed. This is the measurement: the distribution of best-match scores across
# real traffic, bucketed densely exactly where a floor would plausibly sit.
RETRIEVAL_TOP_SCORE = Histogram(
    "eaap_retrieval_top_score",
    "Cosine similarity of the best match, when a search returned anything.",
    buckets=(0.0, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0),
    registry=REGISTRY,
)

# --- Tools ------------------------------------------------------------------

# The `tool` label is bounded by construction, and it takes three kinds of value:
#
#   * a platform tool's name — an import-time constant in `app/tools`;
#   * `mcp` — every remote tool collapses to this, because MCP tool names are
#     typed into a tenant's config row and are therefore unbounded. Which remote
#     tool ran is in the log line and on the span;
#   * `unknown` — a call the allowlist refused. Those names are invented by a
#     model or suggested by a prompt injection, so they are the least bounded
#     strings in the system and the ones it would be most foolish to label on.
TOOL_LABEL_REMOTE = "mcp"
TOOL_LABEL_UNKNOWN = "unknown"

TOOL_CALLS = Counter(
    "eaap_tool_calls_total",
    "Tool invocations, by outcome.",
    ["tool", "outcome"],
    registry=REGISTRY,
)

TOOL_DURATION = Histogram(
    "eaap_tool_call_duration_seconds",
    "Time one tool handler took, including any remote round trip.",
    ["tool"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
    registry=REGISTRY,
)

# --- MCP --------------------------------------------------------------------

# Discovery is on the critical path of every model call in a turn, so its
# latency and its cache hit rate are the two numbers that decide whether the
# Redis-backed cache is doing its job — and whether the deferred connection
# pool is worth building. `outcome` is one of `cache_hit`, `ok`, `blocked`,
# `error` (the discovery itself), or `cache_error` (Redis was unreachable, so
# the call fell through to a live discovery instead of serving a cache hit).
MCP_DISCOVERY = Counter(
    "eaap_mcp_discovery_total",
    "Tool-list lookups against a tenant's MCP server, by outcome.",
    ["outcome"],
    registry=REGISTRY,
)

MCP_DISCOVERY_DURATION = Histogram(
    "eaap_mcp_discovery_duration_seconds",
    "Time to list one server's tools, excluding cache hits.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30),
    registry=REGISTRY,
)


# --- Evaluation -------------------------------------------------------------
#
# `verdict` and `reason` are safe labels for the same reason `outcome` is: every
# value either one can take is written in this repository. A verdict is one of
# four the platform defined and the judge chooses between; it is never a string
# the judge composed. The score itself is a histogram rather than a gauge because
# the interesting question is the shape of the distribution — a mean groundedness
# of 0.8 is a very different platform depending on whether it is every answer
# scoring 0.8 or four perfect answers and one fabrication.
EVALUATIONS = Counter(
    "eaap_evaluations_total",
    "Turns judged, by verdict.",
    ["evaluator", "verdict"],
    registry=REGISTRY,
)

EVALUATION_FAILURES = Counter(
    "eaap_evaluation_failures_total",
    "Judgements that could not be recorded, by reason.",
    ["evaluator", "reason"],
    registry=REGISTRY,
)

EVALUATION_DURATION = Histogram(
    "eaap_evaluation_duration_seconds",
    "Time to judge one turn, including the judge's model call.",
    ["evaluator"],
    buckets=_TURN_BUCKETS,
    registry=REGISTRY,
)

# Abstentions are absent from this histogram, not counted as 1.0. An answer that
# asserts nothing has no groundedness to measure, and folding it in as perfect
# would let a retriever that finds nothing report a flawless platform. They are
# counted by verdict on `eaap_evaluations_total` instead, where a rising
# abstention rate reads as what it is.
EVALUATION_SCORE = Histogram(
    "eaap_evaluation_score",
    "Fraction of an answer's claims the sources supported, for answers that made any.",
    ["evaluator"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)


def tool_label(name: str, *, remote: bool) -> str:
    """The bounded label for a tool that ran. See `TOOL_LABEL_REMOTE`."""
    return TOOL_LABEL_REMOTE if remote else name


def set_build_info(version: str, env: str) -> None:
    """Publish the build's identity once, at startup.

    Exported as a label-only series so a dashboard can join deploy boundaries
    onto a latency graph — the cheapest way to answer "did the release do this".
    """
    APP_INFO.info({"version": version, "env": env})


def render() -> tuple[bytes, str]:
    """The scrape body and its content type."""
    # Imported here rather than at module scope: `generate_latest` binds to the
    # default registry when called with no argument, and keeping the import next
    # to the one call site is what stops that mistake being made later.
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
