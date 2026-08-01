"""Distributed tracing: the view metrics and logs each fail to give.

The three signals answer three different questions and this platform needs all
three, because a slow agent turn is slow for a reason none of them can supply
alone. A metric says *p95 turn latency doubled*. A log line says *this request
took 41 seconds*. Neither says which of the four things a turn does — embed,
scan, call a provider, call somebody else's MCP server — the 41 seconds went
into, and that is the only question worth asking. A trace is a turn's shape.

Two properties make this work in practice:

* **The API is safe to call when nothing is configured.** With no SDK installed
  the OpenTelemetry API hands back non-recording spans, so instrumentation can
  live unconditionally in the hot paths and a checkout with no collector running
  still boots and still costs nothing. Tracing turns on when an operator names
  an endpoint, not when a developer adds a flag.

* **Spans may carry what metrics may not.** A span is sampled, indexed, and
  expires; it can name the tenant, the agent, the conversation, and the exact
  MCP tool, which is precisely the high-cardinality detail
  `app/core/metrics.py` refuses to label on. The two are complementary by
  design: the metric tells you something is wrong across the fleet, the trace
  tells you which turn and where.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Resolved lazily by the API: at import time there is no provider yet, and this
# returns a proxy that picks one up if and when `configure_tracing` installs it.
_tracer: Tracer = trace.get_tracer("app")

_configured = False
_instrumented = False

# --- Attribute names --------------------------------------------------------
#
# OpenTelemetry's GenAI semantic conventions, used verbatim rather than invented
# locally. The value of a convention is that a backend already knows how to
# chart it — a span carrying `gen_ai.usage.input_tokens` shows up in a token-cost
# view without anybody configuring one. Platform-specific facts that no
# convention covers get an `eaap.` prefix so the two are never confused.
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

TENANT_ID = "eaap.tenant_id"
AGENT_ID = "eaap.agent_id"
AGENT_SLUG = "eaap.agent_slug"
CONVERSATION_ID = "eaap.conversation_id"
COLLECTION_ID = "eaap.collection_id"
TOOL_NAME = "eaap.tool.name"
TOOL_REMOTE = "eaap.tool.remote"
MCP_SERVER_SLUG = "eaap.mcp.server_slug"
RETRIEVED_CHUNKS = "eaap.retrieval.chunks"
RETRIEVAL_TOP_SCORE = "eaap.retrieval.top_score"
TOOL_STEPS = "eaap.tool_steps"
STREAMED = "eaap.streamed"

# Evaluation. The verdict and score are on the span as well as in the row
# because the interesting question about a bad score is usually "what did that
# turn *do*", and a span is the only signal that can answer it — the judgement
# hangs in the same trace as the retrieval and the model call it is judging.
EVALUATION_VERDICT = "eaap.evaluation.verdict"
EVALUATION_SCORE = "eaap.evaluation.score"
EVALUATION_CITATIONS = "eaap.evaluation.citations"
EVALUATION_EVIDENCE_COMPLETE = "eaap.evaluation.evidence_complete"


def configure_tracing(settings: Settings) -> None:
    """Install a tracer provider, if an operator has named somewhere to send spans.

    Deliberately silent and inert when `otel_exporter_otlp_endpoint` is empty.
    An observability stack is an operational dependency, and the platform must
    not require one to run — the same discipline the checkpointer follows, for
    the same reason.
    """
    global _configured
    if _configured or not settings.otel_exporter_otlp_endpoint:
        return

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": settings.app_version,
            "deployment.environment.name": settings.app_env,
        }
    )
    provider = TracerProvider(
        resource=resource,
        # Parent-based so a sampling decision made once, at the edge, holds for
        # the whole trace. Sampling each span independently is how a trace ends
        # up with the model call kept and the turn that made it dropped.
        sampler=ParentBased(TraceIdRatioBased(settings.otel_traces_sample_ratio)),
    )
    # Batched, not simple: a span exporter on the request path would put an HTTP
    # call to the collector inside the latency it is trying to measure.
    endpoint = f"{settings.otel_exporter_otlp_endpoint.rstrip('/')}/v1/traces"
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _configured = True
    logger.info(
        "tracing_configured",
        endpoint=settings.otel_exporter_otlp_endpoint,
        service=settings.otel_service_name,
        sample_ratio=settings.otel_traces_sample_ratio,
    )


def instrument_app(app: Any, engine: Any) -> None:
    """Attach the auto-instrumentation the hand-written spans hang off.

    FastAPI supplies the server span that roots every trace; SQLAlchemy supplies
    the statement spans that turn "retrieval was slow" into "the vector scan was
    slow" with nobody instrumenting a query by hand.

    **Called from the app factory, not the lifespan, and that is not a style
    choice.** `instrument_app` works by patching `build_middleware_stack`, which
    Starlette has already called by the time a lifespan handler runs — so
    instrumenting there appears to work, logs nothing, and produces a turn's
    spans as an orphan root trace with no request above them and no statement
    spans beside them. It was verified this way round by looking at the traces.

    Unconditional, like every other piece of instrumentation here. With no
    tracer provider installed these produce non-recording spans, so a process
    that will never export anything pays almost nothing, and an app built for a
    test need not know whether tracing exists.

    `engine` is passed in rather than imported: `app/core` sits below `app/db`
    and does not get to reach upward for it.
    """
    global _instrumented

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    # Per-app and already idempotent, so it runs for every app built — a suite
    # that constructs several must not end up with only the first one traced.
    FastAPIInstrumentor.instrument_app(
        app,
        # Scraping and probing are not traffic. Left in, `/metrics` would be the
        # most-traced endpoint in the platform and would tell nobody anything.
        excluded_urls="health,health/ready,metrics",
        # The ASGI `receive`/`send` spans triple the span count of every request
        # and say nothing this platform's own spans do not. Dropped so a turn's
        # trace is readable as a turn.
        exclude_spans=["receive", "send"],
    )

    if not _instrumented:
        # This half *is* process-wide — it patches the engine, not the app — so
        # it is the one that needs a guard.
        #
        # The async engine's sync core is what issues statements, which is why
        # this instruments `sync_engine`. The statements still land inside the
        # request's span: SQLAlchemy propagates the context into the greenlet it
        # runs them in.
        SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
        _instrumented = True


def shutdown_tracing() -> None:
    """Flush pending spans on the way out.

    Without this, the batch processor's last window is lost on every deploy —
    and the spans in it are disproportionately the interesting ones, because a
    pod being shut down is a pod whose final requests are worth looking at.
    """
    if not _configured:
        return
    provider = trace.get_tracer_provider()
    shutdown = getattr(provider, "shutdown", None)
    if shutdown is not None:
        shutdown()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Start a span, dropping attributes whose value is absent.

    The `None` filter is not cosmetic: an attribute set to `None` is a type
    error in the OTel API, and half the interesting attributes here — a
    conversation id, a collection id, a response model — are legitimately absent
    on some turns. Filtering at the one choke point beats a conditional at every
    call site.
    """
    with _tracer.start_as_current_span(name) as current:
        set_attributes(current, **attributes)
        yield current


def set_attributes(current: Span, **attributes: Any) -> None:
    for key, value in attributes.items():
        if value is not None:
            current.set_attribute(key, value)


