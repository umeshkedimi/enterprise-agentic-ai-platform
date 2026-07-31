"""The observability decisions that are wrong silently.

Every property here fails in a way that looks like nothing at all. A metric with
a tenant label works perfectly until the TSDB falls over months later. A
streaming response timed at the wrong moment reports a plausible number that
happens to be a lie. A span carrying prompt text exports one tenant's documents
to a third-party backend and raises nothing. None of these show up in a test of
what the platform *does*, so they are tested here, for what it *records*.
"""

import asyncio
import uuid

import litellm
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from starlette.responses import StreamingResponse

from app.agents import graph as graph_module
from app.agents.runner import run_turn
from app.core import metrics, tracing
from app.core.config import Settings
from app.core.logging import add_trace_context
from app.core.middleware import UNMATCHED_ROUTE, RequestContextMiddleware, route_label
from app.models.agent import Agent
from app.services.retrieval_service import RetrievedChunk
from app.tools.registry import Tool

# --- label cardinality ------------------------------------------------------

# A label whose values somebody outside the platform chooses is an unbounded
# label, and an unbounded label is a permanent time series per value. These are
# the words that give one away.
FORBIDDEN_LABEL_FRAGMENTS = (
    "tenant",
    "conversation",
    "collection",
    "agent",
    "user",
    "document",
    "server",
    "slug",
    "path",
    "url",
    "query",
    "question",
    "key",
)

# `tool` is the one label named after something a tenant can influence, and it
# is safe only because `tool_label` collapses every remote name to a constant.
# The test below pins that; this exempts the name from the blanket scan.
ALLOWED_LABELS = {"method", "route", "status", "outcome", "streamed", "error",
                  "provider", "model", "kind", "tool"}


def _declared_metrics():
    for name, value in vars(metrics).items():
        labels = getattr(value, "_labelnames", None)
        if labels is not None and not name.startswith("_"):
            yield name, labels


def test_no_metric_is_labelled_by_something_a_tenant_supplies():
    """The rule the whole metrics module exists to enforce, enforced.

    Written as a scan rather than a list of assertions so it covers metrics that
    do not exist yet: adding one with an `agent_id` label fails here rather than
    six months later in production.
    """
    offenders = {
        f"{name}.{label}": label
        for name, labels in _declared_metrics()
        for label in labels
        if any(fragment in label for fragment in FORBIDDEN_LABEL_FRAGMENTS)
    }
    assert offenders == {}


def test_every_metric_label_is_one_the_platform_chose():
    unknown = {
        f"{name}.{label}"
        for name, labels in _declared_metrics()
        for label in labels
        if label not in ALLOWED_LABELS
    }
    assert unknown == set()


def test_a_remote_tool_is_counted_under_a_constant_not_its_name():
    remote = Tool(
        name="jira__search_issues", description="", parameters={}, handler=None, remote=True
    )
    # The name a tenant's server chose never becomes a label. It is on the span
    # instead, where it costs nothing permanent.
    assert remote.metric_label == metrics.TOOL_LABEL_REMOTE


def test_a_platform_tool_keeps_its_own_name():
    builtin = Tool(name="search_knowledge_base", description="", parameters={}, handler=None)
    # Bounded by construction: registration is import-time, so the set of these
    # names is fixed at deploy.
    assert builtin.metric_label == "search_knowledge_base"


# --- route labelling --------------------------------------------------------


class _FakeRoute:
    path = "/agents/{agent_id}/chat"


def test_a_route_is_labelled_by_its_template_not_its_path():
    # The whole point: one series for the endpoint, not one per agent.
    assert route_label({"route": _FakeRoute()}) == "/agents/{agent_id}/chat"


def test_an_unmatched_request_is_labelled_once_for_all_of_them():
    # Otherwise anyone on the internet can mint series by requesting nonsense.
    assert route_label({}) == UNMATCHED_ROUTE


# --- streaming duration -----------------------------------------------------

STREAM_DELAY = 0.15


def _streaming_app() -> FastAPI:
    """A FastAPI app, not a bare Starlette one, because the route *template* only
    lands in the ASGI scope for a FastAPI route — and the template is the whole
    point of the label."""
    app = FastAPI()

    @app.get("/slow-stream")
    async def slow_stream():
        async def body():
            yield b"first"
            await asyncio.sleep(STREAM_DELAY)
            yield b"last"

        return StreamingResponse(body(), media_type="text/plain")

    app.add_middleware(RequestContextMiddleware)
    return app


async def test_a_streaming_response_is_timed_until_its_last_byte():
    """The bug this middleware was rewritten to avoid.

    A middleware that measures when the handler returns records the SSE chat
    endpoint — the slowest thing the platform does — as taking roughly zero
    milliseconds, because a streaming response's body runs after the handler
    has already returned. The number it reports is not obviously wrong, which is
    what makes it dangerous.
    """
    app = _streaming_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        response = await client.get("/slow-stream")
        assert response.text == "firstlast"

    total = metrics.REGISTRY.get_sample_value(
        "eaap_http_request_duration_seconds_sum", {"method": "GET", "route": "/slow-stream"}
    )
    assert total is not None and total >= STREAM_DELAY


async def test_a_finished_stream_leaves_nothing_in_flight():
    app = _streaming_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        await client.get("/slow-stream")

    # A gauge that only goes up is worse than no gauge: it would show a steadily
    # growing number of open streams on a platform with none.
    in_flight = metrics.REGISTRY.get_sample_value(
        "eaap_http_responses_in_flight", {"method": "GET", "route": "/slow-stream"}
    )
    assert in_flight == 0.0


async def test_a_request_carries_its_correlation_id_back():
    app = _streaming_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        supplied = str(uuid.uuid4())
        response = await client.get("/slow-stream", headers={"X-Request-ID": supplied})

    # An id minted upstream has to survive the hop, or a gateway's trace and the
    # platform's logs describe the same request under two different names.
    assert response.headers["X-Request-ID"] == supplied


# --- tracing ----------------------------------------------------------------


@pytest.fixture(scope="module")
def exporter() -> InMemorySpanExporter:
    """A real tracer provider writing spans to memory.

    The SDK rather than a mock, because what is being checked is a property of
    what would actually be exported. Module-scoped by necessity: the global
    tracer provider can only be set once per process, and a second attempt is
    ignored with a warning — so a per-test provider would leave every test after
    the first reading an exporter nothing writes to.
    """
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    trace.set_tracer_provider(provider)
    return memory


@pytest.fixture
def spans(exporter: InMemorySpanExporter) -> InMemorySpanExporter:
    exporter.clear()
    return exporter


def test_tracing_stays_off_until_an_operator_names_a_collector():
    before = tracing._configured
    tracing.configure_tracing(Settings(otel_exporter_otlp_endpoint=""))
    # A checkout with no observability stack has to boot. Tracing is an
    # operational dependency and the platform must not require one.
    assert tracing._configured == before


def test_an_absent_attribute_is_dropped_rather_than_set_to_none(spans):
    with tracing.span("t", **{tracing.CONVERSATION_ID: None, tracing.AGENT_SLUG: "hr"}):
        pass

    (recorded,) = spans.get_finished_spans()
    # `None` is a type error in the OTel API, and half these attributes are
    # legitimately absent — a turn with no conversation, an agent with no
    # collection. Filtering once beats a conditional at every call site.
    assert tracing.CONVERSATION_ID not in recorded.attributes
    assert recorded.attributes[tracing.AGENT_SLUG] == "hr"


def test_a_failing_span_records_the_error(spans):
    with pytest.raises(ValueError), tracing.span("t"):
        raise ValueError("boom")

    (recorded,) = spans.get_finished_spans()
    assert recorded.status.status_code.name == "ERROR"


def test_a_log_line_names_the_trace_it_happened_in(spans):
    with tracing.span("t"):
        stamped = add_trace_context(None, "info", {"event": "x"})

    # The join between two signals that are otherwise searched separately: the
    # trace id in the log line is what turns "this request was slow" into "here
    # is where the time went".
    assert len(stamped["trace_id"]) == 32
    assert len(stamped["span_id"]) == 16


def test_a_log_line_outside_a_trace_is_left_alone(spans):
    # Unconditional instrumentation is only acceptable if it costs nothing when
    # there is nothing to record.
    assert add_trace_context(None, "info", {"event": "x"}) == {"event": "x"}


# --- what a traced turn looks like ------------------------------------------

QUESTION = "How many vacation days do I get?"
SECRET_DOCUMENT_TEXT = "Employees accrue twenty-five days of paid annual leave."


@pytest.fixture
def traced_turn(monkeypatch, spans):
    """Run one real graph turn with both doors to the outside world faked."""

    async def fake_search(session, query, *, collection_id, top_k):
        return [
            RetrievedChunk(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                filename="handbook.pdf",
                content=SECRET_DOCUMENT_TEXT,
                score=0.88,
            )
        ]

    async def fake_acompletion(**params):
        return litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "Twenty-five [1]."}}],
            usage={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
            model=params["model"],
        )

    monkeypatch.setattr(graph_module, "semantic_search", fake_search)
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return spans


async def test_a_turn_traces_as_the_shape_of_the_graph(traced_turn):
    agent = Agent(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        slug="hr",
        name="HR",
        system_prompt="You are the HR assistant.",
        model="gpt-4o-mini",
        collection_id=uuid.uuid4(),
        tool_allowlist=[],
        temperature=0.2,
        max_output_tokens=512,
        retrieval_top_k=3,
        enabled=True,
    )
    await run_turn(
        agent=agent,
        question=QUESTION,
        session=None,
        settings=Settings(openai_api_key="sk-test"),
    )

    names = [s.name for s in traced_turn.get_finished_spans()]
    # The tree is the run: a turn, the nodes it routed through, and the provider
    # call inside one of them. This is the thing a latency number cannot say.
    assert "agent.turn" in names
    assert "agent.retrieve" in names
    assert "agent.generate" in names
    assert "chat gpt-4o-mini" in names


async def test_no_span_carries_prompt_or_document_text(traced_turn):
    agent = Agent(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        slug="hr",
        name="HR",
        system_prompt="You are the HR assistant.",
        model="gpt-4o-mini",
        collection_id=uuid.uuid4(),
        tool_allowlist=[],
        temperature=0.2,
        max_output_tokens=512,
        retrieval_top_k=3,
        enabled=True,
    )
    await run_turn(
        agent=agent,
        question=QUESTION,
        session=None,
        settings=Settings(openai_api_key="sk-test"),
    )

    exported = " ".join(
        str(value)
        for span in traced_turn.get_finished_spans()
        for value in (span.attributes or {}).values()
    )
    # Spans leave the process. A span carrying the prompt would take one
    # tenant's retrieved documents to a third-party tracing backend, quietly and
    # on every request — which is why span attributes here are ids and counts.
    assert SECRET_DOCUMENT_TEXT not in exported
    assert QUESTION not in exported


async def test_a_request_is_traced_as_a_request(spans):
    """The app factory instruments; the lifespan is too late.

    `instrument_app` patches `build_middleware_stack`, and Starlette has already
    called that by the time a lifespan handler runs — so instrumenting there
    logs nothing, raises nothing, and produces every turn as an orphan root
    trace with no request span above it and no statement spans beside it. This
    is the guard: move the call back into the lifespan and the server span
    disappears from here.
    """
    from app.main import create_app

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        assert (await client.get("/tenants/me")).status_code == 401

    assert "GET /tenants/me" in [s.name for s in spans.get_finished_spans()]
