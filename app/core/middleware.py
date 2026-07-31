"""Correlation, access logging, and HTTP metrics, as raw ASGI.

This is deliberately not a `BaseHTTPMiddleware`. That base class hands the
response back as soon as the status and headers are ready and streams the body
afterwards, through a memory object stream in a second task — so a middleware
written against it measures an SSE chat turn, the slowest thing the platform
does, as taking roughly zero milliseconds, and reports it before the answer has
been written. Working at the ASGI level means the two messages that matter
(`http.response.start` and the final `http.response.body`) are observed exactly
where they happen, and the per-request task group disappears from the hot path.
"""

import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core import metrics
from app.core.logging import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_HEADER_BYTES = REQUEST_ID_HEADER.lower().encode()

# The label for a request that matched no route. Without it, every 404 probe for
# `/wp-login.php` would mint a permanent time series, which is a cardinality
# bomb anyone on the internet can throw.
UNMATCHED_ROUTE = "unmatched"


def route_label(scope: Scope) -> str:
    """The route *template* this request matched, e.g. `/agents/{agent_id}/chat`.

    Only meaningful once routing has happened, which is why every read of it in
    this module is after the response has started. The raw path is never used:
    this API puts UUIDs in paths, so labelling by path would create a new series
    per agent, per conversation, and per document.
    """
    route = scope.get("route")
    return getattr(route, "path", None) or UNMATCHED_ROUTE


class RequestContextMiddleware:
    """Assign every request a correlation ID, time it truthfully, and count it.

    The ID is bound into structlog's contextvars, so *every* log line emitted
    anywhere downstream — service, retrieval, LLM call — carries it without any
    of that code having to thread a request object through its signatures. This
    is what makes a single agent run greppable end-to-end in aggregated logs.

    An inbound X-Request-ID is honoured so a trace started by an upstream
    gateway survives the hop; otherwise we mint one.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # Lifespan and websocket traffic has neither a route template nor a
            # status code, and counting it here would only blur both.
            await self.app(scope, receive, send)
            return

        request_id = _inbound_request_id(scope) or str(uuid.uuid4())

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Where `request.state.request_id` reads from, so handlers and the
        # unhandled-exception handler can quote the ID without this middleware
        # having to construct a Request object.
        scope.setdefault("state", {})["request_id"] = request_id

        method = scope.get("method", "GET")
        start = time.perf_counter()
        tracker = _ResponseTracker(scope, method=method, start=start)

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                # Echoed back so a caller can quote it in a bug report.
                message.setdefault("headers", []).append(
                    (_REQUEST_ID_HEADER_BYTES, request_id.encode())
                )
                tracker.started(message["status"])
            await send(message)
            if message["type"] == "http.response.body":
                tracker.body_sent(more=bool(message.get("more_body", False)))
            elif message["type"] == "http.response.pathsend":
                tracker.body_sent(more=False)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception(
                "request_failed",
                method=method,
                path=scope.get("path", ""),
                duration_ms=int((time.perf_counter() - start) * 1000),
            )
            # Counted as the 500 the client is about to be sent. An unhandled
            # error that never reaches the request counter is exactly the gap
            # that makes an error-rate alert quietly under-report.
            tracker.aborted(status=500)
            raise

        # A response that never completed: the client went away mid-stream, or
        # the app returned without finishing the body. Recorded rather than
        # dropped, because an abandoned SSE turn is a thing an operator wants to
        # see — and because the in-flight gauge must come back down either way.
        tracker.aborted()


def _inbound_request_id(scope: Scope) -> str | None:
    for name, value in scope.get("headers", []):
        if name == _REQUEST_ID_HEADER_BYTES:
            return value.decode("latin-1")
    return None


class _ResponseTracker:
    """Records one request exactly once, whenever and however it ends."""

    def __init__(self, scope: Scope, *, method: str, start: float) -> None:
        self._scope = scope
        self._method = method
        self._start = start
        self._status: int | None = None
        self._route: str | None = None
        self._body_messages = 0
        self._finished = False

    def started(self, status: int) -> None:
        self._status = status
        # Read now, not at construction: routing happens inside the app, so the
        # template does not exist until the response has begun.
        self._route = route_label(self._scope)
        metrics.HTTP_IN_FLIGHT.labels(self._method, self._route).inc()

    def body_sent(self, *, more: bool) -> None:
        self._body_messages += 1
        if not more:
            self._finish()

    def aborted(self, *, status: int | None = None) -> None:
        if status is not None and self._status is None:
            # The app blew up before sending anything at all.
            self._status = status
            self._route = route_label(self._scope)
            metrics.HTTP_IN_FLIGHT.labels(self._method, self._route).inc()
        self._finish()

    def _finish(self) -> None:
        if self._finished or self._status is None:
            return
        self._finished = True
        route = self._route or UNMATCHED_ROUTE
        duration = time.perf_counter() - self._start

        metrics.HTTP_IN_FLIGHT.labels(self._method, route).dec()
        metrics.HTTP_REQUESTS.labels(self._method, route, str(self._status)).inc()
        metrics.HTTP_DURATION.labels(self._method, route).observe(duration)

        logger.info(
            "request_completed",
            method=self._method,
            path=self._scope.get("path", ""),
            status_code=self._status,
            duration_ms=int(duration * 1000),
            # More than one body message means the response was produced
            # incrementally — the honest definition of "this was a stream", and
            # one that needs no header parsing to establish.
            streamed=self._body_messages > 1,
        )
