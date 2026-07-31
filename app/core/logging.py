import logging
import sys
from typing import Any

import structlog
from opentelemetry import trace


def add_trace_context(_logger: Any, _method: str, event_dict: dict) -> dict:
    """Stamp the active trace and span ids onto every log line.

    This is the join between two signals that are otherwise searched separately.
    A log line names the trace it happened in, and a trace links back to the
    lines emitted inside it — so "show me the logs for this slow turn" and "show
    me the trace for this error" both stop being manual work.

    Unconditional, because it costs one contextvar read and adds nothing when
    there is no active span: with tracing unconfigured the current span is
    invalid and the fields are simply absent. It lives here rather than in
    `app/core/tracing.py` both to keep the import graph one-way and because what
    a log line contains is a logging decision.
    """
    context = trace.get_current_span().get_span_context()
    if context.is_valid:
        # Rendered as the hex strings every tracing backend's search box expects,
        # rather than the ints the API returns.
        event_dict["trace_id"] = format(context.trace_id, "032x")
        event_dict["span_id"] = format(context.span_id, "016x")
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            add_trace_context,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
