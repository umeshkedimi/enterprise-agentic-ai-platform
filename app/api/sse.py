"""Server-sent events framing.

Small enough to own rather than add a dependency for, and the framing details
below are ones worth having in sight: a stalled proxy and a dropped `\\n\\n` look
identical from the client, and both present as an agent that stopped answering
halfway through.
"""

import json
from typing import Any

# Buffering proxies are the failure mode SSE actually has. An intermediary that
# holds the response until it has "enough" turns token streaming back into a
# single delayed blob, which is worse than not streaming at all — the user waits
# the full latency and then sees the answer appear instantly, as if the feature
# were broken. These headers are the conventional way of asking them not to.
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

MEDIA_TYPE = "text/event-stream"


def frame(event: str, data: Any) -> str:
    """One SSE message: a named event and a JSON payload.

    Every frame is named, including the terminal ones. A client that switches on
    the event name never has to guess what a payload is from its shape, and the
    error frame in particular has to be unmistakable — it arrives inside a 200
    response, because the status line was sent before anything could fail.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
