"""Run an agent's configuration against whichever model it names.

This is the layer that makes "onboarding an assistant is a configuration
change" true at runtime: nothing here branches on which team, tenant, or
provider is involved. The agent row supplies the prompt, the model, and the
execution policy; LiteLLM supplies one calling convention across providers.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import litellm

from app.core.config import Settings, get_settings
from app.core.llm import UnknownModelError, resolve_model_route
from app.core.logging import get_logger
from app.models.agent import Agent
from app.services.errors import (
    AgentDisabledError,
    ModelConfigurationError,
    ModelInvocationError,
    ProviderNotConfiguredError,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class Turn:
    """One conversational turn supplied by the caller.

    The system prompt is absent by design — it comes from the agent's
    configuration, never from the request, so a caller cannot talk an agent out
    of its own instructions.
    """

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation the model asked for.

    `arguments` stays as the raw JSON string the model produced. Parsing it is
    the tool layer's job, because a malformed blob is a model mistake that
    should reach a handler able to answer "that isn't valid" — not an exception
    thrown three layers below where the model could learn from it.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Completion:
    text: str
    # The model that actually served the request, which can differ from the one
    # requested (provider-side aliasing), so it is reported rather than echoed.
    model: str
    provider: str
    usage: TokenUsage
    latency_ms: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    # The assistant message exactly as the provider returned it. A tool loop has
    # to replay its own tool-call message verbatim on the next request — provider
    # ids and argument strings included — or the results it sends back cannot be
    # matched to the calls that asked for them.
    raw_message: dict[str, Any] = field(default_factory=dict)


def _build_messages(
    agent: Agent,
    turns: Sequence[Turn],
    directives: Sequence[str],
    scratchpad: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    # Concatenated into one system message rather than sent as several: providers
    # disagree on whether multiple system messages are even permitted (Anthropic
    # takes a single top-level `system`), and a shape that survives every provider
    # is worth more here than the structure.
    system = "\n\n".join([agent.system_prompt, *directives])
    return [
        {"role": "system", "content": system},
        *({"role": t.role, "content": t.content} for t in turns),
        *scratchpad,
    ]


def _extract_tool_calls(message: Any) -> list[ToolCall]:
    return [
        ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments or "")
        for call in (getattr(message, "tool_calls", None) or [])
    ]


def _extract_usage(response: Any) -> TokenUsage:
    # LiteLLM normalizes every provider's usage block to the OpenAI shape, which
    # is the main reason token accounting can stay provider-agnostic here.
    usage = getattr(response, "usage", None)
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


async def complete(
    *,
    agent: Agent,
    turns: Sequence[Turn],
    system_directives: Sequence[str] = (),
    tools: Sequence[dict[str, Any]] = (),
    scratchpad: Sequence[dict[str, Any]] = (),
    settings: Settings | None = None,
) -> Completion:
    """Invoke `agent`'s model with its configured prompt and execution policy.

    `system_directives` are platform-authored instructions appended after the
    agent's own prompt — how to cite sources, when to refuse. They are a
    caller-of-this-function argument, not a caller-of-the-API one: nothing
    arriving over HTTP reaches the system role, which is the whole point of
    keeping `Turn` restricted to user and assistant.

    `scratchpad` carries the tool-call and tool-result messages accumulated
    within a single turn, in provider shape. That shape is not abstracted away
    on purpose: tool messages are the one place where LiteLLM's normalized
    format *is* the interface, and wrapping it would buy a translation layer
    whose only job is to reproduce what it was handed.
    """
    settings = settings or get_settings()

    if not agent.enabled:
        # A disabled agent keeps its configuration and history but refuses to
        # run — taking an assistant offline must not require deleting it.
        raise AgentDisabledError(str(agent.id))

    try:
        route = resolve_model_route(agent.model, settings)
    except UnknownModelError as exc:
        raise ModelConfigurationError(agent.model) from exc

    if route.api_key is None:
        raise ProviderNotConfiguredError(route.provider)

    params: dict[str, Any] = {
        "model": route.model,
        "messages": _build_messages(agent, turns, system_directives, scratchpad),
        "max_tokens": agent.max_output_tokens,
        "api_key": route.api_key,
        "num_retries": settings.agent_max_retries,
        # Safety net for provider-specific parameter removals we have not
        # modelled explicitly. The temperature case below is handled first so
        # that it is logged rather than silently swallowed here.
        "drop_params": True,
    }

    if tools:
        if route.supports_tool_calling:
            params["tools"] = list(tools)
        else:
            # The agent keeps running, without tools. An allowlist is stored
            # config that outlives any one model choice, so a team owner who
            # switches to a model that cannot call tools should get a working
            # assistant and a log line — not a 400 on every request.
            logger.warning(
                "tools_unsupported_by_model",
                agent_id=str(agent.id),
                model=route.model,
                provider=route.provider,
                tools=[t["function"]["name"] for t in tools],
            )

    if route.supports_sampling_params:
        params["temperature"] = agent.temperature
    else:
        # Not an error: the agent stays runnable and the model applies its own
        # default. Logged because a tenant that tuned temperature deserves to
        # find out why it had no effect.
        logger.info(
            "sampling_params_omitted",
            agent_id=str(agent.id),
            model=route.model,
            provider=route.provider,
            temperature=agent.temperature,
        )

    started = time.perf_counter()
    try:
        response = await litellm.acompletion(**params)
    except Exception as exc:  # noqa: BLE001 - LiteLLM surfaces provider-specific types
        logger.warning(
            "completion_failed",
            agent_id=str(agent.id),
            model=route.model,
            provider=route.provider,
            error=type(exc).__name__,
        )
        raise ModelInvocationError(str(exc)) from exc
    latency_ms = int((time.perf_counter() - started) * 1000)

    message = response.choices[0].message
    # Absent on a response that carried only tool calls; an empty string keeps
    # the contract total rather than making every caller handle None.
    text = message.content or ""
    tool_calls = _extract_tool_calls(message)
    usage = _extract_usage(response)

    logger.info(
        "completion_succeeded",
        agent_id=str(agent.id),
        tenant_id=str(agent.tenant_id),
        model=route.model,
        provider=route.provider,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        tool_calls=len(tool_calls),
        latency_ms=latency_ms,
    )

    return Completion(
        text=text,
        model=getattr(response, "model", route.model) or route.model,
        provider=route.provider,
        usage=usage,
        latency_ms=latency_ms,
        tool_calls=tool_calls,
        raw_message=message.model_dump() if hasattr(message, "model_dump") else {},
    )
