"""Thin interfaces over the model and the MCP servers, so the graph can run against fakes in tests."""

import asyncio
import json
import random
import re
from contextvars import ContextVar
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# The token counter for whichever graph node is running. Set by metrics.instrument(); tasks started
# inside a node (asyncio.gather) inherit it, so concurrent calls are still attributed to their node.
current_usage: ContextVar[dict | None] = ContextVar("current_usage", default=None)


class AgentError(RuntimeError):
    """A failure the user can act on (missing key, out of credits, nothing found)."""


class StructuredLLM(Protocol):
    async def parse(self, messages: list[dict], schema: type[T]) -> T: ...


class Tools(Protocol):
    async def call(self, name: str, args: dict) -> dict: ...


def record_usage(prompt: int = 0, completion: int = 0, cached: int = 0) -> None:
    usage = current_usage.get()
    if usage is not None:
        usage["llm_calls"] += 1
        usage["input_tokens"] += prompt
        usage["cached_input_tokens"] += cached
        usage["output_tokens"] += completion


class OpenAIStructured:
    """Structured output from OpenAI. Retries on rate limits and server errors are left to the
    SDK (exponential backoff), configured with max_retries where the client is created."""

    def __init__(self, client, model: str):
        self.client, self.model = client, model

    async def parse(self, messages: list[dict], schema: type[T]) -> T:
        response = await self.client.chat.completions.parse(
            model=self.model, messages=messages, response_format=schema)
        u = getattr(response, "usage", None)
        if u is not None:
            details = getattr(u, "prompt_tokens_details", None)
            record_usage(u.prompt_tokens, u.completion_tokens, getattr(details, "cached_tokens", 0) or 0)
        parsed = response.choices[0].message.parsed
        if parsed is None:  # refusal or truncated output
            raise AgentError(f"The model returned no {schema.__name__} (refusal or length limit).")
        return parsed


# Errors worth retrying: rate limits, server errors, timeouts and dropped connections.
TRANSIENT = re.compile(r"\b(429|500|502|503|504)\b|timed? ?out|timeout|connect(ion)? ?(error|reset)", re.I)
# Hunter answers 429 when the monthly quota is used up, which no retry fixes.
TRANSIENT_NO_429 = re.compile(r"\b(500|502|503|504)\b|timed? ?out|timeout|connect(ion)? ?(error|reset)", re.I)


class MCPTools:
    """Calls tools on one MCP session. Our servers always return a single JSON text block.
    Errors that look transient are retried with exponential backoff and jitter."""

    def __init__(self, session, retry_on: re.Pattern = TRANSIENT, attempts: int = 3, base_delay: float = 0.5,
                 sleep=asyncio.sleep):
        self.session, self.retry_on, self.attempts = session, retry_on, attempts
        self.base_delay, self.sleep = base_delay, sleep

    async def _once(self, name: str, args: dict) -> dict:
        result = await self.session.call_tool(name, args)
        if getattr(result, "isError", False):
            return {"error": result.content[0].text if result.content else f"{name} failed"}
        return json.loads(result.content[0].text)

    async def call(self, name: str, args: dict) -> dict:
        for attempt in range(self.attempts):
            data = await self._once(name, args)
            error = data.get("error") if isinstance(data, dict) else None
            if not error or not self.retry_on.search(str(error)) or attempt == self.attempts - 1:
                return data
            # 0.5s, 1s, 2s ... plus up to 50% jitter so parallel branches don't retry in lockstep
            delay = self.base_delay * 2 ** attempt
            await self.sleep(delay + random.uniform(0, delay / 2))
        return data
