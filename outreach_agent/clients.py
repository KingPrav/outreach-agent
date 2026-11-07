"""Thin interfaces over the model and the MCP servers, so the graph can run against fakes in tests."""

import json
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class AgentError(RuntimeError):
    """A failure the user can act on (missing key, out of credits, nothing found)."""


class StructuredLLM(Protocol):
    async def parse(self, messages: list[dict], schema: type[T]) -> T: ...


class Tools(Protocol):
    async def call(self, name: str, args: dict) -> dict: ...


class OpenAIStructured:
    def __init__(self, client, model: str):
        self.client, self.model = client, model

    async def parse(self, messages: list[dict], schema: type[T]) -> T:
        response = await self.client.chat.completions.parse(
            model=self.model, messages=messages, response_format=schema)
        parsed = response.choices[0].message.parsed
        if parsed is None:  # refusal or truncated output
            raise AgentError(f"The model returned no {schema.__name__} (refusal or length limit).")
        return parsed


class MCPTools:
    """Calls tools on one MCP session. Our servers always return a single JSON text block."""

    def __init__(self, session):
        self.session = session

    async def call(self, name: str, args: dict) -> dict:
        result = await self.session.call_tool(name, args)
        if getattr(result, "isError", False):
            return {"error": result.content[0].text if result.content else f"{name} failed"}
        return json.loads(result.content[0].text)
