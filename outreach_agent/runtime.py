"""Opens everything a run needs: both MCP servers, the model client and the SQLite checkpointer.
Shared by the command line and the web UI."""

import sys
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from .clients import MCPTools, OpenAIStructured, StructuredLLM, Tools
from .config import ROOT, Settings


@dataclass
class Services:
    people: Tools
    research: Tools
    llm: StructuredLLM
    checkpointer: object


@asynccontextmanager
async def open_services(settings: Settings):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from openai import AsyncOpenAI

    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncExitStack() as stack:
        sessions = {}
        for name in ("people", "research"):
            params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "servers" / f"{name}_server.py")])
            read, write = await stack.enter_async_context(stdio_client(params))
            sessions[name] = await stack.enter_async_context(ClientSession(read, write))
            await sessions[name].initialize()
        checkpointer = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db)))
        yield Services(MCPTools(sessions["people"]), MCPTools(sessions["research"]),
                       OpenAIStructured(AsyncOpenAI(), settings.model), checkpointer)
