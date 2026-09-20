from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp import Client


@dataclass(frozen=True, slots=True)
class McpToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


class McpToolClient:
    """Small typed client for discovering and calling tools on an MCP server."""

    async def list_tools(self, server: Any) -> tuple[McpToolSpec, ...]:
        async with Client(server) as client:
            tools = await client.list_tools()
        return tuple(
            McpToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.input_schema,
            )
            for tool in tools.tools
        )

    async def call_tool(
        self,
        server: Any,
        *,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        async with Client(server) as client:
            result = await client.call_tool(name, arguments)
        if result.is_error:
            raise RuntimeError(f"MCP tool failed: {name}")
        if result.structured_content is not None:
            return dict(result.structured_content)
        return {"content": result.content}
