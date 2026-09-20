from __future__ import annotations

import asyncio
from pathlib import Path

from aerodiagnosis.adapters.mcp import create_server
from aerodiagnosis.adapters.mcp_client import McpToolClient
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.domain import DiagnosisCommand
from aerodiagnosis.tools.mcp_external import mcp_external_tool


def test_mcp_client_discovers_and_calls_server_tools(tmp_path: Path) -> None:
    settings = RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "runtime.db",
    )
    server = create_server(bootstrap(settings))

    async def exercise() -> None:
        client = McpToolClient()
        tools = await client.list_tools(server)
        names = {tool.name for tool in tools}

        assert "get_runtime_status" in names
        result = await client.call_tool(
            server,
            name="get_runtime_status",
            arguments={},
        )
        assert result["vector_backend"] == "sqlite_hashing"

    asyncio.run(exercise())


def test_mcp_tool_can_be_used_as_diagnosis_evidence(tmp_path: Path) -> None:
    settings = RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "runtime.db",
    )
    server = create_server(bootstrap(settings))
    client = McpToolClient()
    handler = mcp_external_tool(
        server,
        client,
        tool_name="hybrid_retrieve_evidence",
    )

    evidence = handler(
        DiagnosisCommand(session_id="session", question="runtime status"),
        "status",
        frozenset(),
    )

    assert evidence[0].source_kind == "external_tool"
    assert "algorithm" in evidence[0].excerpt
