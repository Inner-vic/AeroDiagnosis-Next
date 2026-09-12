from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mcp import Client

from aerodiagnosis.adapters.mcp import create_server
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings


def _application(tmp_path: Path) -> Any:
    settings = RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "runtime.sqlite3",
        operator_token=None,
        vector_backend="sqlite",
        graph_backend="sqlite",
    )
    return bootstrap(settings)


def test_mcp_exposes_read_only_shared_application_tools(tmp_path: Path) -> None:
    application = _application(tmp_path)
    application.ingest_document.execute(
        display_name="manual.txt",
        content=b"EGT rise with compressor efficiency loss.",
    )

    async def exercise() -> None:
        async with Client(create_server(application)) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools.tools}
            assert names == {
                "hybrid_retrieve_evidence",
                "search_manual_chunks",
                "traverse_fault_graph",
                "find_similar_cases",
                "analyze_gas_path_parameters",
                "get_runtime_status",
            }

            hybrid_result = await client.call_tool(
                "hybrid_retrieve_evidence",
                {"query": "compressor EGT", "top_k": 3},
            )
            assert hybrid_result.is_error is False
            assert hybrid_result.structured_content is not None
            assert hybrid_result.structured_content["algorithm"] == "weighted_rrf@1"

            result = await client.call_tool(
                "search_manual_chunks",
                {"query": "compressor EGT", "top_k": 3},
            )
            assert result.is_error is False
            assert result.structured_content is not None
            evidence = result.structured_content["evidence"]
            assert isinstance(evidence, list)
            assert evidence[0]["evidence_id"]

            parameter_result = await client.call_tool(
                "analyze_gas_path_parameters",
                {
                    "query": "gas path anomaly",
                    "parameters": [
                        {
                            "name": "EGT",
                            "value": 710,
                            "expected_min": 500,
                            "expected_max": 650,
                            "unit": "C",
                        }
                    ],
                },
            )
            assert parameter_result.is_error is False
            assert parameter_result.structured_content is not None
            assert parameter_result.structured_content["evidence"][0]["source_kind"] == (
                "parameter_analysis"
            )

    asyncio.run(exercise())
