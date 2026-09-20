from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from aerodiagnosis.adapters.mcp_client import McpToolClient
from aerodiagnosis.domain import DiagnosisCommand, EvidenceItem, SourceKind, make_evidence_id

from .diagnostic import ExternalToolHandler


def mcp_external_tool(
    server: Any,
    client: McpToolClient,
    *,
    tool_name: str,
) -> ExternalToolHandler:
    """Wrap an MCP query-style tool as a synchronous diagnosis evidence tool."""

    def handler(
        command: DiagnosisCommand,
        query: str,
        active_version_ids: frozenset[str],
    ) -> tuple[EvidenceItem, ...]:
        del active_version_ids
        result = asyncio.run(
            client.call_tool(
                server,
                name=tool_name,
                arguments={"query": query, "top_k": command.top_k},
            )
        )
        excerpt = json.dumps(result, ensure_ascii=False, sort_keys=True)[:600]
        content_hash = hashlib.sha256(excerpt.encode()).hexdigest()
        source_ref = hashlib.sha256(
            f"{tool_name}:{excerpt}".encode()
        ).hexdigest()
        locator = {
            "kind": "mcp_tool",
            "coordinates": {"tool": tool_name, "query": query},
        }
        return (
            EvidenceItem(
                evidence_id=make_evidence_id(
                    SourceKind.EXTERNAL_TOOL,
                    source_ref,
                    locator,
                ),
                source_kind=SourceKind.EXTERNAL_TOOL,
                source_ref=source_ref,
                document_id=tool_name,
                version_id="mcp@dynamic",
                content_hash=content_hash,
                excerpt=excerpt,
                locator=locator,
                score=1.0,
            ),
        )

    return handler
