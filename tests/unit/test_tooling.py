from __future__ import annotations

import pytest
from pydantic import ValidationError

from aerodiagnosis.application.tooling import (
    SideEffectMode,
    ToolDefinition,
    ToolRegistry,
)


async def _unused_handler(request):  # type: ignore[no-untyped-def]
    raise AssertionError(request)


def test_side_effect_tool_requires_recovery_semantics() -> None:
    with pytest.raises(ValidationError, match="idempotency or reconciliation"):
        ToolDefinition(
            name="publish_document",
            schema_major=1,
            implementation_digest="code-sha",
            mutable_read_sources=(),
            has_side_effects=True,
        )


def test_registry_rejects_duplicate_names() -> None:
    definition = ToolDefinition(
        name="search_manual_chunks",
        schema_major=1,
        implementation_digest="code-sha",
        mutable_read_sources=("documents", "vector_index"),
    )
    registry = ToolRegistry()
    registry.register(definition, _unused_handler)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(definition, _unused_handler)


def test_safe_side_effect_tool_can_be_registered() -> None:
    definition = ToolDefinition(
        name="publish_document",
        schema_major=1,
        implementation_digest="code-sha",
        mutable_read_sources=("document_manifest",),
        has_side_effects=True,
        side_effect_mode=SideEffectMode.DETERMINISTIC_RECONCILIATION,
    )
    registry = ToolRegistry()

    registry.register(definition, _unused_handler)

    assert registry.names() == ("publish_document",)
