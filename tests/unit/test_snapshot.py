from __future__ import annotations

import pytest
from pydantic import ValidationError

from aerodiagnosis.application.snapshot import (
    BuildIdentity,
    RunSnapshot,
    SnapshotSpec,
    ToolDependency,
    VersionedSource,
)


def _spec(tools: tuple[ToolDependency, ...]) -> SnapshotSpec:
    return SnapshotSpec(
        build=BuildIdentity(
            git_commit="abc123",
            dependency_lock_digest="lock-sha",
            container_digest="local-dev",
        ),
        workflow_version="3.0",
        state_schema_version="1.0",
        prompt_digest="prompt-sha",
        retrieval_config_digest="retrieval-sha",
        model_identity="model@revision",
        embedding_identity="embedding@revision",
        tools=tools,
    )


def test_snapshot_id_is_independent_of_tool_and_source_order() -> None:
    search = ToolDependency(
        tool_name="search_manual_chunks",
        schema_major=1,
        implementation_digest="search-code",
        sources=(
            VersionedSource(name="documents", version="v2"),
            VersionedSource(name="index", version="v4"),
        ),
    )
    cases = ToolDependency(
        tool_name="find_similar_cases",
        schema_major=1,
        implementation_digest="case-code",
        sources=(VersionedSource(name="cases", version="v3"),),
    )
    reversed_search = search.model_copy(update={"sources": tuple(reversed(search.sources))})

    first = RunSnapshot.create(_spec((search, cases)))
    second = RunSnapshot.create(_spec((cases, reversed_search)))

    assert first.snapshot_id == second.snapshot_id


def test_snapshot_rejects_tampered_content_address() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        RunSnapshot(snapshot_id="0" * 64, spec=_spec(()))


def test_snapshot_rejects_duplicate_tool_names() -> None:
    tool = ToolDependency(
        tool_name="search_manual_chunks",
        schema_major=1,
        implementation_digest="search-code",
    )

    with pytest.raises(ValidationError, match="must be unique"):
        _spec((tool, tool))
