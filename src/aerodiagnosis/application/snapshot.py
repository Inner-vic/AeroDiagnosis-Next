"""Content-addressed run snapshots for reproducible execution and resume."""

from __future__ import annotations

import hashlib
import json
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BuildIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    git_commit: str = Field(min_length=1)
    dependency_lock_digest: str = Field(min_length=1)
    container_digest: str = Field(min_length=1)


class VersionedSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class ToolDependency(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    schema_major: int = Field(ge=1)
    implementation_digest: str = Field(min_length=1)
    sources: tuple[VersionedSource, ...] = ()


class SnapshotSpec(BaseModel):
    """Everything mutable that can affect one diagnostic run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    build: BuildIdentity
    workflow_version: str = Field(min_length=1)
    state_schema_version: str = Field(min_length=1)
    prompt_digest: str = Field(min_length=1)
    retrieval_config_digest: str = Field(min_length=1)
    model_identity: str = Field(min_length=1)
    embedding_identity: str = Field(min_length=1)
    tools: tuple[ToolDependency, ...] = ()

    @model_validator(mode="after")
    def require_unique_tools_and_sources(self) -> Self:
        tool_names = [tool.tool_name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tool names must be unique")
        for tool in self.tools:
            source_names = [source.name for source in tool.sources]
            if len(source_names) != len(set(source_names)):
                raise ValueError(f"source names for {tool.tool_name} must be unique")
        return self

    def canonical_payload(self) -> dict[str, object]:
        payload = self.model_dump(mode="json")
        tools = payload.get("tools", [])
        assert isinstance(tools, list)
        for tool in tools:
            assert isinstance(tool, dict)
            sources = tool.get("sources", [])
            assert isinstance(sources, list)
            sources.sort(key=lambda item: (item["name"], item["version"]))
        tools.sort(key=lambda item: item["tool_name"])
        return payload


class RunSnapshot(BaseModel):
    """Immutable spec plus its deterministic content address."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec: SnapshotSpec

    @staticmethod
    def compute_id(spec: SnapshotSpec) -> str:
        """Return the content address without constructing a model."""

        encoded = json.dumps(
            spec.canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def create(cls, spec: SnapshotSpec) -> Self:
        return cls(snapshot_id=cls.compute_id(spec), spec=spec)

    @model_validator(mode="after")
    def verify_content_address(self) -> Self:
        expected = type(self).compute_id(self.spec)
        if self.snapshot_id != expected:
            raise ValueError("snapshot_id does not match snapshot content")
        return self
