"""Transport-independent tool request and result contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    DEGRADED = "degraded"


class ActorContext(BaseModel):
    """Server-derived caller identity; never populated from a tool payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str = Field(min_length=1)
    scopes: tuple[str, ...] = ()


class ToolError(BaseModel):
    """Stable internal error that adapters map to HTTP or MCP errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    message: str = Field(min_length=1)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ToolRequest(BaseModel):
    """One logical domain-tool request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    run_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    deadline: datetime
    actor: ActorContext
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_timezone(self) -> Self:
        if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
            raise ValueError("deadline must be timezone-aware")
        return self


class ToolResult(BaseModel):
    """Transport-neutral terminal result for a tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    call_id: str = Field(min_length=1)
    status: ToolStatus
    data: dict[str, Any] | None = None
    error: ToolError | None = None
    evidence_ids: tuple[str, ...] = ()
    backend: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime

    @model_validator(mode="after")
    def validate_terminal_shape(self) -> Self:
        for name, value in (("started_at", self.started_at), ("finished_at", self.finished_at)):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.status in {ToolStatus.OK, ToolStatus.DEGRADED} and self.error is not None:
            raise ValueError("successful or degraded results must not contain an error")
        if (
            self.status in {ToolStatus.ERROR, ToolStatus.TIMEOUT, ToolStatus.CANCELLED}
            and self.error is None
        ):
            raise ValueError("non-success results must contain an error")
        return self
