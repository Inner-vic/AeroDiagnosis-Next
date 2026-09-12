"""Registration rules for domain tools before runtime orchestration is introduced."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aerodiagnosis.application.contracts import ToolRequest, ToolResult

ToolHandler = Callable[[ToolRequest], Awaitable[ToolResult]]


class SideEffectMode(StrEnum):
    NONE = "none"
    PROVIDER_IDEMPOTENCY_KEY = "provider_idempotency_key"
    DETERMINISTIC_RECONCILIATION = "deterministic_reconciliation"


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    schema_major: int = Field(ge=1)
    implementation_digest: str = Field(min_length=1)
    mutable_read_sources: tuple[str, ...]
    has_side_effects: bool = False
    side_effect_mode: SideEffectMode = SideEffectMode.NONE

    @model_validator(mode="after")
    def require_safe_side_effect_semantics(self) -> Self:
        if self.has_side_effects and self.side_effect_mode is SideEffectMode.NONE:
            raise ValueError("side-effect tools require an idempotency or reconciliation mode")
        if not self.has_side_effects and self.side_effect_mode is not SideEffectMode.NONE:
            raise ValueError("read-only tools must use side_effect_mode=none")
        return self


class ToolRegistry:
    """A small registry that rejects ambiguous or unsafe tool definitions."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDefinition, ToolHandler]] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        if definition.name in self._tools:
            raise ValueError(f"tool already registered: {definition.name}")
        self._tools[definition.name] = (definition, handler)

    def definition(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name][0]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))
