"""Explicit validation helpers for every future LangGraph node boundary."""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class BudgetLedger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    token_used: int = Field(default=0, ge=0)
    elapsed_ms: int = Field(default=0, ge=0)
    rounds_used: int = Field(default=0, ge=0)
    tool_calls_used: int = Field(default=0, ge=0)


class DiagnosisStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    workflow_version: str = Field(min_length=1)
    state_schema_version: str = Field(min_length=1)
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    question: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    budget: BudgetLedger = Field(default_factory=BudgetLedger)
    status: str = "created"


class DiagnosisState(TypedDict):
    run_id: str
    workflow_version: str
    state_schema_version: str
    snapshot_id: str
    question: str
    evidence_ids: tuple[str, ...]
    budget: dict[str, int]
    status: str


def validate_state(raw: DiagnosisState | dict[str, Any]) -> DiagnosisStateModel:
    """Validate a full graph state at a node boundary."""

    return DiagnosisStateModel.model_validate(raw)


def apply_state_update(
    current: DiagnosisState | dict[str, Any],
    update: dict[str, Any],
) -> DiagnosisStateModel:
    """Apply a node update and reject invalid or decreasing budget counters."""

    before = validate_state(current)
    merged = before.model_dump(mode="python")
    merged.update(update)
    after = DiagnosisStateModel.model_validate(merged)
    for field_name in BudgetLedger.model_fields:
        if getattr(after.budget, field_name) < getattr(before.budget, field_name):
            raise ValueError(f"budget field {field_name} must be monotonic")
    return after
