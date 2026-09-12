"""Validated domain records for evidence-first diagnostic runs."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DiagnosisReportStatus(StrEnum):
    EVIDENCE_READY = "evidence_ready"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ToolExecutionStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"


class ParameterObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=80)
    value: float
    expected_min: float
    expected_max: float
    unit: str = Field(default="", max_length=30)

    @model_validator(mode="after")
    def require_ordered_reference_range(self) -> Self:
        if self.expected_max <= self.expected_min:
            raise ValueError("expected_max must be greater than expected_min")
        return self


class DiagnosisCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    min_relevance: float = Field(default=0.05, ge=0.0, le=1.0)
    parameters: tuple[ParameterObservation, ...] = Field(default=(), max_length=50)
    max_tool_calls: int = Field(default=12, ge=1, le=50)


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_kind: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    excerpt: str = Field(min_length=1, max_length=600)
    locator: dict[str, Any]
    score: float = Field(ge=-1.0, le=1.0)


class EvidenceClaim(BaseModel):
    """A generated claim approved by the constrained verifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    statement: str = Field(min_length=1, max_length=600)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    confidence: float = Field(ge=0.0, le=1.0)
    verification_method: Literal["llm_verifier@1"] = "llm_verifier@1"


class ToolExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    query: str = Field(min_length=1)
    status: ToolExecutionStatus
    evidence_count: int = Field(ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def validate_status_shape(self) -> Self:
        if self.status is ToolExecutionStatus.ERROR and self.error is None:
            raise ValueError("failed tool executions require an error")
        if self.status is not ToolExecutionStatus.ERROR and self.error is not None:
            raise ValueError("non-failed tool executions cannot contain an error")
        return self


class DiagnosisReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: DiagnosisReportStatus
    question: str = Field(min_length=3)
    summary: str | None = Field(default=None, max_length=2000)
    claims: tuple[EvidenceClaim, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    tool_executions: tuple[ToolExecution, ...] = ()
    attempted_queries: tuple[str, ...] = Field(min_length=1)
    refusal_reason: str | None = None
    workflow_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_fail_closed_evidence_binding(self) -> Self:
        evidence_by_id = {item.evidence_id: item for item in self.evidence}
        if self.status is DiagnosisReportStatus.EVIDENCE_READY:
            if not self.claims or not evidence_by_id or self.refusal_reason is not None:
                raise ValueError("evidence-ready reports require claims and evidence only")
            for claim in self.claims:
                if any(identifier not in evidence_by_id for identifier in claim.evidence_ids):
                    raise ValueError("claim references evidence outside the run snapshot")
        elif self.claims or self.refusal_reason is None or self.summary is not None:
            raise ValueError("insufficient-evidence reports must refuse without claims")
        if not self.tool_executions:
            raise ValueError("diagnosis reports require a tool execution trace")
        return self
