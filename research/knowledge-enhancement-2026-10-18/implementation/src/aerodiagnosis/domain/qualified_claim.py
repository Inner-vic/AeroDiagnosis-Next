"""Exact-source qualified assertions for research extraction; not semantic truth labels."""

from __future__ import annotations

import builtins
import hashlib
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Predicate = Literal[
    "CAUSES",
    "CONTRIBUTES_TO",
    "INDICATES",
    "CHECK_WITH",
    "REQUIRES_ACTION",
    "MITIGATED_BY",
    "APPLIES_TO",
    "HAS_COMPONENT",
    "ATTRACTS",
]
EntityType = Literal[
    "Engine",
    "Component",
    "Symptom",
    "Cause",
    "Mechanism",
    "Parameter",
    "Check",
    "Action",
    "Condition",
    "Model",
    "Stage",
    "Document",
    "Failure",
    "Material",
]
Polarity = Literal["positive", "negative"]
Certainty = Literal["finding", "hypothesis", "conditional", "reported"]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceUnit(StrictRecord):
    unit_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    canonical_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    canonical_start: int = Field(ge=0)
    text: str = Field(min_length=1)
    source_locator: dict[str, builtins.object] = Field(default_factory=dict)
    source_kind: str = "unknown"

    @property
    def canonical_end(self) -> int:
        return self.canonical_start + len(self.text)

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class QualifiedEntity(StrictRecord):
    surface: str = Field(min_length=1, max_length=1000)
    entity_type: EntityType
    role: str = Field(default="unknown", min_length=1, max_length=120)
    role_quote: str | None = Field(default=None, min_length=1)

    @field_validator("surface", "role")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("entity text must not be blank")
        return value


class QualifierValue(StrictRecord):
    value: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class ClaimQualifiers(StrictRecord):
    engine_models: tuple[QualifierValue, ...] = ()
    flight_phases: tuple[QualifierValue, ...] = ()
    conditions: tuple[QualifierValue, ...] = ()
    revision: QualifierValue | None = None
    polarity_quote: str | None = Field(default=None, min_length=1)
    certainty_quote: str | None = Field(default=None, min_length=1)


class QualifiedCandidate(StrictRecord):
    subject: QualifiedEntity
    predicate: Predicate
    object: QualifiedEntity
    polarity: Polarity
    certainty: Certainty
    evidence_quote: str = Field(min_length=1)
    qualifiers: ClaimQualifiers = Field(default_factory=ClaimQualifiers)


class CandidateBatch(StrictRecord):
    claims: tuple[QualifiedCandidate, ...]


class SpanEvidence(StrictRecord):
    document_id: str
    canonical_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    canonical_start: int = Field(ge=0)
    canonical_end: int = Field(gt=0)
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def ordered_span(self) -> Self:
        if self.canonical_end <= self.canonical_start:
            raise ValueError("evidence span must have positive length")
        return self


class QualifiedClaim(StrictRecord):
    schema_version: Literal["qualified-claim@1"] = "qualified-claim@1"
    claim_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    unit_id: str
    subject: QualifiedEntity
    predicate: Predicate
    object: QualifiedEntity
    polarity: Polarity
    certainty: Certainty
    qualifiers: ClaimQualifiers
    evidence: SpanEvidence
    subject_evidence: SpanEvidence
    object_evidence: SpanEvidence
    qualifier_evidence: dict[str, tuple[SpanEvidence, ...]]
    source_locator: dict[str, builtins.object]
    validation_status: Literal["necessary_source_checks_passed"] = "necessary_source_checks_passed"
    semantic_support: Literal["not_established_by_validator"] = "not_established_by_validator"
    human_expert_reviewed: Literal[False] = False


class RejectedCandidate(StrictRecord):
    candidate_index: int
    candidate: dict[str, object]
    reasons: tuple[str, ...]


class ExtractionValidation(StrictRecord):
    accepted: tuple[QualifiedClaim, ...] = ()
    rejected: tuple[RejectedCandidate, ...] = ()
    input_count: int
    validator_version: Literal["source-qualifier-validator@1"] = "source-qualifier-validator@1"
