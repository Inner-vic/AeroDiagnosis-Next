"""Common source bundle for knowledge answers, troubleshooting and reports."""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aerodiagnosis.domain.diagnosis import EvidenceItem
from aerodiagnosis.domain.knowledge_query import KnowledgeQueryContext


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    query: str = Field(min_length=1, max_length=4000)
    known_conditions: KnowledgeQueryContext = Field(default_factory=KnowledgeQueryContext)
    evidence: tuple[EvidenceItem, ...] = ()
    conflicts: tuple[str, ...] = ()
    missing_conditions: tuple[str, ...] = ()
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_references(self) -> Self:
        ids = [e.evidence_id for e in self.evidence]
        if len(set(ids)) != len(ids):
            raise ValueError("Evidence bundle references must be unique")
        if any(e.applicability_status == "conflict" for e in self.evidence):
            raise ValueError(
                "Incompatible evidence belongs in conflict records, not answer support"
            )
        return self
