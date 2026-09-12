"""Validated structured outputs expected from the language-model reasoning core."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidencePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search_queries: tuple[str, ...] = Field(min_length=1, max_length=3)
    tools: tuple[str, ...] = Field(min_length=1, max_length=4)
    rationale: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def require_unique_nonempty_actions(self) -> Self:
        if len(set(self.search_queries)) != len(self.search_queries):
            raise ValueError("search queries must be unique")
        if len(set(self.tools)) != len(self.tools):
            raise ValueError("tool names must be unique")
        if any(not query.strip() for query in self.search_queries):
            raise ValueError("search queries must not be blank")
        return self


class MemoryTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(pattern=r"^(user|assistant|tool)$")
    content: str = Field(min_length=1, max_length=12000)


class CandidateClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    statement: str = Field(min_length=1, max_length=1000)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    confidence: float = Field(ge=0.0, le=1.0)


class CandidateDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1, max_length=2000)
    claims: tuple[CandidateClaim, ...] = Field(min_length=1, max_length=10)


class VerificationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sufficient: bool
    supported_claim_indexes: tuple[int, ...] = Field(default=(), max_length=10)
    issues: tuple[str, ...] = Field(default=(), max_length=20)
    corrected_queries: tuple[str, ...] = Field(default=(), max_length=3)
    corrected_tools: tuple[str, ...] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def require_consistent_decision(self) -> Self:
        if self.sufficient and not self.supported_claim_indexes:
            raise ValueError("sufficient verification requires supported claims")
        if not self.sufficient and not self.issues:
            raise ValueError("failed verification requires actionable issues")
        if any(index < 0 for index in self.supported_claim_indexes):
            raise ValueError("supported claim indexes must be non-negative")
        return self
