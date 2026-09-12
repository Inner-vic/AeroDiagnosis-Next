"""Explainable contracts for multi-route evidence retrieval."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from aerodiagnosis.domain.diagnosis import EvidenceItem


class RetrievalStrategy(StrEnum):
    """Comparable retrieval policies shared by production and offline experiments."""

    MANUAL_ONLY = "manual_only"
    GRAPH_ONLY = "graph_only"
    CASE_ONLY = "case_only"
    RRF = "rrf"
    WEIGHTED_RRF = "weighted_rrf"
    NORMALIZED_SCORE = "normalized_score"


class RetrievalContribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: str = Field(pattern=r"^(manual|graph|case)$")
    route_rank: int = Field(ge=1)
    raw_score: float = Field(ge=-1.0, le=1.0)
    normalized_score: float = Field(ge=0.0, le=1.0)
    reciprocal_rank_score: float = Field(gt=0.0, le=1.0)
    weight: float = Field(gt=0.0, le=1.0)


class FusedEvidenceHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    evidence: EvidenceItem
    fused_score: float = Field(ge=0.0, le=1.0)
    selection_reason: str = Field(pattern=r"^(route_coverage|fused_score)$")
    contributions: tuple[RetrievalContribution, ...] = Field(min_length=1, max_length=3)


class RetrievalRouteSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: str = Field(pattern=r"^(manual|graph|case)$")
    candidate_count: int = Field(ge=0)
    included_count: int = Field(ge=0)
    top_raw_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class HybridRetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=4000)
    algorithm: str = Field(
        pattern=r"^(manual_only|graph_only|case_only|rrf|weighted_rrf|normalized_score)@\d+$"
    )
    top_k: int = Field(ge=1, le=20)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    route_weights: dict[str, float] = Field(default_factory=dict)
    route_coverage_enabled: bool = True
    candidate_count: int = Field(ge=0)
    hits: tuple[FusedEvidenceHit, ...]
    routes: tuple[RetrievalRouteSummary, ...] = Field(min_length=3, max_length=3)
