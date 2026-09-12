"""Domain contracts for model-grounded, knowledge-enhanced diagnosis."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PluginStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"


class RootCauseSessionStatus(StrEnum):
    INVESTIGATING = "investigating"
    READY_FOR_REPORT = "ready_for_report"
    COMPLETED = "completed"


class CaseDraftStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"


class FeatureSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,79}$")
    unit: str = Field(default="", max_length=30)
    mean: float = 0.0
    scale: float = Field(default=1.0, gt=0.0)


class ClassSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=r"^[a-z][a-z0-9_]{1,79}$")
    label: str = Field(min_length=1, max_length=120)
    component: str = Field(pattern=r"^[a-z][a-z0-9_]{1,79}$")
    weights: tuple[float, ...] = Field(min_length=1, max_length=200)
    bias: float = 0.0


class ModelPluginManifest(BaseModel):
    """Safe declarative baseline; binary runtimes can implement the same contract later."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,79}$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$")
    display_name: str = Field(min_length=1, max_length=160)
    dataset_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,79}$")
    runtime: str = Field(default="linear_json_v1", pattern=r"^linear_json_v1$")
    task_type: str = Field(
        default="component_classification", pattern=r"^component_classification$"
    )
    features: tuple[FeatureSpecification, ...] = Field(min_length=1, max_length=200)
    classes: tuple[ClassSpecification, ...] = Field(min_length=2, max_length=100)
    ood_threshold: float = Field(default=6.0, gt=0.0, le=100.0)
    description: str = Field(default="", max_length=1000)
    teaching_demo: bool = False

    @model_validator(mode="after")
    def validate_dimensions_and_uniqueness(self) -> Self:
        feature_names = [feature.name for feature in self.features]
        class_codes = [item.code for item in self.classes]
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("model feature names must be unique")
        if len(set(class_codes)) != len(class_codes):
            raise ValueError("model class codes must be unique")
        if any(len(item.weights) != len(self.features) for item in self.classes):
            raise ValueError("every class weight vector must match the feature count")
        return self


class ModelPluginRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: ModelPluginManifest
    status: PluginStatus
    created_at: str
    updated_at: str


class DatasetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    display_name: str
    row_count: int = Field(ge=1)
    columns: tuple[str, ...] = Field(min_length=1)
    numeric_columns: tuple[str, ...]
    missing_values: int = Field(ge=0)
    quality_score: float = Field(ge=0.0, le=1.0)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ComponentPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    label: str
    component: str
    probability: float = Field(ge=0.0, le=1.0)


class ModelInferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    model_version: str
    dataset_id: str
    applicability: str = Field(pattern=r"^(in_domain|out_of_domain)$")
    input_quality: float = Field(ge=0.0, le=1.0)
    predictions: tuple[ComponentPrediction, ...] = Field(min_length=1, max_length=10)
    ood_score: float = Field(ge=0.0)
    abnormal_features: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class RootCauseHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str
    label: str
    component: str
    description: str
    score: float = Field(ge=0.0, le=1.0)
    status: str = Field(pattern=r"^(active|weakened|excluded|confirmed|unresolved)$")
    supporting_observations: tuple[str, ...] = ()
    contradicting_observations: tuple[str, ...] = ()


class InspectionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str
    title: str
    question: str
    purpose: str
    source: str
    allowed_outcomes: tuple[str, ...] = ("present", "absent", "uncertain", "not_checked")


class EngineerObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: str
    action_id: str
    outcome: str = Field(pattern=r"^(present|absent|uncertain|not_checked)$")
    notes: str = Field(default="", max_length=2000)
    created_at: str


class AgentTraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    agent_role: str
    event_type: str
    summary: str
    status: str = Field(pattern=r"^(ok|waiting|completed|error)$")
    created_at: str


class KnowledgeEnhancedReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1, max_length=2000)
    root_cause_analysis: str = Field(min_length=1, max_length=4000)
    maintenance_support: tuple[str, ...] = Field(min_length=1, max_length=10)
    limitations: tuple[str, ...] = Field(min_length=1, max_length=10)


class RootCauseCaseDraft(BaseModel):
    """Reviewable case distilled from one completed root-cause session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=8, max_length=80)
    status: CaseDraftStatus = CaseDraftStatus.DRAFT
    summary: str = Field(min_length=1, max_length=2000)
    source_rca_id: str
    initial_component: str
    leading_root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    observation_count: int = Field(ge=0)
    tags: tuple[str, ...] = Field(max_length=20)
    created_at: str
    published_at: str | None = None


class RootCauseSession(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rca_id: str
    status: RootCauseSessionStatus
    dataset: DatasetProfile
    model_result: ModelInferenceResult
    hypotheses: tuple[RootCauseHypothesis, ...]
    observations: tuple[EngineerObservation, ...]
    next_action: InspectionProposal | None
    trace: tuple[AgentTraceEvent, ...]
    report: KnowledgeEnhancedReport | None = None
    case_draft: RootCauseCaseDraft | None = None
    created_at: str
    updated_at: str


class RootCauseCoordination(BaseModel):
    """Strict private output expected from the coordinating LLM role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ordered_hypothesis_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    next_action_id: str | None
    rationale: str = Field(min_length=1, max_length=1200)
