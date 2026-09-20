"""HTTP-only request and response models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr

from aerodiagnosis.domain import DiagnosisCommand, HybridRetrievalResult, ModelPluginManifest
from aerodiagnosis.evaluation import RetrievalMetrics


class IngestDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=10 * 1024 * 1024)
    document_id: str | None = Field(default=None, min_length=1)


class IngestDocumentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    version_id: str
    revision: int
    status: str
    chunk_count: int
    evidence_ids: tuple[str, ...]
    reused: bool


class SearchEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)


class EvidenceHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    document_id: str
    version_id: str
    content: str
    score: float
    locator: dict[str, object]


class HybridRetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    min_relevance: float = Field(default=0.05, ge=0.0, le=1.0)


class RetrievalEvaluationRequest(HybridRetrievalRequest):
    relevant_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=100)


class RetrievalEvaluationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval: HybridRetrievalResult
    metrics: RetrievalMetrics


class ProviderConfiguration(BaseModel):
    """Per-request credential supplied by the local frontend and never persisted."""

    model_config = ConfigDict(extra="forbid")

    base_url: HttpUrl
    api_key: SecretStr = Field(min_length=1)
    model: str = Field(min_length=1, max_length=200)
    timeout_seconds: float = Field(default=60.0, gt=0, le=300)


class RunDiagnosisRequest(DiagnosisCommand):
    provider: ProviderConfiguration | None = None


class ResumeDiagnosisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderConfiguration | None = None


class RegisterModelPluginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: ModelPluginManifest


class StartRootCauseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=255)
    csv_content: str = Field(min_length=1, max_length=10 * 1024 * 1024)
    model_id: str = Field(min_length=3, max_length=80)
    provider: ProviderConfiguration | None = None


class RootCauseObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1, max_length=160)
    outcome: str = Field(pattern=r"^(present|absent|uncertain|not_checked)$")
    notes: str = Field(default="", max_length=2000)
    provider: ProviderConfiguration | None = None


class FinalizeRootCauseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderConfiguration | None = None


class VersionCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version_id: str
    document_id: str
    revision: int
    parser_version: str
    content_hash: str
    status: str


class DocumentCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    display_name: str
    active_version_id: str | None
    desired_revision: int
    created_at: str
    updated_at: str
    versions: tuple[VersionCatalogResponse, ...]


class GraphNodeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    name: str
    kind: str
    description: str
    source_ref: str
    version_id: str
    properties: dict[str, object]


class GraphEdgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str
    source_id: str
    target_id: str
    relation: str
    source_ref: str
    version_id: str
    confidence: float
    properties: dict[str, object]


class GraphOverviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: tuple[GraphNodeResponse, ...]
    edges: tuple[GraphEdgeResponse, ...]


class CaseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    version: int
    summary: str
    attributes: dict[str, object]


class CaseVerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str = Field(pattern=r"^(correct|partial|wrong)$")
    actual_cause: str = Field(default="", max_length=2000)
    actual_fault_ids: tuple[str, ...] = Field(default=(), max_length=100)
    notes: str = Field(default="", max_length=2000)


class SessionCreatedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str


class ConversationMessageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    session_id: str
    ordinal: int
    role: str
    content: str
    created_at: str
