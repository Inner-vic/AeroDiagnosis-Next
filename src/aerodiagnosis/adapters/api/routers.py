"""Thin HTTP mappings onto application use cases."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from aerodiagnosis.adapters.llm import OpenAICompatibleLanguageModel
from aerodiagnosis.adapters.persistence import ExternalStoreOutbox
from aerodiagnosis.adapters.persistence.sqlite import LATEST_SCHEMA_VERSION
from aerodiagnosis.application.diagnostic_workflow import (
    DiagnosisRunNotFound,
    IncompatibleWorkflowError,
)
from aerodiagnosis.bootstrap import Application
from aerodiagnosis.domain import (
    DiagnosisCommand,
    DiagnosisReport,
    HybridRetrievalResult,
    ModelPluginRecord,
    RootCauseSession,
)
from aerodiagnosis.evaluation import evaluate_retrieval
from aerodiagnosis.ingestion import DocumentParseError, IngestionError
from aerodiagnosis.ports import LanguageModelError
from aerodiagnosis.version import __version__

from .dependencies import get_application, require_operator
from .schemas import (
    CaseResponse,
    ConversationMessageResponse,
    DocumentCatalogResponse,
    EvidenceHit,
    FinalizeRootCauseRequest,
    GraphOverviewResponse,
    HybridRetrievalRequest,
    IngestDocumentRequest,
    IngestDocumentResponse,
    ProviderConfiguration,
    RegisterModelPluginRequest,
    ResumeDiagnosisRequest,
    RetrievalEvaluationRequest,
    RetrievalEvaluationResponse,
    RootCauseObservationRequest,
    RunDiagnosisRequest,
    SearchEvidenceRequest,
    SessionCreatedResponse,
    StartRootCauseRequest,
)

router = APIRouter(prefix="/api")
ApplicationDependency = Annotated[Application, Depends(get_application)]
OperatorDependency = Annotated[Application, Depends(require_operator)]


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "AeroDiagnosis", "version": __version__}


@router.get("/system", tags=["system"])
def system(application: ApplicationDependency) -> dict[str, Any]:
    state = application.get_runtime_status.execute()
    outbox = ExternalStoreOutbox(application.settings.database_path).counts()
    return {
        "version": __version__,
        "mode": "application",
        "diagnostic_agent": "llm_orchestrated_v2",
        "provider_configuration": "per_request_not_persisted",
        "default_provider": {
            "available": application.settings.default_provider_available,
            "model": application.settings.default_llm_model,
        },
        "database_schema": LATEST_SCHEMA_VERSION,
        "vector": {"backend": state.vector_backend, "chunks": state.vector_chunks},
        "graph": {
            "backend": state.graph_backend,
            "nodes": state.graph_nodes,
            "edges": state.graph_edges,
        },
        "cases": {"backend": state.case_backend, "cases": state.case_count},
        "documents": {
            "documents": state.document_count,
            "versions": state.version_count,
        },
        "memory": {
            "sessions": state.session_count,
            "messages": state.message_count,
        },
        "external_sync": {
            "mode": "transactional_outbox",
            "events": outbox,
        },
        "knowledge_enhancement": {
            "positioning": "model_result_plus_knowledge_root_cause_support",
            "models": len(application.knowledge_enhanced_diagnosis.list_models()),
            "root_cause_sessions": len(
                application.knowledge_enhanced_diagnosis.list_sessions(limit=500)
            ),
        },
    }


@router.get("/models", response_model=list[ModelPluginRecord], tags=["models"])
def list_model_plugins(application: ApplicationDependency) -> list[ModelPluginRecord]:
    return list(application.knowledge_enhanced_diagnosis.list_models())


@router.post(
    "/models",
    response_model=ModelPluginRecord,
    status_code=status.HTTP_201_CREATED,
    tags=["models"],
)
def register_model_plugin(
    request: RegisterModelPluginRequest,
    application: OperatorDependency,
) -> ModelPluginRecord:
    return application.knowledge_enhanced_diagnosis.register_model(request.manifest)


@router.get("/root-cause-sessions", response_model=list[RootCauseSession], tags=["root-cause"])
def list_root_cause_sessions(
    application: ApplicationDependency,
    limit: int = 50,
) -> list[RootCauseSession]:
    try:
        return list(application.knowledge_enhanced_diagnosis.list_sessions(limit=limit))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@router.get("/root-cause-sessions/{rca_id}", response_model=RootCauseSession, tags=["root-cause"])
def get_root_cause_session(
    rca_id: str,
    application: ApplicationDependency,
) -> RootCauseSession:
    try:
        return application.knowledge_enhanced_diagnosis.get(rca_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/root-cause-sessions",
    response_model=RootCauseSession,
    status_code=status.HTTP_201_CREATED,
    tags=["root-cause"],
)
def start_root_cause_session(
    request: StartRootCauseRequest,
    application: ApplicationDependency,
) -> RootCauseSession:
    model = _provider_model(request.provider, application)
    try:
        return application.knowledge_enhanced_diagnosis.start(
            display_name=request.display_name,
            csv_content=request.csv_content,
            model_id=request.model_id,
            language_model=model,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except LanguageModelError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post(
    "/root-cause-sessions/{rca_id}/observations",
    response_model=RootCauseSession,
    tags=["root-cause"],
)
def observe_root_cause_session(
    rca_id: str,
    request: RootCauseObservationRequest,
    application: ApplicationDependency,
) -> RootCauseSession:
    model = _provider_model(request.provider, application)
    try:
        return application.knowledge_enhanced_diagnosis.observe(
            rca_id,
            action_id=request.action_id,
            outcome=request.outcome,
            notes=request.notes,
            language_model=model,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except LanguageModelError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post(
    "/root-cause-sessions/{rca_id}/finalize",
    response_model=RootCauseSession,
    tags=["root-cause"],
)
def finalize_root_cause_session(
    rca_id: str,
    request: FinalizeRootCauseRequest,
    application: ApplicationDependency,
) -> RootCauseSession:
    model = _provider_model(request.provider, application)
    try:
        return application.knowledge_enhanced_diagnosis.finalize(rca_id, model)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except LanguageModelError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post(
    "/root-cause-sessions/{rca_id}/case",
    response_model=RootCauseSession,
    tags=["root-cause"],
)
def publish_root_cause_case(
    rca_id: str,
    application: ApplicationDependency,
) -> RootCauseSession:
    try:
        return application.knowledge_enhanced_diagnosis.publish_case(rca_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@router.post(
    "/documents",
    response_model=IngestDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["evidence"],
)
def ingest_document(
    request: IngestDocumentRequest,
    application: OperatorDependency,
) -> IngestDocumentResponse:
    try:
        result = application.ingest_document.execute(
            display_name=request.display_name,
            content=request.content.encode("utf-8"),
            document_id=request.document_id,
        )
    except (DocumentParseError, IngestionError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return IngestDocumentResponse.model_validate(asdict(result))


@router.post("/evidence/search", response_model=list[EvidenceHit], tags=["evidence"])
def search_evidence(
    request: SearchEvidenceRequest,
    application: ApplicationDependency,
) -> list[EvidenceHit]:
    matches = application.search_evidence.execute(request.query, top_k=request.top_k)
    return [
        EvidenceHit(
            evidence_id=str(match.chunk.metadata["evidence_id"]),
            document_id=match.chunk.document_id,
            version_id=match.chunk.version_id,
            content=match.chunk.content,
            score=match.score,
            locator=dict(match.chunk.metadata["locator"]),
        )
        for match in matches
    ]


@router.post("/retrieval/hybrid", tags=["retrieval"])
def hybrid_retrieval(
    request: HybridRetrievalRequest,
    application: ApplicationDependency,
) -> HybridRetrievalResult:
    try:
        return application.hybrid_retrieval.execute(
            request.query,
            top_k=request.top_k,
            min_relevance=request.min_relevance,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@router.post(
    "/retrieval/evaluate",
    response_model=RetrievalEvaluationResponse,
    tags=["retrieval"],
)
def evaluate_hybrid_retrieval(
    request: RetrievalEvaluationRequest,
    application: ApplicationDependency,
) -> RetrievalEvaluationResponse:
    try:
        retrieval = application.hybrid_retrieval.execute(
            request.query,
            top_k=request.top_k,
            min_relevance=request.min_relevance,
        )
        metrics = evaluate_retrieval(retrieval, set(request.relevant_evidence_ids))
        return RetrievalEvaluationResponse(retrieval=retrieval, metrics=metrics)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@router.get(
    "/documents",
    response_model=list[DocumentCatalogResponse],
    tags=["knowledge"],
)
def list_documents(application: ApplicationDependency) -> list[DocumentCatalogResponse]:
    return [
        DocumentCatalogResponse.model_validate(asdict(document))
        for document in application.browse_knowledge.execute()
    ]


@router.get("/graph", response_model=GraphOverviewResponse, tags=["knowledge"])
def browse_graph(
    application: ApplicationDependency,
    query: str = "",
    limit: int = 300,
) -> GraphOverviewResponse:
    try:
        overview = application.browse_graph.execute(query=query, limit=limit)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return GraphOverviewResponse.model_validate(asdict(overview))


@router.get("/cases", response_model=list[CaseResponse], tags=["knowledge"])
def browse_cases(
    application: ApplicationDependency,
    query: str = "",
    limit: int = 100,
) -> list[CaseResponse]:
    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="case limit must be between 1 and 500",
        )
    return [
        CaseResponse.model_validate(asdict(case))
        for case in application.browse_cases.execute(query=query, limit=limit)
    ]


@router.post("/diagnoses", response_model=DiagnosisReport, tags=["diagnosis"])
def run_diagnosis(
    request: RunDiagnosisRequest,
    application: ApplicationDependency,
) -> DiagnosisReport:
    model = _provider_model(request.provider, application)
    command = DiagnosisCommand.model_validate(request.model_dump(exclude={"provider"}))
    try:
        return application.run_diagnosis.start(command, model)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except LanguageModelError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post(
    "/diagnoses/{run_id}/resume",
    response_model=DiagnosisReport,
    tags=["diagnosis"],
)
def resume_diagnosis(
    run_id: str,
    request: ResumeDiagnosisRequest,
    application: ApplicationDependency,
) -> DiagnosisReport:
    model = _provider_model(request.provider, application)
    try:
        return application.run_diagnosis.resume(run_id, model)
    except DiagnosisRunNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except IncompatibleWorkflowError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except LanguageModelError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


def _provider_model(
    configuration: ProviderConfiguration | None,
    application: Application,
) -> OpenAICompatibleLanguageModel:
    if configuration is None:
        settings = application.settings
        if not settings.default_provider_available:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="configure a personal provider or a local default LLM provider",
            )
        assert settings.default_llm_base_url is not None
        assert settings.default_llm_model is not None
        assert settings.default_llm_api_key is not None
        return OpenAICompatibleLanguageModel(
            base_url=settings.default_llm_base_url,
            api_key=settings.default_llm_api_key.get_secret_value(),
            model=settings.default_llm_model,
        )
    return OpenAICompatibleLanguageModel(
        base_url=str(configuration.base_url),
        api_key=configuration.api_key.get_secret_value(),
        model=configuration.model,
        timeout_seconds=configuration.timeout_seconds,
    )


@router.post(
    "/sessions",
    response_model=SessionCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["memory"],
)
def create_session(application: ApplicationDependency) -> SessionCreatedResponse:
    return SessionCreatedResponse(session_id=application.conversation_sessions.create())


@router.get(
    "/sessions/{session_id}/messages",
    response_model=list[ConversationMessageResponse],
    tags=["memory"],
)
def get_session_messages(
    session_id: str,
    application: ApplicationDependency,
    limit: int = 20,
) -> list[ConversationMessageResponse]:
    try:
        messages = application.conversation_sessions.messages(session_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return [ConversationMessageResponse.model_validate(asdict(message)) for message in messages]


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["memory"],
)
def delete_session(session_id: str, application: ApplicationDependency) -> None:
    if not application.conversation_sessions.delete(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown conversation session: {session_id}",
        )
