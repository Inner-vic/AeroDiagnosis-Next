"""Framework-neutral application use cases."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from aerodiagnosis.application.knowledge_applicability import assess_applicability, permits
from aerodiagnosis.domain import (
    DiagnosisCommand,
    DiagnosisReport,
    DocumentMetadata,
    EvidenceItem,
    HybridRetrievalResult,
    ParameterObservation,
    RetrievalStrategy,
)
from aerodiagnosis.domain.knowledge_query import KnowledgeQueryContext
from aerodiagnosis.ingestion import DocumentIngestionService, IngestionResult
from aerodiagnosis.ingestion.manifest import DocumentCatalogRecord, DocumentManifest
from aerodiagnosis.ports import (
    CaseRecord,
    CaseStore,
    ConversationMemoryStore,
    ConversationMessage,
    GraphEdge,
    GraphNode,
    GraphStore,
    LanguageModel,
    OperationLedger,
    OperationRecord,
    VectorMatch,
    VectorStore,
)
from aerodiagnosis.tools import DiagnosticToolset

from .agent_events import AgentEventSink
from .diagnostic_workflow import DiagnosticWorkflow


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    vector_backend: str
    vector_embedding_identity: str
    vector_chunks: int
    graph_backend: str
    graph_nodes: int
    graph_edges: int
    case_backend: str
    case_count: int
    document_count: int
    version_count: int
    session_count: int
    message_count: int


@dataclass(frozen=True, slots=True)
class GraphOverview:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


class IngestDocument:
    def __init__(self, ingestion: DocumentIngestionService) -> None:
        self._ingestion = ingestion

    def execute(
        self,
        *,
        display_name: str,
        content: bytes,
        document_id: str | None = None,
        metadata: DocumentMetadata | None = None,
    ) -> IngestionResult:
        return self._ingestion.ingest(
            display_name=display_name,
            content=content,
            document_id=document_id,
            metadata=metadata,
        )


class BrowseKnowledge:
    def __init__(self, manifest: DocumentManifest) -> None:
        self._manifest = manifest

    def execute(self) -> tuple[DocumentCatalogRecord, ...]:
        return self._manifest.list_documents()


class BrowseGraph:
    def __init__(
        self,
        graph_store: GraphStore,
        active_versions: Callable[[], frozenset[str]],
    ) -> None:
        self._graph_store = graph_store
        self._active_versions = active_versions

    def execute(self, *, query: str = "", limit: int = 300) -> GraphOverview:
        nodes, edges = self._graph_store.snapshot(
            active_version_ids=self._active_versions(),
            limit=limit,
        )
        cleaned = query.strip().casefold()
        if cleaned:
            nodes = [
                node
                for node in nodes
                if any(
                    cleaned in str(text).casefold()
                    for text in (
                        node.name,
                        node.description,
                        node.properties.get("name_zh", ""),
                        node.properties.get("description_zh", ""),
                    )
                )
            ]
            node_ids = {node.node_id for node in nodes}
            edges = [
                edge for edge in edges if edge.source_id in node_ids and edge.target_id in node_ids
            ]
        return GraphOverview(nodes=tuple(nodes), edges=tuple(edges))


class BrowseCases:
    def __init__(self, case_store: CaseStore) -> None:
        self._case_store = case_store

    def execute(self, *, query: str = "", limit: int = 100) -> tuple[CaseRecord, ...]:
        if query.strip():
            return tuple(match.case for match in self._case_store.search(query, limit=limit))
        return tuple(self._case_store.list_cases(limit=limit))


class RunDiagnosis:
    def __init__(
        self,
        workflow_factory: Callable[[LanguageModel, AgentEventSink | None], DiagnosticWorkflow],
    ) -> None:
        self._workflow_factory = workflow_factory

    def start(self, command: DiagnosisCommand, language_model: LanguageModel) -> DiagnosisReport:
        return self._workflow_factory(language_model, None).start(command)

    def start_streaming(
        self,
        command: DiagnosisCommand,
        language_model: LanguageModel,
        event_sink: AgentEventSink,
    ) -> DiagnosisReport:
        return self._workflow_factory(language_model, event_sink).start(command)

    def resume(self, run_id: str, language_model: LanguageModel) -> DiagnosisReport:
        return self._workflow_factory(language_model, None).resume(run_id)


class QueryDiagnosticTool:
    """Transport-neutral, read-only access to the Agent's evidence tools."""

    def __init__(
        self,
        tools: DiagnosticToolset,
        active_versions: Callable[[], frozenset[str]],
    ) -> None:
        self._tools = tools
        self._active_versions = active_versions

    def execute(
        self,
        tool_name: str,
        *,
        query: str,
        top_k: int = 5,
        parameters: tuple[ParameterObservation, ...] = (),
        knowledge_context: KnowledgeQueryContext | None = None,
    ) -> tuple[EvidenceItem, ...]:
        command = DiagnosisCommand(
            session_id="external-read-only-tool",
            question=query,
            top_k=top_k,
            parameters=parameters,
            knowledge_context=knowledge_context or KnowledgeQueryContext(),
        )
        return self._tools.execute(
            tool_name,
            command=command,
            query=query,
            active_version_ids=self._active_versions(),
        )


class ConversationSessions:
    def __init__(self, memory: ConversationMemoryStore) -> None:
        self._memory = memory

    def create(self) -> str:
        session_id = str(uuid.uuid4())
        self._memory.create_session(session_id)
        return session_id

    def messages(self, session_id: str, *, limit: int = 20) -> tuple[ConversationMessage, ...]:
        return self._memory.recent(session_id, limit=limit)

    def delete(self, session_id: str) -> bool:
        return self._memory.delete_session(session_id)


class BrowseOperations:
    def __init__(self, ledger: OperationLedger) -> None:
        self._ledger = ledger

    def execute(self, run_id: str) -> tuple[OperationRecord, ...]:
        return self._ledger.list_run(run_id)


class SearchEvidence:
    def __init__(
        self,
        vector_store: VectorStore,
        active_versions: Callable[[], frozenset[str]],
        *,
        version_metadata: Callable[[str], Mapping[str, Any] | None] | None = None,
        version_document_id: Callable[[str], str | None] | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._active_versions = active_versions
        self._version_metadata = version_metadata
        self._version_document_id = version_document_id

    def execute(
        self,
        query: str,
        *,
        top_k: int = 5,
        knowledge_context: KnowledgeQueryContext | None = None,
    ) -> tuple[VectorMatch, ...]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        context = knowledge_context or KnowledgeQueryContext()
        versions = self._active_versions()
        if context.document_id is not None:
            if self._version_document_id is None:
                return ()
            versions = frozenset(
                v for v in versions if self._version_document_id(v) == context.document_id
            )
        if context.has_constraints and self._version_metadata is not None:
            versions = frozenset(
                version
                for version in versions
                if permits(assess_applicability(self._version_metadata(version), context), context)
            )
        limit = 200 if context.has_constraints and self._version_metadata is None else top_k
        matches = self._vector_store.search(query, top_k=limit, active_version_ids=versions)
        result = []
        for match in matches:
            if match.chunk.version_id not in versions or (
                context.document_id is not None and match.chunk.document_id != context.document_id
            ):
                # Defend against external stores returning records outside the filter.
                continue
            if self._version_metadata is not None:
                # The manifest is authoritative even if a mirror supplies another scope.
                metadata = dict(self._version_metadata(match.chunk.version_id) or {})
            else:
                raw = match.chunk.metadata.get("document_metadata")
                metadata = dict(raw) if isinstance(raw, Mapping) else {}
            if permits(assess_applicability(metadata, context), context):
                chunk = replace(
                    match.chunk,
                    metadata={**match.chunk.metadata, "document_metadata": metadata},
                )
                result.append(replace(match, chunk=chunk))
        return tuple(result[:top_k])


class HybridRetrieval:
    """Run the production multi-route retrieval policy without invoking an LLM."""

    def __init__(
        self,
        tools: DiagnosticToolset,
        active_versions: Callable[[], frozenset[str]],
    ) -> None:
        self._tools = tools
        self._active_versions = active_versions

    def execute(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_relevance: float = 0.05,
        strategy: RetrievalStrategy = RetrievalStrategy.WEIGHTED_RRF,
        rrf_k: int = 60,
        route_weights: dict[str, float] | None = None,
        enforce_route_coverage: bool = True,
        knowledge_context: KnowledgeQueryContext | None = None,
    ) -> HybridRetrievalResult:
        command = DiagnosisCommand(
            session_id="hybrid-retrieval-analysis",
            question=query,
            top_k=top_k,
            min_relevance=min_relevance,
            knowledge_context=knowledge_context or KnowledgeQueryContext(),
        )
        return self._tools.hybrid_search(
            command=command,
            query=query,
            active_version_ids=self._active_versions(),
            strategy=strategy,
            rrf_k=rrf_k,
            route_weights=route_weights,
            enforce_route_coverage=enforce_route_coverage,
        )


class GetRuntimeStatus:
    def __init__(
        self,
        *,
        vector_store: VectorStore,
        graph_store: GraphStore,
        case_store: CaseStore,
        memory: ConversationMemoryStore,
        manifest_counts: Callable[[], tuple[int, int]],
    ) -> None:
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._case_store = case_store
        self._memory = memory
        self._manifest_counts = manifest_counts

    def execute(self) -> RuntimeStatus:
        nodes, edges = self._graph_store.counts()
        documents, versions = self._manifest_counts()
        sessions, messages = self._memory.counts()
        return RuntimeStatus(
            vector_backend=self._vector_store.backend_name,
            vector_embedding_identity=self._vector_store.embedding_identity,
            vector_chunks=self._vector_store.count(),
            graph_backend=self._graph_store.backend_name,
            graph_nodes=nodes,
            graph_edges=edges,
            case_backend=self._case_store.backend_name,
            case_count=self._case_store.count(),
            document_count=documents,
            version_count=versions,
            session_count=sessions,
            message_count=messages,
        )
