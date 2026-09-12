"""Framework-neutral application use cases."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass

from aerodiagnosis.domain import (
    DiagnosisCommand,
    DiagnosisReport,
    EvidenceItem,
    HybridRetrievalResult,
    ParameterObservation,
)
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
    VectorMatch,
    VectorStore,
)
from aerodiagnosis.tools import DiagnosticToolset

from .diagnostic_workflow import DiagnosticWorkflow


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    vector_backend: str
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
    ) -> IngestionResult:
        return self._ingestion.ingest(
            display_name=display_name,
            content=content,
            document_id=document_id,
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
                if cleaned in node.name.casefold() or cleaned in node.description.casefold()
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
        workflow_factory: Callable[[LanguageModel], DiagnosticWorkflow],
    ) -> None:
        self._workflow_factory = workflow_factory

    def start(self, command: DiagnosisCommand, language_model: LanguageModel) -> DiagnosisReport:
        return self._workflow_factory(language_model).start(command)

    def resume(self, run_id: str, language_model: LanguageModel) -> DiagnosisReport:
        return self._workflow_factory(language_model).resume(run_id)


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
    ) -> tuple[EvidenceItem, ...]:
        command = DiagnosisCommand(
            session_id="external-read-only-tool",
            question=query,
            top_k=top_k,
            parameters=parameters,
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


class SearchEvidence:
    def __init__(
        self,
        vector_store: VectorStore,
        active_versions: Callable[[], frozenset[str]],
    ) -> None:
        self._vector_store = vector_store
        self._active_versions = active_versions

    def execute(self, query: str, *, top_k: int = 5) -> tuple[VectorMatch, ...]:
        if not query.strip():
            raise ValueError("query must not be empty")
        return tuple(
            self._vector_store.search(
                query,
                top_k=top_k,
                active_version_ids=self._active_versions(),
            )
        )


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
    ) -> HybridRetrievalResult:
        command = DiagnosisCommand(
            session_id="hybrid-retrieval-analysis",
            question=query,
            top_k=top_k,
            min_relevance=min_relevance,
        )
        return self._tools.hybrid_search(
            command=command,
            query=query,
            active_version_ids=self._active_versions(),
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
