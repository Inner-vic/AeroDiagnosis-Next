"""Single composition root for every inbound adapter."""

from __future__ import annotations

from dataclasses import dataclass

from aerodiagnosis.adapters.persistence import (
    SQLiteCaseStore,
    SQLiteCheckpointStore,
    SQLiteConversationMemory,
    SQLiteKnowledgeEnhancementStore,
    SQLiteOperationLedger,
    create_graph_store,
    create_vector_store,
)
from aerodiagnosis.adapters.persistence.sqlite import SQLiteDatabase
from aerodiagnosis.application.case_verification import RecordCaseVerification
from aerodiagnosis.application.demo_content import seed_demo_content
from aerodiagnosis.application.diagnostic_workflow import DiagnosticWorkflow
from aerodiagnosis.application.knowledge_enhancement import KnowledgeEnhancedDiagnosis
from aerodiagnosis.application.use_cases import (
    BrowseCases,
    BrowseGraph,
    BrowseKnowledge,
    BrowseOperations,
    ConversationSessions,
    GetRuntimeStatus,
    HybridRetrieval,
    IngestDocument,
    QueryDiagnosticTool,
    RunDiagnosis,
    SearchEvidence,
)
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.ingestion import DocumentIngestionService, DocumentManifest
from aerodiagnosis.ports import LanguageModel
from aerodiagnosis.tools import DiagnosticToolset


@dataclass(frozen=True, slots=True)
class Application:
    """The complete transport-independent application surface."""

    settings: RuntimeSettings
    ingest_document: IngestDocument
    browse_knowledge: BrowseKnowledge
    browse_graph: BrowseGraph
    browse_cases: BrowseCases
    query_diagnostic_tool: QueryDiagnosticTool
    run_diagnosis: RunDiagnosis
    record_case_verification: RecordCaseVerification
    browse_operations: BrowseOperations
    conversation_sessions: ConversationSessions
    search_evidence: SearchEvidence
    hybrid_retrieval: HybridRetrieval
    get_runtime_status: GetRuntimeStatus
    knowledge_enhanced_diagnosis: KnowledgeEnhancedDiagnosis


def bootstrap(settings: RuntimeSettings | None = None) -> Application:
    configured = settings or RuntimeSettings.from_env()
    configured.prepare()
    database = SQLiteDatabase(configured.database_path)
    database.migrate()
    manifest = DocumentManifest(configured.database_path)
    vector_store = create_vector_store(configured)
    graph_store = create_graph_store(configured)
    case_store = SQLiteCaseStore(configured.database_path)
    checkpoints = SQLiteCheckpointStore(configured.database_path)
    memory = SQLiteConversationMemory(configured.database_path)
    operation_ledger = SQLiteOperationLedger(configured.database_path)
    knowledge_enhancement_store = SQLiteKnowledgeEnhancementStore(configured.database_path)
    ingestion = DocumentIngestionService(manifest, vector_store)
    if configured.seed_demo_content:
        seed_demo_content(
            ingestion=ingestion,
            manifest=manifest,
            graph_store=graph_store,
            case_store=case_store,
        )
    tools = DiagnosticToolset(
        vector_store=vector_store,
        graph_store=graph_store,
        case_store=case_store,
    )

    def workflow_factory(model: LanguageModel) -> DiagnosticWorkflow:
        return DiagnosticWorkflow(
            tools=tools,
            language_model=model,
            active_versions=manifest.active_version_ids,
            checkpoints=checkpoints,
            memory=memory,
            ledger=operation_ledger,
        )

    return Application(
        settings=configured,
        ingest_document=IngestDocument(ingestion),
        browse_knowledge=BrowseKnowledge(manifest),
        browse_graph=BrowseGraph(graph_store, manifest.active_version_ids),
        browse_cases=BrowseCases(case_store),
        query_diagnostic_tool=QueryDiagnosticTool(tools, manifest.active_version_ids),
        run_diagnosis=RunDiagnosis(workflow_factory),
        record_case_verification=RecordCaseVerification(case_store),
        browse_operations=BrowseOperations(operation_ledger),
        conversation_sessions=ConversationSessions(memory),
        search_evidence=SearchEvidence(vector_store, manifest.active_version_ids),
        hybrid_retrieval=HybridRetrieval(tools, manifest.active_version_ids),
        get_runtime_status=GetRuntimeStatus(
            vector_store=vector_store,
            graph_store=graph_store,
            case_store=case_store,
            memory=memory,
            manifest_counts=manifest.counts,
        ),
        knowledge_enhanced_diagnosis=KnowledgeEnhancedDiagnosis(
            knowledge_enhancement_store, case_store
        ),
    )
