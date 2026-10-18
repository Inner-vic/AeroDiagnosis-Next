"""Single composition root for every inbound adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from aerodiagnosis.adapters.persistence import (
    SQLiteCaseStore,
    SQLiteCheckpointStore,
    SQLiteConversationMemory,
    SQLiteKnowledgeEnhancementStore,
    SQLiteOperationLedger,
    create_embedding_provider,
    create_graph_store,
    create_vector_store,
)
from aerodiagnosis.adapters.persistence.sqlite import SQLiteDatabase
from aerodiagnosis.adapters.persistence.sqlite_context import SQLiteParentContextStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.application.agent_events import AgentEventSink
from aerodiagnosis.application.case_verification import RecordCaseVerification
from aerodiagnosis.application.demo_content import seed_demo_content
from aerodiagnosis.application.diagnostic_workflow import DiagnosticWorkflow
from aerodiagnosis.application.knowledge_enhancement import KnowledgeEnhancedDiagnosis
from aerodiagnosis.application.knowledge_graph_extractor import KnowledgeGraphExtractor
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
from aerodiagnosis.ingestion import DocumentIngestionService, DocumentManifest, ParserRegistry
from aerodiagnosis.ingestion.source_archive import SourceArchive
from aerodiagnosis.ports import LanguageModel
from aerodiagnosis.ports.graph import GraphExtractor
from aerodiagnosis.retrieval.graph_paths import GraphPathCandidates
from aerodiagnosis.retrieval.graph_sources import GraphSourceCandidates
from aerodiagnosis.retrieval.manual import ManualEvidenceRetriever, ManualRetrievalPolicy
from aerodiagnosis.retrieval.reranker import (
    E5SemanticReranker,
    IdentityReranker,
    LexicalReranker,
    LLMReranker,
)
from aerodiagnosis.tools import DiagnosticToolset


@dataclass(frozen=True, slots=True)
class Application:
    """The complete transport-independent application surface."""

    settings: RuntimeSettings
    source_archive: SourceArchive
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
    vector_backend = create_vector_store(configured)
    lexical_store = SQLiteLexicalStore(configured.database_path)
    lexical_store.backfill_from_vectors()
    context_store = SQLiteParentContextStore(configured.database_path)
    source_reranker = (
        E5SemanticReranker(create_embedding_provider(configured))
        if configured.manual_reranker == "local_e5"
        else None
    )
    graph_store = create_graph_store(configured)
    vector_store = ManualEvidenceRetriever(
        vector_store=vector_backend,
        lexical_store=lexical_store,
        context_store=context_store,
        policy=ManualRetrievalPolicy(
            strategy=cast(
                Literal["vector_only", "sparse_only", "hybrid_rrf"],
                configured.manual_retrieval_strategy,
            ),
            candidate_k=configured.manual_candidate_k,
            rrf_k=configured.manual_rrf_k,
            vector_weight=configured.manual_vector_weight,
            sparse_weight=configured.manual_sparse_weight,
            graph_weight=configured.manual_graph_weight,
            max_per_document=configured.manual_max_per_document,
            single_source_unlimited=configured.manual_single_source_unlimited,
            expand_context=configured.manual_expand_context,
        ),
        graph_candidates=(
            GraphPathCandidates(graph_store, lexical_store.get_chunks)
            if configured.manual_graph_mode == "path"
            else GraphSourceCandidates(graph_store, lexical_store.get_chunks)
        )
        if configured.manual_graph_weight > 0
        else None,
        reranker=source_reranker
        or (IdentityReranker() if configured.manual_reranker == "identity" else LexicalReranker()),
    )
    parsers = ParserRegistry(
        text_strategy=configured.knowledge_chunking,
        parent_budget=configured.knowledge_parent_budget,
        child_budget=configured.knowledge_child_budget,
        upstream_backend=configured.knowledge_parser_backend,
        tokenizer_path=configured.knowledge_tokenizer_path,
    )
    case_store = SQLiteCaseStore(configured.database_path)
    checkpoints = SQLiteCheckpointStore(configured.database_path)
    memory = SQLiteConversationMemory(configured.database_path)
    operation_ledger = SQLiteOperationLedger(configured.database_path)
    knowledge_enhancement_store = SQLiteKnowledgeEnhancementStore(configured.database_path)
    seed_ingestion = DocumentIngestionService(
        manifest, vector_store, parsers=parsers, context_store=context_store
    )
    if configured.seed_demo_content:
        seed_demo_content(
            ingestion=seed_ingestion,
            manifest=manifest,
            graph_store=graph_store,
            case_store=case_store,
        )
    source_archive = SourceArchive(configured.runtime_dir / "sources")
    graph_extractor: GraphExtractor = KnowledgeGraphExtractor()
    if configured.knowledge_extractor_backend == "langextract":
        from aerodiagnosis.adapters.llm.langextract_graph import LangExtractGraphExtractor
        from aerodiagnosis.adapters.llm.research_client import ResearchClient

        assert configured.default_llm_api_key is not None
        assert configured.default_llm_model is not None
        graph_extractor = LangExtractGraphExtractor(
            ResearchClient(
                api_key=configured.default_llm_api_key.get_secret_value(),
                runtime=configured.knowledge_budget_runtime,
                model=configured.default_llm_model,
                cap_cny=configured.knowledge_api_cap_cny,
            ),
            audit_root=configured.runtime_dir / "extraction-audits",
            stage=configured.knowledge_extraction_stage,
        )
    chinese_chunks = None
    if configured.knowledge_graph_chinese_display or (
        configured.embedding_input_policy == "chinese_display"
        and configured.default_provider_available
    ):
        from aerodiagnosis.adapters.llm.chinese_document import ChineseDocumentDisplay
        from aerodiagnosis.adapters.llm.chinese_graph import (
            ChineseDisplayGraphExtractor,
            ChineseGraphDisplay,
        )
        from aerodiagnosis.adapters.llm.research_client import ResearchClient

        assert configured.default_llm_api_key is not None
        assert configured.default_llm_model is not None
        display_client = ResearchClient(
            api_key=configured.default_llm_api_key.get_secret_value(),
            runtime=configured.knowledge_budget_runtime,
            model=configured.default_llm_model,
            cap_cny=configured.knowledge_api_cap_cny,
        )
        chinese_chunks = ChineseDocumentDisplay(
            display_client, stage=configured.knowledge_extraction_stage
        ).enrich
        if configured.knowledge_graph_chinese_display:
            graph_extractor = ChineseDisplayGraphExtractor(
                graph_extractor,
                ChineseGraphDisplay(display_client, stage=configured.knowledge_extraction_stage),
            )
    ingestion = DocumentIngestionService(
        manifest,
        vector_store,
        graph_store=graph_store,
        graph_extractor=graph_extractor,
        source_archive=source_archive,
        chinese_chunks=chinese_chunks,
        parsers=parsers,
        context_store=context_store,
    )

    def version_metadata(version_id: str) -> dict[str, Any] | None:
        record = manifest.get_version(version_id)
        return record.metadata.model_dump(mode="json") if record is not None else None

    def version_document_id(version_id: str) -> str | None:
        record = manifest.get_version(version_id)
        return record.document_id if record is not None else None

    tools = DiagnosticToolset(
        vector_store=vector_store,
        graph_store=graph_store,
        case_store=case_store,
        version_metadata=version_metadata,
        version_document_id=version_document_id,
        source_guided_graph=configured.knowledge_extractor_backend == "langextract",
        source_reranker=source_reranker,
        text_hybrid_only=configured.text_hybrid_only,
    )

    def workflow_factory(
        model: LanguageModel,
        event_sink: AgentEventSink | None = None,
    ) -> DiagnosticWorkflow:
        return DiagnosticWorkflow(
            tools=DiagnosticToolset(
                vector_store=(
                    vector_store
                    if source_reranker is not None or configured.manual_reranker == "identity"
                    else vector_store.with_reranker(LLMReranker(model))
                ),
                graph_store=graph_store,
                case_store=case_store,
                version_metadata=version_metadata,
                version_document_id=version_document_id,
                source_guided_graph=configured.knowledge_extractor_backend == "langextract",
                source_reranker=source_reranker,
                text_hybrid_only=configured.text_hybrid_only,
            ),
            language_model=model,
            active_versions=manifest.active_version_ids,
            checkpoints=checkpoints,
            memory=memory,
            ledger=operation_ledger,
            event_sink=event_sink,
        )

    return Application(
        settings=configured,
        source_archive=source_archive,
        ingest_document=IngestDocument(ingestion),
        browse_knowledge=BrowseKnowledge(manifest),
        browse_graph=BrowseGraph(graph_store, manifest.active_version_ids),
        browse_cases=BrowseCases(case_store),
        query_diagnostic_tool=QueryDiagnosticTool(tools, manifest.active_version_ids),
        run_diagnosis=RunDiagnosis(workflow_factory),
        record_case_verification=RecordCaseVerification(case_store),
        browse_operations=BrowseOperations(operation_ledger),
        conversation_sessions=ConversationSessions(memory),
        search_evidence=SearchEvidence(
            vector_store,
            manifest.active_version_ids,
            version_metadata=version_metadata,
            version_document_id=version_document_id,
        ),
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
