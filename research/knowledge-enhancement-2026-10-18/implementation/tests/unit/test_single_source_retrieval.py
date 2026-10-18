from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aerodiagnosis.adapters.api.app import create_app
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.domain import DocumentMetadata
from aerodiagnosis.domain.knowledge_query import KnowledgeQueryContext
from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.retrieval.manual import ManualEvidenceRetriever, ManualRetrievalPolicy


class CountedVector(SQLiteVectorStore):
    def __init__(self, path: Path, chunks: list[VectorChunk]):
        super().__init__(path)
        self.chunks = chunks
        self.calls = []

    def search(self, query, *, top_k=5, active_version_ids=None):
        self.calls.append(top_k)
        return [
            VectorMatch(c, 0.9 - i * 0.001)
            for i, c in enumerate(self.chunks)
            if active_version_ids is None or c.version_id in active_version_ids
        ][:top_k]


def retriever(tmp_path, chunks, **policy):
    store = CountedVector(tmp_path / "vectors.db", chunks)
    lexical = SQLiteLexicalStore(tmp_path / "lexical.db")
    lexical.upsert(chunks)
    return ManualEvidenceRetriever(
        vector_store=store,
        lexical_store=lexical,
        policy=ManualRetrievalPolicy(strategy="vector_only", expand_context=False, **policy),
    ), store


def test_single_explicit_version_can_return_more_than_three_whole_chunks(tmp_path):
    chunks = [
        VectorChunk(str(i), "doc", "v", f"Unique fan inspection evidence {i}") for i in range(12)
    ]
    search, store = retriever(tmp_path, chunks)
    hits = search.search("fan inspection", top_k=8, active_version_ids=frozenset({"v"}))
    assert len(hits) == 8 and len(store.calls) == 1
    assert all(h.chunk.metadata["retrieval"]["effective_max_per_document"] == 0 for h in hits)
    assert all(
        h.chunk.metadata["retrieval"]["document_limit_reason"] == "explicit_single_source"
        for h in hits
    )


def test_global_request_does_not_relax_limit_from_one_source_in_candidate_pool(tmp_path):
    chunks = [
        VectorChunk(str(i), "doc", "v", f"Unique fan inspection evidence {i}") for i in range(8)
    ]
    search, _ = retriever(tmp_path, chunks)
    for scope in [None, frozenset({"v", "other"})]:
        hits = search.search("fan", top_k=8, active_version_ids=scope)
        assert len(hits) == 3
        assert hits[0].chunk.metadata["retrieval"]["effective_max_per_document"] == 3


def test_policy_can_replay_legacy_single_source_cap_without_futile_widening(tmp_path):
    chunks = [VectorChunk(str(i), "doc", "v", f"fan evidence {i}") for i in range(80)]
    search, store = retriever(tmp_path, chunks, single_source_unlimited=False)
    assert len(search.search("fan", top_k=8, active_version_ids=frozenset({"v"}))) == 3
    assert len(store.calls) == 1


def test_scope_upper_bound_stops_impossible_diversity_top_k(tmp_path):
    chunks = [VectorChunk(str(i), str(i % 2), str(i % 2), f"fan evidence {i}") for i in range(40)]
    search, store = retriever(tmp_path, chunks, max_per_document=1, candidate_k=4)
    hits = search.search("fan", top_k=4, active_version_ids=frozenset({"0", "1"}))
    assert len(hits) == 2 and store.calls == [4]


def test_source_identity_is_case_sensitive_and_independent_of_applicability():
    context = KnowledgeQueryContext(document_id=" Case-Sensitive-ID ")
    assert context.document_id == "Case-Sensitive-ID"
    assert not context.has_constraints


def test_document_filter_is_applied_before_both_retrieval_routes_and_in_api(tmp_path):
    settings = RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "knowledge.sqlite",
        manual_reranker="identity",
        text_hybrid_only=True,
        manual_expand_context=False,
        rate_limit_enabled=False,
    )
    app = bootstrap(settings)
    body = "\n\n".join(
        f"Fan inspection item {i}. " + ("Inspect engine blades and record findings. " * 10)
        for i in range(12)
    )
    a = app.ingest_document.execute(
        document_id="Source-A",
        display_name="a.txt",
        content=body.encode(),
        metadata=DocumentMetadata(source_revision="R1"),
    )
    app.ingest_document.execute(
        document_id="Source-B", display_name="b.txt", content=("Fan inspection " + body).encode()
    )
    assert a.chunk_count > 8
    context = KnowledgeQueryContext(document_id="Source-A")
    hits = app.search_evidence.execute("fan inspection", top_k=8, knowledge_context=context)
    assert len(hits) == 8 and {h.chunk.document_id for h in hits} == {"Source-A"}
    assert not app.search_evidence.execute(
        "fan", knowledge_context=KnowledgeQueryContext(document_id="source-a")
    )
    assert not app.search_evidence.execute(
        "fan", knowledge_context=KnowledgeQueryContext(document_id="Source-A", source_revision="R2")
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/retrieval/hybrid",
            json={
                "query": "fan inspection",
                "top_k": 8,
                "min_relevance": 0,
                "knowledge_context": {"document_id": "Source-A"},
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert len(result["hits"]) == 8
        assert result["hits"][0]["evidence"]["locator"]["manual_retrieval"]["route_candidates"] == {
            "vector": a.chunk_count,
            "sparse": a.chunk_count,
        }
        assert {h["evidence"]["document_id"] for h in result["hits"]} == {"Source-A"}
        assert all(
            h["evidence"]["locator"]["manual_retrieval"]["effective_max_per_document"] == 0
            for h in result["hits"]
        )
        missing = client.post(
            "/api/retrieval/hybrid",
            json={"query": "fan", "knowledge_context": {"document_id": "missing"}},
        )
        assert missing.status_code == 200 and not missing.json()["hits"]


@pytest.mark.parametrize(
    "version,resolver,expected", [("vA", True, 1), ("vB", True, 0), ("vA", False, 0)]
)
def test_external_tool_cannot_forge_requested_document_identity(
    tmp_path, version, resolver, expected
):
    import hashlib

    from aerodiagnosis.adapters.persistence import SQLiteCaseStore
    from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
    from aerodiagnosis.domain import DiagnosisCommand, EvidenceItem
    from aerodiagnosis.tools import DiagnosticToolset

    path = tmp_path / "external.sqlite"
    item = EvidenceItem(
        evidence_id="f" * 64,
        source_kind="document_chunk",
        source_ref="chunk",
        document_id="Source-A",
        version_id=version,
        excerpt="Fan source evidence.",
        content_hash=hashlib.sha256(b"Fan source evidence.").hexdigest(),
        locator={},
        score=0.9,
    )
    tools = DiagnosticToolset(
        vector_store=SQLiteVectorStore(path),
        graph_store=SQLiteGraphStore(path),
        case_store=SQLiteCaseStore(path),
        external_tools={"search_manual_chunks": lambda *args: (item,)},
        version_document_id=(lambda v: {"vA": "Source-A", "vB": "Source-B"}.get(v))
        if resolver
        else None,
    )
    result = tools.execute(
        "search_manual_chunks",
        command=DiagnosisCommand(
            session_id="s",
            question="fan",
            knowledge_context=KnowledgeQueryContext(document_id="Source-A"),
        ),
        query="fan",
        active_version_ids=frozenset({"vA", "vB"}),
    )
    assert len(result) == expected
