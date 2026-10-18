from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese source and query fixtures.
import hashlib
from dataclasses import replace

import pytest

from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.ports import GraphEdge, GraphNode, VectorChunk, VectorMatch
from aerodiagnosis.retrieval.graph_sources import GraphSourceCandidates
from aerodiagnosis.retrieval.manual import ManualEvidenceRetriever, ManualRetrievalPolicy


class Graph:
    backend_name = "test-graph"

    def __init__(self, nodes, edges):
        self.nodes, self.edges, self.calls = nodes, edges, 0

    def snapshot(self, **kwargs):
        self.calls += 1
        return self.nodes, self.edges


def data():
    text = "发现压气机叶片裂纹，可能与疲劳有关。"
    chunk = VectorChunk(
        "c",
        "d",
        "v",
        text,
        {
            "locator": {
                "kind": "text",
                "coordinates": {
                    "start": 10,
                    "end": 10 + len(text),
                    "canonical_sha256": "source-sha",
                },
            }
        },
    )
    nodes = [
        GraphNode("n1", "压气机叶片", "Component", "", "document:d", "v"),
        GraphNode("n2", "疲劳", "Fault", "", "document:d", "v"),
    ]
    edge = GraphEdge(
        "e",
        "n1",
        "n2",
        "POSSIBLY_ASSOCIATED_WITH",
        "document:d",
        "v",
        properties={
            "chunk_id": "c",
            "evidence_text": text,
            "canonical_start": 10,
            "canonical_end": 10 + len(text),
            "canonical_sha256": "source-sha",
            "certainty": "possible",
            "validation_status": "necessary_source_checks_passed",
        },
    )
    return chunk, nodes, edge


def test_graph_returns_original_child_with_qualification_and_exact_source(tmp_path):
    chunk, nodes, edge = data()
    lexical = SQLiteLexicalStore(tmp_path / "db")
    lexical.upsert([chunk])
    graph = Graph(nodes, [edge])
    route = GraphSourceCandidates(graph, lexical.get_chunks)
    assert route.search("压气机", active_version_ids=None) == []
    assert graph.calls == 0
    result = route.search("压气机", active_version_ids=frozenset({"v"}))
    assert len(result) == 1 and result[0].chunk.content == chunk.content
    assert result[0].chunk.metadata["locator"] == chunk.metadata["locator"]
    evidence = result[0].chunk.metadata["graph_retrieval"]
    assert evidence["edges"][0]["certainty"] == "possible"
    assert evidence["edges"][0]["relation"] == edge.relation


@pytest.mark.parametrize(
    "change",
    [
        {"polarity": "negative"},
        {"deprecated": True},
        {"canonical_start": 11},
        {"canonical_sha256": "wrong"},
        {"evidence_text": "没有裂纹"},
        {"validation_status": "failed"},
        {"chunk_id": "missing"},
    ],
)
def test_graph_rejects_stale_unverified_or_forged_source(tmp_path, change):
    chunk, nodes, edge = data()
    lexical = SQLiteLexicalStore(tmp_path / "db")
    lexical.upsert([chunk])
    edge = replace(edge, properties={**edge.properties, **change})
    route = GraphSourceCandidates(Graph(nodes, [edge]), lexical.get_chunks)
    assert route.search("压气机", active_version_ids=frozenset({"v"})) == []


def test_graph_rejects_cross_version_endpoints_and_wrong_document(tmp_path):
    chunk, nodes, edge = data()
    lexical = SQLiteLexicalStore(tmp_path / "db")
    lexical.upsert([chunk])
    graph = Graph([replace(nodes[0], version_id="old"), nodes[1]], [edge])
    assert (
        GraphSourceCandidates(graph, lexical.get_chunks).search(
            "压气机", active_version_ids=frozenset({"v"})
        )
        == []
    )
    graph = Graph(nodes, [replace(edge, source_ref="document:other")])
    assert (
        GraphSourceCandidates(graph, lexical.get_chunks).search(
            "压气机", active_version_ids=frozenset({"v"})
        )
        == []
    )


class Route:
    backend_name = embedding_identity = "test"

    def __init__(self, chunks):
        self.chunks, self.calls = chunks, 0

    def search(self, query, *, top_k=5, active_version_ids=None):
        self.calls += 1
        return [VectorMatch(c, 1.0) for c in self.chunks][:top_k]


def test_third_route_uses_same_source_identity_without_duplicates(tmp_path):
    chunk, nodes, edge = data()
    other = replace(chunk, chunk_id="other", document_id="other", content="压气机检查")
    lexical = SQLiteLexicalStore(tmp_path / "db")
    lexical.upsert([chunk, other])
    graph = Graph(nodes, [edge])
    route = GraphSourceCandidates(graph, lexical.get_chunks)
    vector = Route([other, chunk])
    base = ManualEvidenceRetriever(
        vector_store=vector,
        lexical_store=lexical,
        graph_candidates=route,
        policy=ManualRetrievalPolicy(graph_weight=0, expand_context=False),
    )
    base.search("压气机", active_version_ids=frozenset({"v"}))
    assert graph.calls == 0
    triple = ManualEvidenceRetriever(
        vector_store=vector,
        lexical_store=lexical,
        graph_candidates=route,
        policy=ManualRetrievalPolicy(graph_weight=0.25, vector_weight=0.25, expand_context=False),
    )
    results = triple.search("压气机", active_version_ids=frozenset({"v"}))
    assert len({r.chunk.chunk_id for r in results}) == len(results)
    matched = next(r for r in results if r.chunk.chunk_id == "c")
    assert {c["route"] for c in matched.chunk.metadata["retrieval"]["contributions"]} == {
        "vector",
        "sparse",
        "graph",
    }
    assert matched.chunk.metadata["retrieval"]["graph_source"]["edges"][0]["edge_id"] == "e"
    assert triple.with_reranker(base._reranker).search(
        "压气机", active_version_ids=frozenset({"v"})
    )


def test_live_bootstrap_reads_chinese_policy_and_weights(tmp_path, monkeypatch):
    from aerodiagnosis.adapters import persistence
    from aerodiagnosis.adapters.embeddings import HashingEmbeddingProvider
    from aerodiagnosis.bootstrap import bootstrap
    from aerodiagnosis.config import RuntimeSettings

    monkeypatch.setattr(
        persistence, "create_embedding_provider", lambda s: HashingEmbeddingProvider(384)
    )
    (tmp_path / ".env.local").write_text(
        "AERODIAGNOSIS_EMBEDDING_BACKEND=local_e5\n"
        "AERODIAGNOSIS_EMBEDDING_DIMENSIONS=384\n"
        f"AERODIAGNOSIS_EMBEDDING_MODEL_PATH={tmp_path}\n"
        "AERODIAGNOSIS_EMBEDDING_INPUT_POLICY=chinese_display\n"
        "AERODIAGNOSIS_MANUAL_VECTOR_WEIGHT=0.25\n"
        "AERODIAGNOSIS_MANUAL_SPARSE_WEIGHT=1\n"
        "AERODIAGNOSIS_MANUAL_GRAPH_WEIGHT=0\n"
        "AERODIAGNOSIS_MANUAL_RERANKER=identity\n",
        encoding="utf-8",
    )
    settings = RuntimeSettings.from_env(tmp_path)
    app = bootstrap(settings)
    ingested = app.ingest_document.execute(
        display_name="test.txt", content="压气机叶片出现裂纹，需要检查。".encode()
    )
    results = app.search_evidence.execute("压气机")
    assert results and settings.manual_vector_weight == 0.25
    assert results[0].chunk.metadata["retrieval"]["policy"]["vector_weight"] == 0.25
    assert app.get_runtime_status.execute().vector_embedding_identity.endswith(":chinese_display@1")
    assert ingested.version_id


def test_force_repair_preserves_existing_english_translation(tmp_path):
    from aerodiagnosis.adapters.embeddings import HashingEmbeddingProvider
    from aerodiagnosis.adapters.persistence.sqlite_chinese_vector import SQLiteChineseVectorStore
    from aerodiagnosis.ingestion import DocumentIngestionService, DocumentManifest

    def translate(chunks):
        assert all(
            not c.metadata.get("display_content_zh")
            or c.metadata["display_content_zh"] == "压气机叶片裂纹。"
            for c in chunks
        )
        return tuple(
            replace(
                c,
                metadata={
                    **c.metadata,
                    "display_content_zh": "压气机叶片裂纹.",
                    "display_translation_source_sha256": hashlib.sha256(
                        c.content.encode()
                    ).hexdigest(),
                },
            )
            for c in chunks
        )

    db = tmp_path / "db"
    store = SQLiteChineseVectorStore(db, embedding_provider=HashingEmbeddingProvider(384))
    service = DocumentIngestionService(DocumentManifest(db), store, chinese_chunks=translate)
    result = service.ingest(
        display_name="en.txt", content=b"Compressor blade crack.", document_id="d"
    )
    # The active repair must not lose its display or call an external translator.
    service._chinese_chunks = None
    again = service.ingest(
        display_name="en.txt",
        content=b"Compressor blade crack.",
        document_id="d",
        force_vector_upsert=True,
    )
    assert again.version_id == result.version_id
    assert (
        store.search("压气机", active_version_ids=frozenset({result.version_id}))[0].chunk.metadata[
            "display_content_zh"
        ]
        == "压气机叶片裂纹."
    )


@pytest.mark.parametrize("node_count,edge_count", [(2000, 1), (2, 4000)])
def test_graph_refuses_possibly_truncated_snapshot(node_count, edge_count):
    _, nodes, edge = data()
    graph = Graph([nodes[0]] * node_count, [edge] * edge_count)
    with pytest.raises(ValueError, match="truncated"):
        GraphSourceCandidates(graph, lambda keys: {}).search(
            "压气机", active_version_ids=frozenset({"v"})
        )
