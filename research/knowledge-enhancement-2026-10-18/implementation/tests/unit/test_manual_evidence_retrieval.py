from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from aerodiagnosis.adapters.persistence.sqlite import SQLiteDatabase
from aerodiagnosis.adapters.persistence.sqlite_context import SQLiteParentContextStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.domain import SourceKind, make_chunk_id, make_evidence_id
from aerodiagnosis.domain.knowledge_query import KnowledgeQueryContext
from aerodiagnosis.ingestion import DocumentIngestionService, DocumentManifest, ParserRegistry
from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.ports.context import ParentContext
from aerodiagnosis.retrieval.manual import ManualEvidenceRetriever, ManualRetrievalPolicy
from aerodiagnosis.retrieval.reranker import IdentityReranker, RankedText


class FixedCandidates(SQLiteVectorStore):
    def __init__(self, path: Path, matches: list[VectorMatch]) -> None:
        super().__init__(path)
        self.matches = matches

    def search(
        self, query: str, *, top_k: int = 5, active_version_ids: frozenset[str] | None = None
    ) -> list[VectorMatch]:
        del query, active_version_ids
        return self.matches[:top_k]


def test_persistent_bm25_filters_before_limit_handles_chinese_and_safe_query(
    tmp_path: Path,
) -> None:
    path = tmp_path / "knowledge.db"
    store = SQLiteLexicalStore(path)
    store.upsert(
        [
            VectorChunk("old", "d", "old-v", "CFM56 fan fan fan 裂纹检查"),
            VectorChunk("current", "d", "current-v", "CFM56 fan 叶片裂纹检查"),
        ]
    )
    reopened = SQLiteLexicalStore(path)
    assert (
        reopened.search("CFM56", top_k=1, active_version_ids=frozenset({"current-v"}))[
            0
        ].chunk.chunk_id
        == "current"
    )
    assert reopened.search("叶片裂纹", active_version_ids=frozenset({"current-v"}))[0].score > 0
    assert reopened.search("fan", active_version_ids=frozenset()) == []
    assert reopened.search('" ; DROP TABLE knowledge_lexical_chunks --') == []
    store.upsert([VectorChunk("current", "d", "current-v", "Revised borescope instruction")])
    assert reopened.search("CFM56", active_version_ids=frozenset({"current-v"})) == []
    assert reopened.delete_document("d") == 2
    assert store.count() == 0
    assert store.search("fan") == []


def test_lexical_backfill_preserves_existing_vector_data_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    vector = SQLiteVectorStore(path)
    vector.upsert([VectorChunk("existing", "manual", "v1", "Review EGT trend")])
    lexical = SQLiteLexicalStore(path)
    assert lexical.backfill_from_vectors() == 1
    assert lexical.backfill_from_vectors() == 0
    assert vector.count() == 1
    assert lexical.search("EGT")[0].chunk.chunk_id == "existing"


def test_hybrid_candidate_union_can_recover_lexical_only_source_and_reject_outside_snapshot(
    tmp_path: Path,
) -> None:
    bad = VectorChunk("bad", "unrelated", "v", "Unrelated fuel scheduling example")
    forged = VectorChunk("forged", "wrong", "not-active", "CFM56 fan inspection")
    vector = FixedCandidates(tmp_path / "v.db", [VectorMatch(forged, 1), VectorMatch(bad, 0.7)])
    sparse = SQLiteLexicalStore(tmp_path / "s.db")
    relevant = VectorChunk("match", "manual", "v", "CFM56 fan inspection procedure")
    sparse.upsert([relevant, bad])
    retriever = ManualEvidenceRetriever(
        vector_store=vector,
        lexical_store=sparse,
        policy=ManualRetrievalPolicy(candidate_k=5, expand_context=False),
    )
    results = retriever.search("CFM56 fan inspection", top_k=5, active_version_ids=frozenset({"v"}))
    assert {r.chunk.chunk_id for r in results} == {"match", "bad"}
    matching = next(r for r in results if r.chunk.chunk_id == "match")
    assert matching.chunk.metadata["retrieval"]["contributions"][0]["route"] == "sparse"
    assert 0 < matching.score <= 1
    vector_only = ManualEvidenceRetriever(
        vector_store=vector,
        lexical_store=sparse,
        policy=ManualRetrievalPolicy(strategy="vector_only"),
    )
    assert [
        r.chunk.chunk_id for r in vector_only.search("CFM56", active_version_ids=frozenset({"v"}))
    ] == ["bad"]


def test_conflicting_dense_mirror_cannot_vote_for_local_chunk_identity(tmp_path: Path) -> None:
    local = VectorChunk("same", "manual", "v", "CFM56 fan crack")
    forged = replace(local, document_id="other", content="Forged content")
    vector = FixedCandidates(tmp_path / "v.db", [VectorMatch(forged, 0.99)])
    sparse = SQLiteLexicalStore(tmp_path / "s.db")
    sparse.upsert([local])
    retriever = ManualEvidenceRetriever(vector_store=vector, lexical_store=sparse)
    hit = retriever.search("fan crack", active_version_ids=frozenset({"v"}))[0]
    assert hit.chunk.content == local.content
    assert [p["route"] for p in hit.chunk.metadata["retrieval"]["contributions"]] == ["sparse"]


def test_parent_context_is_stored_once_and_returned_with_its_own_source_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.db"
    parents = SQLiteParentContextStore(path)
    retriever = ManualEvidenceRetriever(
        vector_store=SQLiteVectorStore(path),
        lexical_store=SQLiteLexicalStore(path),
        context_store=parents,
        policy=ManualRetrievalPolicy(strategy="sparse_only"),
    )
    service = DocumentIngestionService(
        DocumentManifest(path),
        retriever,
        context_store=parents,
        parsers=ParserRegistry(parent_budget=1000, child_budget=100),
    )
    text = (
        "# CFM56 cruise\n\nOnly during stabilized cruise.\n\n"
        + "A fan inspection includes a borescope assessment of blade condition. " * 5
    )
    result = service.ingest(display_name="manual.md", content=text.encode(), document_id="manual")
    hits = retriever.search("borescope blade", active_version_ids=frozenset({result.version_id}))
    assert hits and len({h.chunk.chunk_id for h in hits}) == len(hits)
    expanded = next(
        h for h in hits if h.chunk.metadata["retrieval"]["context_expansion"] == "expanded"
    )
    coords = expanded.chunk.metadata["locator"]["coordinates"]
    assert text[coords["start"] : coords["end"]] == expanded.chunk.content
    assert "Only during stabilized cruise." in expanded.chunk.content
    assert expanded.chunk.metadata["evidence_id"] == make_evidence_id(
        SourceKind.DOCUMENT_CHUNK, expanded.chunk.chunk_id, expanded.chunk.metadata["locator"]
    )
    with SQLiteDatabase(path).connect() as connection:
        count = connection.execute("SELECT count(*) FROM knowledge_parent_contexts").fetchone()[0]
    assert count < result.chunk_count
    assert parents.get(expanded.chunk.chunk_id, version_id="other-version") is None


def test_parent_anchor_mismatch_and_budget_preserve_whole_child(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    parent_text = "Only at cruise. Inspect fan condition."
    locator = {"kind": "character_range", "coordinates": {"start": 0, "end": len(parent_text)}}
    pid = make_chunk_id("v", locator, parent_text)
    parent = ParentContext(pid, "manual", "v", parent_text, locator)
    contexts = SQLiteParentContextStore(path)
    contexts.upsert([parent])
    child = VectorChunk(
        "fine",
        "manual",
        "v",
        "Inspect fan condition.",
        {
            "parent_context_id": pid,
            "locator": {
                "kind": "character_range",
                "coordinates": {"start": 16, "end": len(parent_text)},
            },
        },
    )
    sparse = SQLiteLexicalStore(path)
    sparse.upsert([replace(child, content="Forged fan condition")])
    retriever = ManualEvidenceRetriever(
        vector_store=SQLiteVectorStore(path),
        lexical_store=sparse,
        context_store=contexts,
        policy=ManualRetrievalPolicy(strategy="sparse_only"),
    )
    hit = retriever.search("fan")[0]
    assert hit.chunk.content == "Forged fan condition"
    assert hit.chunk.metadata["retrieval"]["context_expansion"] == "parent_anchor_mismatch"
    long_text = "Only during cruise. " * 10
    long_loc = {"kind": "character_range", "coordinates": {"start": 0, "end": len(long_text)}}
    long_id = make_chunk_id("v", long_loc, long_text)
    contexts.upsert([ParentContext(long_id, "manual", "v", long_text, long_loc)])
    sparse.upsert([replace(child, metadata={**child.metadata, "parent_context_id": long_id})])
    bounded = ManualEvidenceRetriever(
        vector_store=SQLiteVectorStore(path),
        lexical_store=sparse,
        context_store=contexts,
        policy=ManualRetrievalPolicy(strategy="sparse_only", max_context_chars=100),
    )
    hit = bounded.search("fan")[0]
    assert hit.chunk.content == child.content
    assert hit.chunk.metadata["retrieval"]["context_expansion"] == "parent_over_budget"


def test_parent_store_rejects_cross_document_immutable_identity_collision(tmp_path: Path) -> None:
    store = SQLiteParentContextStore(tmp_path / "runtime.db")
    locator = {"kind": "character_range", "coordinates": {"start": 0, "end": 4}}
    parent = ParentContext(make_chunk_id("v", locator, "text"), "d", "v", "text", locator)
    assert store.upsert([parent, parent]) == 2
    with pytest.raises(ValueError, match="collision"):
        store.upsert([replace(parent, document_id="different")])
    with pytest.raises(ValueError, match="source"):
        store.upsert([replace(parent, content="altered")])


def test_diversity_and_identical_source_text_do_not_fill_budget_with_duplicates(
    tmp_path: Path,
) -> None:
    chunks = [VectorChunk(str(i), "same", "v", f"Fan inspection procedure {i}") for i in range(8)]
    chunks.append(VectorChunk("other", "second", "v2", "Fan inspection second document"))
    chunks.append(VectorChunk("dup", "second", "v2", "Fan inspection second document"))
    sparse = SQLiteLexicalStore(tmp_path / "runtime.db")
    sparse.upsert(chunks)
    retriever = ManualEvidenceRetriever(
        vector_store=SQLiteVectorStore(tmp_path / "runtime.db"),
        lexical_store=sparse,
        policy=ManualRetrievalPolicy(strategy="sparse_only", max_per_document=2),
    )
    hits = retriever.search("fan inspection", top_k=5)
    assert sum(h.chunk.document_id == "same" for h in hits) == 2
    assert sum(h.chunk.document_id == "second" for h in hits) == 1


def test_reranking_is_bounded_and_happens_before_context_expansion(tmp_path: Path) -> None:
    class AuditReranker(IdentityReranker):
        def __init__(self) -> None:
            self.observed: list[RankedText] = []

        def rerank(self, query: str, candidates: list[RankedText]) -> list[RankedText]:
            self.observed = candidates
            return list(reversed(candidates))

    reranker = AuditReranker()
    path = tmp_path / "runtime.db"
    chunks = [VectorChunk(str(i), str(i), "v", f"Fan inspection {i}. " * 3) for i in range(6)]
    sparse = SQLiteLexicalStore(path)
    sparse.upsert(chunks)
    retriever = ManualEvidenceRetriever(
        vector_store=SQLiteVectorStore(path),
        lexical_store=sparse,
        reranker=reranker,
        policy=ManualRetrievalPolicy(strategy="sparse_only", candidate_k=3, rerank_char_budget=110),
    )
    hits = retriever.search("fan", top_k=2)
    assert len(reranker.observed) <= 3
    assert sum(len(c.text) for c in reranker.observed) <= 110
    assert all(h.chunk.metadata["retrieval"]["fused_candidate_count"] == 3 for h in hits)


def test_failed_lexical_write_does_not_publish_new_version(tmp_path: Path) -> None:
    class FailingLexical(SQLiteLexicalStore):
        def upsert(self, chunks: list[VectorChunk]) -> int:
            raise RuntimeError("lexical unavailable")

    path = tmp_path / "runtime.db"
    manifest = DocumentManifest(path)
    vector = SQLiteVectorStore(path)
    initial = DocumentIngestionService(manifest, vector).ingest(
        display_name="manual.txt", content=b"Old active fan instructions", document_id="manual"
    )
    failing = ManualEvidenceRetriever(vector_store=vector, lexical_store=FailingLexical(path))
    with pytest.raises(RuntimeError, match="lexical unavailable"):
        DocumentIngestionService(manifest, failing).ingest(
            display_name="manual.txt", content=b"New instructions", document_id="manual"
        )
    assert manifest.active_version_ids() == frozenset({initial.version_id})


def test_production_profile_rejects_hashing_and_demo_but_allows_explicit_sparse(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="trained embedding"):
        RuntimeSettings(
            runtime_dir=tmp_path, database_path=tmp_path / "db", knowledge_profile="production"
        )
    with pytest.raises(ValidationError, match="demo seeding"):
        RuntimeSettings(
            runtime_dir=tmp_path,
            database_path=tmp_path / "db",
            knowledge_profile="production",
            manual_retrieval_strategy="sparse_only",
            seed_demo_content=True,
        )
    settings = RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "db",
        knowledge_profile="production",
        manual_retrieval_strategy="sparse_only",
    )
    assert not settings.seed_demo_content


def test_bootstrap_query_applies_applicability_before_both_candidate_indexes(
    tmp_path: Path,
) -> None:
    from aerodiagnosis.bootstrap import bootstrap
    from aerodiagnosis.domain import DocumentMetadata

    app = bootstrap(RuntimeSettings(runtime_dir=tmp_path, database_path=tmp_path / "runtime.db"))
    app.ingest_document.execute(
        display_name="a.txt",
        content=b"Fan inspection cruise condition",
        metadata=DocumentMetadata(
            applicability={
                "scope_status": "known",
                "engine_models": ["a"],
                "flight_phases": ["cruise"],
            }
        ),
    )
    assert (
        app.search_evidence.execute(
            "fan",
            knowledge_context=KnowledgeQueryContext(
                engine_model="a", flight_phase="takeoff", require_known=True
            ),
        )
        == ()
    )
    assert app.search_evidence.execute(
        "fan",
        knowledge_context=KnowledgeQueryContext(
            engine_model="a", flight_phase="cruise", require_known=True
        ),
    )


def test_dense_payload_is_verified_even_without_sparse_query_match(tmp_path: Path) -> None:
    source = VectorChunk("original", "manual", "active", "compressor pressure")
    mirror = replace(source, content="rotor counterfeit")
    sparse = SQLiteLexicalStore(tmp_path / "s.db")
    sparse.upsert([source])
    vector = FixedCandidates(tmp_path / "v.db", [VectorMatch(mirror, 0.99)])
    for strategy in ("vector_only", "hybrid_rrf"):
        retriever = ManualEvidenceRetriever(
            vector_store=vector,
            lexical_store=sparse,
            policy=ManualRetrievalPolicy(strategy=strategy),
        )
        assert retriever.search("rotor", active_version_ids=frozenset({"active"})) == []


def test_adaptive_overfetch_recovers_document_after_same_parent_candidates(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    contexts = SQLiteParentContextStore(path)
    text = "fan inspection. " * 50
    locator = {"kind": "character_range", "coordinates": {"start": 0, "end": len(text)}}
    parent_id = make_chunk_id("v", locator, text)
    contexts.upsert([ParentContext(parent_id, "first", "v", text, locator)])
    chunks = [
        VectorChunk(
            str(i),
            "first",
            "v",
            "fan inspection. ",
            {
                "parent_context_id": parent_id,
                "locator": {
                    "kind": "character_range",
                    "coordinates": {"start": i * 16, "end": (i + 1) * 16},
                },
            },
        )
        for i in range(50)
    ]
    chunks.append(VectorChunk("last", "second", "v", "fan spare document"))
    sparse = SQLiteLexicalStore(path)
    sparse.upsert(chunks)
    vector = FixedCandidates(path, [VectorMatch(c, 0.99 - i * 0.001) for i, c in enumerate(chunks)])
    retriever = ManualEvidenceRetriever(
        vector_store=vector,
        lexical_store=sparse,
        context_store=contexts,
        policy=ManualRetrievalPolicy(strategy="vector_only"),
    )
    hits = retriever.search("fan", top_k=2, active_version_ids=frozenset({"v"}))
    assert [h.chunk.document_id for h in hits] == ["first", "second"]


def test_incomplete_active_repair_raises_without_retiring_active_version(tmp_path: Path) -> None:
    class IncompleteVector(SQLiteVectorStore):
        incomplete = False

        def upsert(self, chunks):
            return 0 if self.incomplete else super().upsert(chunks)

    path = tmp_path / "runtime.db"
    vector = IncompleteVector(path)
    manifest = DocumentManifest(path)
    service = DocumentIngestionService(manifest, vector)
    initial = service.ingest(
        display_name="manual.txt", content=b"fan instructions", document_id="d"
    )
    vector.incomplete = True
    with pytest.raises(RuntimeError, match="repair was incomplete"):
        service.ingest(
            display_name="manual.txt",
            content=b"fan instructions",
            document_id="d",
            force_vector_upsert=True,
        )
    assert manifest.active_version_ids() == frozenset({initial.version_id})


def test_union_truncation_expands_even_when_both_routes_are_exhausted(tmp_path: Path) -> None:
    class FixedLexical(SQLiteLexicalStore):
        def search(self, query, *, top_k=5, active_version_ids=None):
            del query, active_version_ids
            return self.matches[:top_k]

    dense = [VectorChunk(f"d{i:02}", "same", "v", "fan inspection") for i in range(49)]
    sparse = [VectorChunk(f"s{i:02}", "same", "v", "fan inspection") for i in range(25)]
    sparse += [
        VectorChunk(f"s{i:02}", f"other{i}", "v", f"fan alternative {i}") for i in range(25, 49)
    ]
    lexical = FixedLexical(tmp_path / "s.db")
    lexical.upsert(dense + sparse)
    lexical.matches = [VectorMatch(c, 1) for c in sparse]
    vector = FixedCandidates(tmp_path / "v.db", [VectorMatch(c, 0.9) for c in dense])
    retriever = ManualEvidenceRetriever(vector_store=vector, lexical_store=lexical)
    hits = retriever.search("fan", top_k=2, active_version_ids=frozenset({"v"}))
    assert len(hits) == 2
    assert hits[0].chunk.document_id == "same"
    assert hits[1].chunk.document_id.startswith("other")


def test_dedup_retains_independent_sources_revisions_and_uncovered_tails(tmp_path):
    def chunk(key, doc, version, content, start=0):
        return VectorChunk(
            key,
            doc,
            version,
            content,
            {
                "locator": {
                    "kind": "text",
                    "coordinates": {"start": start, "end": start + len(content)},
                }
            },
        )

    values = [
        chunk("full", "a", "v1", "fan damage causes vibration"),
        chunk("inside", "a", "v1", "damage", 4),
        chunk("tail", "a", "v1", "vibration but not fire", 18),
        chunk("independent", "b", "v2", "fan damage causes vibration"),
        chunk("revision", "a", "v3", "fan damage causes vibration"),
        chunk("negative", "a", "v1", "fan damage does not cause fire", 100),
    ]
    vector = FixedCandidates(
        tmp_path / "v.db", [VectorMatch(c, 1 - i * 0.01) for i, c in enumerate(values)]
    )
    sparse = SQLiteLexicalStore(tmp_path / "s.db")
    sparse.upsert(values)
    r = ManualEvidenceRetriever(
        vector_store=vector,
        lexical_store=sparse,
        policy=ManualRetrievalPolicy(
            strategy="vector_only", max_per_document=0, expand_context=False
        ),
    )
    hits = r.search("fan damage", top_k=10)
    assert {h.chunk.chunk_id for h in hits} == {
        "full",
        "tail",
        "independent",
        "revision",
        "negative",
    }
    full = next(h for h in hits if h.chunk.chunk_id == "full")
    assert full.chunk.metadata["retrieval"]["duplicate_source_ranges"][0]["chunk_id"] == "inside"


def test_translated_child_never_inherits_larger_parent_locator(tmp_path):
    path = tmp_path / "v.db"
    parents = SQLiteParentContextStore(path)
    parent = ParentContext(
        "parent",
        "d",
        "v",
        "fan damage causes vibration",
        {"kind": "text", "coordinates": {"start": 0, "end": 27}},
    )
    parent = replace(parent, context_id=make_chunk_id("v", parent.locator, parent.content))
    parents.upsert([parent])
    child = VectorChunk(
        "child",
        "d",
        "v",
        "damage",
        {
            "parent_context_id": parent.context_id,
            "display_content_zh": "损伤",
            "locator": {"kind": "text", "coordinates": {"start": 4, "end": 10}},
        },
    )
    vector = FixedCandidates(path, [VectorMatch(child, 0.8)])
    sparse = SQLiteLexicalStore(path)
    sparse.upsert([child])
    r = ManualEvidenceRetriever(
        vector_store=vector,
        lexical_store=sparse,
        context_store=parents,
        policy=ManualRetrievalPolicy(strategy="vector_only"),
    )
    hit = r.search("damage")[0]
    assert hit.chunk.chunk_id == "child"
    assert hit.chunk.metadata["locator"]["coordinates"] == {"start": 4, "end": 10}
    assert hit.chunk.metadata["retrieval"]["context_expansion"] == "translated_child_preserved"
