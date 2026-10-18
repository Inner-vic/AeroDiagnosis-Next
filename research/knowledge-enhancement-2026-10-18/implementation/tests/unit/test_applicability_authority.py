from __future__ import annotations

from collections.abc import Sequence

import pytest

from aerodiagnosis.application.knowledge_applicability import assess_applicability, permits
from aerodiagnosis.application.use_cases import SearchEvidence
from aerodiagnosis.domain import DocumentApplicability, DocumentMetadata
from aerodiagnosis.domain.knowledge_query import KnowledgeQueryContext
from aerodiagnosis.ports import VectorChunk, VectorMatch


def metadata(model: str = "engine-a", phase: str | None = None) -> dict:
    return DocumentMetadata(
        applicability=DocumentApplicability(
            scope_status="known", engine_models=(model,), flight_phases=(phase,) if phase else ()
        )
    ).model_dump(mode="json")


class FilterIgnoringVectorStore:
    """Model a stale or faulty external mirror without any disk or network I/O."""

    backend_name = "synthetic"
    embedding_identity = "synthetic"

    def __init__(self, matches: list[VectorMatch]) -> None:
        self.matches = matches
        self.requested_versions: frozenset[str] | None = None

    def search(
        self, query: str, *, top_k: int = 5, active_version_ids: frozenset[str] | None = None
    ) -> list[VectorMatch]:
        self.requested_versions = active_version_ids
        return self.matches

    def upsert(self, chunks: Sequence[VectorChunk]) -> int:
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        return 0

    def count(self) -> int:
        return len(self.matches)


def match(version: str, scope: dict) -> VectorMatch:
    return VectorMatch(
        VectorChunk(
            "chunk-" + version,
            "doc",
            version,
            "EGT rise",
            {"document_metadata": scope, "locator": {"page": 1}},
        ),
        1.0,
    )


@pytest.mark.parametrize(
    "query",
    [
        KnowledgeQueryContext(engine_model="engine-a", require_known=True),
        KnowledgeQueryContext(flight_phase="cruise", require_known=True),
    ],
)
def test_missing_required_query_condition_is_unknown(query: KnowledgeQueryContext) -> None:
    status = assess_applicability(metadata(phase="cruise"), query)
    assert status == "unknown"
    assert not permits(status, query)
    assert permits(status, query.model_copy(update={"require_known": False}))


def test_explicit_conflict_wins_over_another_missing_query_condition() -> None:
    query = KnowledgeQueryContext(engine_model="engine-b")
    assert assess_applicability(metadata(phase="cruise"), query) == "conflict"
    assert (
        assess_applicability(metadata(), KnowledgeQueryContext(engine_model="engine-a"))
        == "matched"
    )


def test_manifest_matching_scope_overrides_stale_conflicting_mirror() -> None:
    store = FilterIgnoringVectorStore([match("v1", metadata("engine-b"))])
    authoritative = metadata()
    search = SearchEvidence(
        store, lambda: frozenset({"v1"}), version_metadata=lambda _: authoritative
    )
    found = search.execute(
        "EGT", knowledge_context=KnowledgeQueryContext(engine_model="engine-a", require_known=True)
    )
    assert len(found) == 1
    assert found[0].chunk.metadata["document_metadata"] == authoritative
    assert store.matches[0].chunk.metadata["document_metadata"] == metadata("engine-b")
    assert found[0].chunk.metadata["locator"] == {"page": 1}


def test_unknown_manifest_cannot_be_replaced_by_claimed_known_mirror() -> None:
    store = FilterIgnoringVectorStore([match("v1", metadata())])
    search = SearchEvidence(store, lambda: frozenset({"v1"}), version_metadata=lambda _: {})
    query = KnowledgeQueryContext(engine_model="engine-a")
    found = search.execute("EGT", knowledge_context=query)
    assert len(found) == 1
    assert found[0].chunk.metadata["document_metadata"] == {}
    assert assess_applicability(found[0].chunk.metadata["document_metadata"], query) == "unknown"
    assert (
        search.execute("EGT", knowledge_context=query.model_copy(update={"require_known": True}))
        == ()
    )


def test_missing_manifest_record_does_not_fall_back_to_mirror_scope() -> None:
    store = FilterIgnoringVectorStore([match("v1", metadata())])
    search = SearchEvidence(store, lambda: frozenset({"v1"}), version_metadata=lambda _: None)
    found = search.execute("EGT")
    assert found[0].chunk.metadata["document_metadata"] == {}


def test_external_mirror_must_honor_active_and_prefiltered_versions() -> None:
    store = FilterIgnoringVectorStore(
        [match("old", metadata()), match("conflicting", metadata()), match("right", metadata())]
    )
    records = {"right": metadata(), "conflicting": metadata("engine-b")}
    search = SearchEvidence(store, lambda: frozenset(records), version_metadata=records.get)
    found = search.execute(
        "EGT", knowledge_context=KnowledgeQueryContext(engine_model="engine-a", require_known=True)
    )
    assert store.requested_versions == frozenset({"right"})
    assert [result.chunk.version_id for result in found] == ["right"]
    unconstrained = search.execute("EGT")
    assert {result.chunk.version_id for result in unconstrained} == {"right", "conflicting"}


def test_no_manifest_fallback_still_filters_supplied_scope_and_active_versions() -> None:
    store = FilterIgnoringVectorStore(
        [match("old", metadata()), match("v1", metadata("engine-b")), match("v2", metadata())]
    )
    search = SearchEvidence(store, lambda: frozenset({"v1", "v2"}))
    found = search.execute("EGT", knowledge_context=KnowledgeQueryContext(engine_model="engine-a"))
    assert [result.chunk.version_id for result in found] == ["v2"]


@pytest.mark.parametrize(
    ("revision", "expected"),
    [("R2", "matched"), ("R1", "conflict"), (None, "unknown")],
)
def test_revision_condition_is_checked_on_authoritative_metadata(
    revision: str | None, expected: str
) -> None:
    record = {**metadata(), "source_revision": revision}
    query = KnowledgeQueryContext(engine_model="engine-a", source_revision=" R2 ")
    assert assess_applicability(record, query) == expected


def test_revision_conflict_excludes_generic_and_unknown_scope() -> None:
    query = KnowledgeQueryContext(source_revision="R2")
    for status in ("unknown", "generic"):
        record = DocumentMetadata(
            source_revision="R1", applicability=DocumentApplicability(scope_status=status)
        ).model_dump(mode="json")
        assert assess_applicability(record, query) == "conflict"
        assert not permits("conflict", query)


def test_revision_unknown_does_not_hide_model_conflict() -> None:
    query = KnowledgeQueryContext(engine_model="engine-b", source_revision="R2")
    assert assess_applicability(metadata(), query) == "conflict"
