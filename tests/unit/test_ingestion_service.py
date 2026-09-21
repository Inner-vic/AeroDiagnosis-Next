from __future__ import annotations

from pathlib import Path

import pytest

from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.domain import make_version_id
from aerodiagnosis.ingestion import (
    DocumentIngestionService,
    DocumentManifest,
    IngestionError,
    VersionStatus,
)


class ShortWriteVectorStore(SQLiteVectorStore):
    def upsert(self, chunks: object) -> int:
        return 0


def test_ingestion_publishes_stable_evidence_and_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "runtime.db"
    manifest = DocumentManifest(database)
    vectors = SQLiteVectorStore(database)
    service = DocumentIngestionService(manifest, vectors)

    created = service.ingest(display_name="manual.md", content=b"compressor stall warning")
    repeated = service.ingest(
        display_name="manual.md",
        content=b"compressor stall warning",
        document_id=created.document_id,
    )
    matches = vectors.search(
        "compressor stall",
        active_version_ids=manifest.active_version_ids(),
    )

    assert created.status is VersionStatus.ACTIVE
    assert repeated.version_id == created.version_id
    assert repeated.reused is True
    assert manifest.counts() == (1, 1)
    assert vectors.count() == 1
    assert matches[0].chunk.metadata["evidence_id"] == created.evidence_ids[0]

    forced = service.ingest(
        display_name="manual.md",
        content=b"compressor stall warning",
        document_id=created.document_id,
        force_vector_upsert=True,
    )

    assert forced.reused is True
    assert vectors.count() == 1


def test_new_revision_supersedes_old_visibility_without_deleting_history(tmp_path: Path) -> None:
    database = tmp_path / "runtime.db"
    manifest = DocumentManifest(database)
    vectors = SQLiteVectorStore(database)
    service = DocumentIngestionService(manifest, vectors)
    first = service.ingest(display_name="manual.txt", content=b"old compressor guidance")

    second = service.ingest(
        display_name="manual.txt",
        content=b"new turbine guidance",
        document_id=first.document_id,
    )
    visible = vectors.search("guidance", active_version_ids=manifest.active_version_ids())

    assert second.revision == 2
    assert manifest.get_version(first.version_id).status is VersionStatus.SUPERSEDED  # type: ignore[union-attr]
    assert {match.chunk.version_id for match in visible} == {second.version_id}
    assert vectors.count() == 2


def test_failed_index_is_recorded_but_never_becomes_visible(tmp_path: Path) -> None:
    database = tmp_path / "runtime.db"
    manifest = DocumentManifest(database)
    service = DocumentIngestionService(manifest, ShortWriteVectorStore(database))
    document_id = "document-1"
    content = b"compressor evidence"
    version_id = make_version_id(document_id, content, "plain-text@1")

    with pytest.raises(IngestionError, match="indexed 0 of 1"):
        service.ingest(
            display_name="manual.txt",
            content=content,
            document_id=document_id,
        )

    assert manifest.get_version(version_id).status is VersionStatus.FAILED  # type: ignore[union-attr]
    assert manifest.active_version_ids() == frozenset()


def test_ingestion_rejects_unsafe_names_size_and_failed_replays(tmp_path: Path) -> None:
    database = tmp_path / "runtime.db"
    manifest = DocumentManifest(database)
    service = DocumentIngestionService(manifest, SQLiteVectorStore(database), max_bytes=4)

    with pytest.raises(IngestionError, match="single safe file name"):
        service.ingest(display_name="../manual.txt", content=b"ok")
    with pytest.raises(IngestionError, match="4-byte"):
        service.ingest(display_name="manual.txt", content=b"large")

    failing = DocumentIngestionService(manifest, ShortWriteVectorStore(database))
    with pytest.raises(IngestionError, match="indexed"):
        failing.ingest(display_name="manual.txt", content=b"retry", document_id="doc")
    with pytest.raises(IngestionError, match="non-active state"):
        failing.ingest(display_name="manual.txt", content=b"retry", document_id="doc")
