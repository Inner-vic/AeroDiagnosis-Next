from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from pydantic import ValidationError

from aerodiagnosis.adapters.api.routers import ingest_document, list_documents
from aerodiagnosis.adapters.api.schemas import IngestDocumentRequest
from aerodiagnosis.adapters.persistence.sqlite import MIGRATIONS
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.cli import ingest_file, main
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.domain import DocumentApplicability, DocumentMetadata, make_version_id
from aerodiagnosis.ingestion import DocumentIngestionService, DocumentManifest
from aerodiagnosis.ingestion.parsers import ParserRegistry


def test_metadata_is_canonical_and_validated() -> None:
    a = DocumentMetadata(
        applicability=DocumentApplicability(
            scope_status="known",
            engine_models=(" CFM56-7B ", "cfm56-7b"),
            flight_phases=("Cruise", "Takeoff"),
        )
    )
    b = DocumentMetadata.model_validate(
        {
            "applicability": {
                "flight_phases": ["takeoff", "cruise"],
                "engine_models": ["cfm56-7b"],
                "scope_status": "known",
            }
        }
    )
    assert a.canonical_json() == b.canonical_json()
    assert a.bind_version_id("legacy") == b.bind_version_id("legacy")
    assert DocumentMetadata().bind_version_id("legacy") == "legacy"
    assert (
        DocumentMetadata(
            applicability=DocumentApplicability(scope_status="generic")
        ).bind_version_id("legacy")
        != "legacy"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"scope_status": "known"},
        {"scope_status": "generic", "engine_models": ["engine"]},
        {"scope_status": "unknown", "flight_phases": ["cruise"]},
        {"scope_status": "known", "engine_models": [" "]},
    ],
)
def test_invalid_scope_is_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DocumentApplicability.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"source_revision": " "},
        {"source_url": "file:///local/manual"},
        {"schema_version": "2"},
        {"unrecognized": "field"},
    ],
)
def test_invalid_metadata_is_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DocumentMetadata.model_validate(payload)


def test_metadata_change_versions_content_and_preserves_history(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    manifest = DocumentManifest(path)
    vectors = SQLiteVectorStore(path)
    service = DocumentIngestionService(manifest, vectors)
    content = b"Compressor deterioration causes elevated EGT."
    first = service.ingest(display_name="manual.txt", content=content)
    parsed = ParserRegistry().parse("manual.txt", content)
    assert first.version_id == make_version_id(
        first.document_id, parsed.canonical_content, parsed.parser_version
    )
    metadata = DocumentMetadata(
        source_type="manual",
        source_revision="R2",
        applicability=DocumentApplicability(scope_status="known", engine_models=("CFM56",)),
    )
    second = service.ingest(
        display_name="manual.txt", content=content, document_id=first.document_id, metadata=metadata
    )
    repeated = service.ingest(
        display_name="manual.txt", content=content, document_id=first.document_id, metadata=metadata
    )
    assert second.version_id != first.version_id
    assert second.evidence_ids != first.evidence_ids
    assert repeated.reused and repeated.version_id == second.version_id
    assert manifest.get_version(first.version_id).metadata == DocumentMetadata()  # type: ignore[union-attr]
    assert manifest.get_version(second.version_id).metadata == metadata  # type: ignore[union-attr]
    match = vectors.search("elevated EGT", active_version_ids=manifest.active_version_ids())[0]
    assert match.chunk.metadata["document_metadata"] == metadata.model_dump(mode="json")
    assert manifest.list_documents()[0].versions[0].metadata == metadata


def test_v7_migration_reads_old_versions_as_unknown(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with closing(sqlite3.connect(path)) as connection:
        for migration in MIGRATIONS[:7]:
            connection.executescript(migration.sql)
        connection.execute("PRAGMA user_version = 7")
        connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
            ("doc", "manual.txt", "version", 1, "now", "now"),
        )
        connection.execute(
            "INSERT INTO document_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "version",
                "doc",
                1,
                "parser",
                hashlib.sha256(b"x").hexdigest(),
                "active",
                "now",
                "now",
            ),
        )
        connection.commit()
    manifest = DocumentManifest(path)
    record = manifest.get_version("version")
    assert record is not None and record.metadata.applicability.scope_status == "unknown"
    assert manifest.active_version_ids() == frozenset({"version"})


def test_api_and_cli_metadata_round_trip(tmp_path: Path) -> None:
    settings = RuntimeSettings(runtime_dir=tmp_path, database_path=tmp_path / "runtime.db")
    application = bootstrap(settings)
    request = IngestDocumentRequest(
        display_name="manual.txt",
        content="EGT temperature",
        metadata=DocumentMetadata(source_revision="R3"),
    )
    response = ingest_document(request, application)
    catalog = list_documents(application)
    assert catalog[0].versions[0].version_id == response.version_id
    assert catalog[0].versions[0].metadata.source_revision == "R3"
    source = tmp_path / "new.txt"
    source.write_text("Compressor inspection", encoding="utf-8")
    result = ingest_file(source, settings=settings, metadata=DocumentMetadata(source_revision="R4"))
    records = bootstrap(settings).browse_knowledge.execute()
    assert any(
        version.version_id == result.version_id and version.metadata.source_revision == "R4"
        for document in records
        for version in document.versions
    )


def test_cli_rejects_invalid_metadata_before_bootstrap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "metadata.json"
    source.write_text('{"applicability":{"scope_status":"known"}}', encoding="utf-8")
    assert main(["unused.txt", "--metadata", str(source)]) == 1
    assert "known scope" in capsys.readouterr().err
