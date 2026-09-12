from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aerodiagnosis.ingestion import DocumentManifest, VersionStatus


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _stage(manifest: DocumentManifest, version_id: str) -> None:
    manifest.transition(version_id, VersionStatus.PARSED)
    manifest.transition(version_id, VersionStatus.INDEXING)
    manifest.transition(version_id, VersionStatus.STAGED)


def test_manifest_assigns_revisions_and_activates_latest_candidate(tmp_path: Path) -> None:
    manifest = DocumentManifest(tmp_path / "runtime.db")
    manifest.register_document("document-1", "manual.pdf")
    first = manifest.create_version(
        document_id="document-1",
        version_id="version-1",
        parser_version="parser-1",
        content_hash=_hash("first"),
    )
    _stage(manifest, first.version_id)
    active = manifest.activate(first.version_id)

    assert first.revision == 1
    assert active.status is VersionStatus.ACTIVE
    assert manifest.active_version_ids() == frozenset({"version-1"})
    assert manifest.counts() == (1, 1)
    catalog = manifest.list_documents()
    assert catalog[0].display_name == "manual.pdf"
    assert catalog[0].active_version_id == "version-1"
    assert catalog[0].versions[0].revision == 1


def test_manifest_rejects_stale_publish_and_invalid_transition(tmp_path: Path) -> None:
    manifest = DocumentManifest(tmp_path / "runtime.db")
    manifest.register_document("document-1", "manual.pdf")
    first = manifest.create_version(
        document_id="document-1",
        version_id="version-1",
        parser_version="parser-1",
        content_hash=_hash("first"),
    )
    _stage(manifest, first.version_id)
    second = manifest.create_version(
        document_id="document-1",
        version_id="version-2",
        parser_version="parser-1",
        content_hash=_hash("second"),
    )

    assert second.revision == 2
    with pytest.raises(ValueError, match="stale candidate"):
        manifest.activate(first.version_id)
    with pytest.raises(ValueError, match="invalid version transition"):
        manifest.transition(second.version_id, VersionStatus.ACTIVE)


def test_generic_transition_cannot_bypass_guarded_activation(tmp_path: Path) -> None:
    manifest = DocumentManifest(tmp_path / "runtime.db")
    manifest.register_document("document-1", "manual.pdf")
    version = manifest.create_version(
        document_id="document-1",
        version_id="version-1",
        parser_version="parser-1",
        content_hash=_hash("content"),
    )
    _stage(manifest, version.version_id)

    with pytest.raises(ValueError, match="invalid version transition"):
        manifest.transition(version.version_id, VersionStatus.ACTIVE)

    assert manifest.active_version_ids() == frozenset()


def test_manifest_requires_registered_document_and_content_hash(tmp_path: Path) -> None:
    manifest = DocumentManifest(tmp_path / "runtime.db")

    with pytest.raises(KeyError, match="unknown document"):
        manifest.create_version(
            document_id="missing",
            version_id="version",
            parser_version="parser",
            content_hash=_hash("content"),
        )
    manifest.register_document("document", "manual")
    with pytest.raises(ValueError, match="SHA-256"):
        manifest.create_version(
            document_id="document",
            version_id="version",
            parser_version="parser",
            content_hash="short",
        )
