from __future__ import annotations

import hashlib
import uuid

import pytest

from aerodiagnosis.domain.evidence import (
    Evidence,
    EvidenceLocator,
    SourceKind,
    create_document_id,
    make_chunk_id,
    make_evidence_id,
    make_version_id,
)


def test_document_identity_is_an_opaque_uuid4() -> None:
    document_id = create_document_id()

    assert uuid.UUID(document_id).version == 4


def test_version_identity_changes_with_content_or_parser() -> None:
    document_id = create_document_id()
    baseline = make_version_id(document_id, b"same bytes", "parser-1")

    assert baseline != make_version_id(document_id, b"other bytes", "parser-1")
    assert baseline != make_version_id(document_id, b"same bytes", "parser-2")


def test_chunk_identity_is_stable_for_mapping_order_and_whitespace() -> None:
    first = make_chunk_id("version", {"page": 2, "section": "HPC"}, "叶片   损伤")
    second = make_chunk_id("version", {"section": "HPC", "page": 2}, "叶片 损伤")

    assert first == second


def test_evidence_rejects_forged_identity() -> None:
    locator = EvidenceLocator(kind="page", coordinates={"page": 3})

    with pytest.raises(ValueError, match="does not match"):
        Evidence(
            evidence_id="0" * 64,
            source_kind=SourceKind.DOCUMENT_CHUNK,
            immutable_source_ref="chunk-1",
            locator=locator,
            content_hash=hashlib.sha256(b"content").hexdigest(),
        )


def test_evidence_accepts_matching_provenance() -> None:
    locator = EvidenceLocator(kind="page", coordinates={"page": 3})
    evidence_id = make_evidence_id(
        SourceKind.DOCUMENT_CHUNK,
        "chunk-1",
        locator.canonical(),
    )

    evidence = Evidence(
        evidence_id=evidence_id,
        source_kind=SourceKind.DOCUMENT_CHUNK,
        immutable_source_ref="chunk-1",
        locator=locator,
        content_hash=hashlib.sha256(b"content").hexdigest(),
    )

    assert evidence.evidence_id == evidence_id
