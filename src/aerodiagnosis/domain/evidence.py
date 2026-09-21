"""Stable, framework-free evidence identity and provenance types."""

from __future__ import annotations

import hashlib
import json
import unicodedata
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class SourceKind(StrEnum):
    """Kinds of immutable evidence understood by the diagnosis domain."""

    DOCUMENT_CHUNK = "document_chunk"
    KNOWLEDGE_GRAPH_PATH = "knowledge_graph_path"
    CASE = "case"
    PARAMETER_ANALYSIS = "parameter_analysis"
    EXTERNAL_TOOL = "external_tool"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_content(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def create_document_id() -> str:
    """Create an opaque logical document identity independent of path or content."""

    return str(uuid.uuid4())


def make_version_id(document_id: str, canonical_content: bytes, parser_version: str) -> str:
    """Create an immutable identity for one parsed version of a logical document."""

    content_hash = hashlib.sha256(canonical_content).hexdigest()
    return _sha256_text(f"{document_id}\0{content_hash}\0{parser_version}")


def make_chunk_id(version_id: str, locator: Mapping[str, Any], content: str) -> str:
    """Create a stable chunk identity from source version, location and normalized text."""

    content_hash = _sha256_text(_normalize_content(content))
    return _sha256_text(f"{version_id}\0{_canonical_json(locator)}\0{content_hash}")


def make_evidence_id(
    source_kind: SourceKind,
    immutable_source_ref: str,
    locator: Mapping[str, Any],
) -> str:
    """Create a stable evidence identity that cannot be forged from display text."""

    return _sha256_text(f"{source_kind.value}\0{immutable_source_ref}\0{_canonical_json(locator)}")


@dataclass(frozen=True, slots=True)
class EvidenceLocator:
    """Typed location inside an immutable source version."""

    kind: str
    coordinates: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("locator kind must not be empty")
        object.__setattr__(self, "coordinates", MappingProxyType(dict(self.coordinates)))

    def canonical(self) -> Mapping[str, Any]:
        return {"kind": self.kind, "coordinates": dict(self.coordinates)}


@dataclass(frozen=True, slots=True)
class Evidence:
    """An immutable piece of evidence returned by a domain tool."""

    evidence_id: str
    source_kind: SourceKind
    immutable_source_ref: str
    locator: EvidenceLocator
    content_hash: str
    upstream_ids: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = make_evidence_id(
            self.source_kind,
            self.immutable_source_ref,
            self.locator.canonical(),
        )
        if self.evidence_id != expected:
            raise ValueError("evidence_id does not match source provenance")
        if len(self.content_hash) != 64:
            raise ValueError("content_hash must be a SHA-256 hex digest")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
