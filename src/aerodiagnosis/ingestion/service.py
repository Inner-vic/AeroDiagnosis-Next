"""Atomic-visibility orchestration for deterministic local document ingestion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from aerodiagnosis.domain import (
    SourceKind,
    create_document_id,
    make_chunk_id,
    make_evidence_id,
    make_version_id,
)
from aerodiagnosis.ports import VectorChunk, VectorStore

from .manifest import DocumentManifest, VersionStatus
from .parsers import ParsedDocument, ParserRegistry


class IngestionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: str
    version_id: str
    revision: int
    status: VersionStatus
    chunk_count: int
    evidence_ids: tuple[str, ...]
    reused: bool = False


class DocumentIngestionService:
    def __init__(
        self,
        manifest: DocumentManifest,
        vector_store: VectorStore,
        parsers: ParserRegistry | None = None,
        *,
        max_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self._manifest = manifest
        self._vector_store = vector_store
        self._parsers = parsers or ParserRegistry()
        self._max_bytes = max_bytes

    @staticmethod
    def _validate_display_name(display_name: str) -> str:
        cleaned = display_name.strip()
        if not cleaned or cleaned in {".", ".."} or any(char in cleaned for char in ("/", "\\")):
            raise IngestionError("display_name must be a single safe file name")
        return cleaned

    @staticmethod
    def _vector_chunks(
        document_id: str,
        version_id: str,
        display_name: str,
        parsed: ParsedDocument,
    ) -> tuple[tuple[VectorChunk, ...], tuple[str, ...]]:
        chunks = []
        evidence_ids = []
        for parsed_chunk in parsed.chunks:
            locator = parsed_chunk.locator.canonical()
            chunk_id = make_chunk_id(version_id, locator, parsed_chunk.content)
            evidence_id = make_evidence_id(SourceKind.DOCUMENT_CHUNK, chunk_id, locator)
            chunks.append(
                VectorChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    version_id=version_id,
                    content=parsed_chunk.content,
                    metadata={
                        "display_name": display_name,
                        "parser_version": parsed.parser_version,
                        "media_type": parsed.media_type,
                        "locator": locator,
                        "content_hash": hashlib.sha256(
                            parsed_chunk.content.encode("utf-8")
                        ).hexdigest(),
                        "evidence_id": evidence_id,
                    },
                )
            )
            evidence_ids.append(evidence_id)
        return tuple(chunks), tuple(evidence_ids)

    def ingest(
        self,
        *,
        display_name: str,
        content: bytes,
        document_id: str | None = None,
        force_vector_upsert: bool = False,
    ) -> IngestionResult:
        safe_name = self._validate_display_name(display_name)
        if len(content) > self._max_bytes:
            raise IngestionError(f"document exceeds the {self._max_bytes}-byte local limit")
        parsed = self._parsers.parse(safe_name, content)
        if not parsed.chunks:
            raise IngestionError("parser produced no indexable chunks")
        logical_id = document_id or create_document_id()
        version_id = make_version_id(
            logical_id,
            parsed.canonical_content,
            parsed.parser_version,
        )
        chunks, evidence_ids = self._vector_chunks(logical_id, version_id, safe_name, parsed)
        self._manifest.register_document(logical_id, safe_name)
        existing = self._manifest.get_version(version_id)
        if existing is not None:
            if existing.status is VersionStatus.ACTIVE:
                if force_vector_upsert:
                    self._vector_store.upsert(chunks)
                return IngestionResult(
                    document_id=logical_id,
                    version_id=version_id,
                    revision=existing.revision,
                    status=existing.status,
                    chunk_count=len(chunks),
                    evidence_ids=evidence_ids,
                    reused=True,
                )
            raise IngestionError(
                f"identical version already exists in non-active state: {existing.status}"
            )

        content_hash = hashlib.sha256(parsed.canonical_content).hexdigest()
        record = self._manifest.create_version(
            document_id=logical_id,
            version_id=version_id,
            parser_version=parsed.parser_version,
            content_hash=content_hash,
        )
        try:
            self._manifest.transition(version_id, VersionStatus.PARSED)
            self._manifest.transition(version_id, VersionStatus.INDEXING)
            indexed = self._vector_store.upsert(chunks)
            if indexed != len(chunks):
                raise IngestionError(
                    f"vector store indexed {indexed} of {len(chunks)} expected chunks"
                )
            self._manifest.transition(version_id, VersionStatus.STAGED)
            active = self._manifest.activate(version_id)
        except Exception:
            current = self._manifest.get_version(version_id)
            if current is not None and current.status in {
                VersionStatus.RECEIVED,
                VersionStatus.PARSED,
                VersionStatus.INDEXING,
                VersionStatus.STAGED,
            }:
                self._manifest.transition(version_id, VersionStatus.FAILED)
            raise
        return IngestionResult(
            document_id=logical_id,
            version_id=version_id,
            revision=record.revision,
            status=active.status,
            chunk_count=len(chunks),
            evidence_ids=evidence_ids,
        )
