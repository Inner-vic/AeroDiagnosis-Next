"""Atomic-visibility orchestration for deterministic local document ingestion."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from aerodiagnosis.application.knowledge_graph_extractor import KnowledgeGraphExtractor
from aerodiagnosis.domain import (
    DocumentMetadata,
    SourceKind,
    create_document_id,
    make_chunk_id,
    make_evidence_id,
    make_version_id,
)
from aerodiagnosis.ports import GraphStore, VectorChunk, VectorStore
from aerodiagnosis.ports.context import ParentContext, ParentContextStore
from aerodiagnosis.ports.graph import GraphExtractor

from .manifest import DocumentManifest, VersionStatus
from .parsers import ParsedDocument, ParserRegistry
from .source_archive import SourceArchive


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
        graph_store: GraphStore | None = None,
        graph_extractor: GraphExtractor | None = None,
        parsers: ParserRegistry | None = None,
        source_archive: SourceArchive | None = None,
        *,
        max_bytes: int = 10 * 1024 * 1024,
        context_store: ParentContextStore | None = None,
        chinese_chunks: Callable[[Sequence[VectorChunk]], tuple[VectorChunk, ...]] | None = None,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self._manifest = manifest
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._graph_extractor = graph_extractor or KnowledgeGraphExtractor()
        self._parsers = parsers or ParserRegistry()
        self._source_archive = source_archive
        self._max_bytes = max_bytes
        self._contexts = context_store
        self._chinese_chunks = chinese_chunks

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
        metadata: DocumentMetadata,
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
                        **(
                            {
                                "parent_context_id": make_chunk_id(
                                    version_id,
                                    parsed_chunk.parent.locator.canonical(),
                                    parsed_chunk.parent.content,
                                )
                            }
                            if parsed_chunk.parent is not None
                            else {}
                        ),
                        "display_name": display_name,
                        "document_metadata": metadata.model_dump(mode="json"),
                        "parser_version": parsed.parser_version,
                        "media_type": parsed.media_type,
                        "locator": locator,
                        "source_boundary_warning": (
                            "Source excerpt ends mid-sentence; do not invent continuation"
                        )
                        if re.search(
                            r"(?:must be|or with|and a|remove the|on the)\s*$",
                            parsed_chunk.content,
                            re.I,
                        )
                        else "",
                        "content_hash": hashlib.sha256(
                            parsed_chunk.content.encode("utf-8")
                        ).hexdigest(),
                        "evidence_id": evidence_id,
                    },
                )
            )
            evidence_ids.append(evidence_id)
        return tuple(chunks), tuple(evidence_ids)

    @staticmethod
    def _parent_contexts(
        document_id: str, version_id: str, parsed: ParsedDocument
    ) -> tuple[ParentContext, ...]:
        contexts: dict[str, ParentContext] = {}
        for chunk in parsed.chunks:
            if chunk.parent is None:
                continue
            locator = chunk.parent.locator.canonical()
            key = make_chunk_id(version_id, locator, chunk.parent.content)
            contexts[key] = ParentContext(
                key, document_id, version_id, chunk.parent.content, locator
            )
        return tuple(contexts.values())

    def ingest(
        self,
        *,
        display_name: str,
        content: bytes,
        document_id: str | None = None,
        force_vector_upsert: bool = False,
        metadata: DocumentMetadata | None = None,
    ) -> IngestionResult:
        metadata = metadata or DocumentMetadata()
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
        if self._graph_store is not None:
            pipeline = f"{version_id}\0{self._graph_extractor.identity}"
            version_id = hashlib.sha256(pipeline.encode()).hexdigest()
        version_id = metadata.bind_version_id(version_id)
        chunks, evidence_ids = self._vector_chunks(
            logical_id, version_id, safe_name, parsed, metadata
        )
        self._manifest.register_document(logical_id, safe_name)
        existing = self._manifest.get_version(version_id)
        if existing is not None:
            if existing.status is VersionStatus.ACTIVE:
                if force_vector_upsert:
                    if self._contexts is not None:
                        contexts = self._parent_contexts(logical_id, version_id, parsed)
                        if self._contexts.upsert(contexts) != len(contexts):
                            raise IngestionError("parent context repair was incomplete")
                    getter = getattr(self._vector_store, "get_chunk", None)
                    repaired = []
                    for chunk in chunks:
                        old = getter(chunk.chunk_id) if callable(getter) else None
                        if old is not None and old.content == chunk.content:
                            preserved = {
                                key: value
                                for key, value in old.metadata.items()
                                if key.startswith("display_")
                            }
                            chunk = replace(chunk, metadata={**chunk.metadata, **preserved})
                        repaired.append(chunk)
                    if self._chinese_chunks is not None:
                        repaired = list(self._chinese_chunks(repaired))
                    if self._vector_store.upsert(repaired) != len(chunks):
                        raise IngestionError("vector index repair was incomplete")
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
            metadata=metadata,
        )
        try:
            if self._source_archive is not None:
                self._source_archive.store(
                    version_id,
                    safe_name,
                    content,
                    parsed.canonical_content,
                    parsed.source_artifacts,
                )
            self._manifest.transition(version_id, VersionStatus.PARSED)
            self._manifest.transition(version_id, VersionStatus.INDEXING)
            if self._contexts is not None:
                contexts = self._parent_contexts(logical_id, version_id, parsed)
                if self._contexts.upsert(contexts) != len(contexts):
                    raise IngestionError("parent context write was incomplete")
            if self._chinese_chunks is not None:
                chunks = self._chinese_chunks(chunks)
            indexed = self._vector_store.upsert(chunks)
            if indexed != len(chunks):
                raise IngestionError(
                    f"vector store indexed {indexed} of {len(chunks)} expected chunks"
                )
            if self._graph_store is not None:
                nodes, edges = self._graph_extractor.extract(
                    source_ref=f"document:{logical_id}",
                    version_id=version_id,
                    chunks=tuple(chunk.content for chunk in chunks),
                    chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
                    chunk_locators=tuple(
                        {
                            **dict(chunk.metadata["locator"]),
                            "coordinates": {
                                **dict(chunk.metadata["locator"]["coordinates"]),
                                "canonical_sha256": hashlib.sha256(
                                    parsed.canonical_content
                                ).hexdigest(),
                            },
                        }
                        if self._graph_extractor.identity.startswith("langextract@")
                        else dict(chunk.metadata["locator"])
                        for chunk in chunks
                    ),
                )
                context = metadata.model_dump(mode="json")
                nodes = [
                    replace(
                        node,
                        properties={
                            **node.properties,
                            "document_metadata": context,
                            "document_display_name": safe_name,
                        },
                    )
                    for node in nodes
                ]
                edges = [
                    replace(
                        edge,
                        properties={
                            **edge.properties,
                            "document_metadata": context,
                            "document_display_name": safe_name,
                        },
                    )
                    for edge in edges
                ]
                if self._graph_store.upsert_nodes(nodes) != len(nodes):
                    raise IngestionError("graph node write was incomplete")
                if self._graph_store.upsert_edges(edges) != len(edges):
                    raise IngestionError("graph edge write was incomplete")
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
