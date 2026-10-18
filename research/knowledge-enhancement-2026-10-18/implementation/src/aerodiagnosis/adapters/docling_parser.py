"""Docling SDK adapter preserving structured source provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
from typing import Any

from aerodiagnosis.adapters.docling_structure import structured_records
from aerodiagnosis.domain.evidence import EvidenceLocator
from aerodiagnosis.ingestion.parsers import (
    DocumentParseError,
    ParsedChunk,
    ParsedContext,
    ParsedDocument,
)


class DoclingParser:
    def __init__(self, extension: str, *, tokenizer_path: Path, token_budget: int = 400) -> None:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise DocumentParseError("Install aerodiagnosis[upstream] for Docling parsing") from exc
        self.extension = extension
        formats = {
            ".pdf": InputFormat.PDF,
            ".html": InputFormat.HTML,
            ".htm": InputFormat.HTML,
            ".docx": InputFormat.DOCX,
            ".xlsx": InputFormat.XLSX,
            ".csv": InputFormat.CSV,
        }
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        self._tokenizer = tokenizer
        self._token_budget = token_budget
        self._chunker: Any = HybridChunker(
            tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=token_budget),
            merge_peers=True,
        )
        self._converter: Any = DocumentConverter(
            allowed_formats=[formats[extension]],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=PdfPipelineOptions(do_ocr=False, do_table_structure=True)
                )
            },
        )
        tokenizer_digest = hashlib.sha256(
            (tokenizer_path / "tokenizer.json").read_bytes()
        ).hexdigest()
        self.parser_version = (
            f"docling@{importlib.metadata.version('docling-slim')}"
            f":core={importlib.metadata.version('docling-core')}:hybrid={token_budget}"
            f":tokenizer={tokenizer_digest}:source-rows@5"
        )

    def parse(self, content: bytes) -> ParsedDocument:
        from docling.datamodel.base_models import DocumentStream

        if not content:
            raise DocumentParseError("Document is empty")
        try:
            result = self._converter.convert(
                DocumentStream(name="source" + self.extension, stream=io.BytesIO(content))
            )
            if result.status.value != "success":
                raise DocumentParseError("Docling conversion did not complete successfully")
            return self.parse_structure(content, result.document.export_to_dict())
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError("Docling conversion failed: " + type(exc).__name__) from None

    def parse_structure(self, content: bytes, structure: dict[str, Any]) -> ParsedDocument:
        records = structured_records(structure, self._tokenizer, self._token_budget)
        if not records:
            raise DocumentParseError("Docling produced no indexable content")
        canonical = "\n\n".join(text for text, _ in records)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        raw_digest = hashlib.sha256(content).hexdigest()
        offset = 0
        chunks = []
        for text, metadata in records:
            locator = EvidenceLocator(
                kind="docling_chunk",
                coordinates={
                    **metadata,
                    "start": offset,
                    "end": offset + len(text),
                    "canonical_sha256": digest,
                    "original_sha256": raw_digest,
                },
            )
            chunks.append(
                ParsedChunk(
                    content=text,
                    locator=locator,
                    parent=ParsedContext(content=text, locator=locator),
                )
            )
            offset += len(text) + 2
        return ParsedDocument(
            parser_version=self.parser_version + ":raw=" + raw_digest,
            media_type=structure["origin"]["mimetype"],
            canonical_content=canonical.encode(),
            chunks=tuple(chunks),
            source_artifacts={"docling.json": json.dumps(structure, ensure_ascii=False).encode()},
        )
