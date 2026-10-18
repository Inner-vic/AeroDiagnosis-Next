"""Deterministic, offline parsers for the first v3 ingestion slice."""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Protocol

from pypdf import PdfReader

from aerodiagnosis.domain import EvidenceLocator

from .structure import SourceSpan, algorithm_version, hierarchical_spans


class DocumentParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedContext:
    content: str
    locator: EvidenceLocator


@dataclass(frozen=True, slots=True)
class ParsedChunk:
    content: str
    locator: EvidenceLocator
    parent: ParsedContext | None = None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    parser_version: str
    media_type: str
    canonical_content: bytes
    chunks: tuple[ParsedChunk, ...]
    source_artifacts: dict[str, bytes] = field(default_factory=dict, kw_only=True)


class DocumentParser(Protocol):
    @property
    def parser_version(self) -> str: ...

    def parse(self, content: bytes) -> ParsedDocument: ...


def _decode_text(content: bytes) -> str:
    if not content:
        raise DocumentParseError("document is empty")
    encodings = (
        ("utf-8-sig", "utf-16")
        if content.startswith((b"\xff\xfe", b"\xfe\xff"))
        else ("utf-8-sig",)
    )
    for encoding in encodings:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentParseError("document must be valid UTF-8 or BOM-marked UTF-16")


def _canonical_text(content: bytes) -> str:
    text = _decode_text(content).replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise DocumentParseError("document contains no text")
    return text


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _text_spans(
    text: str, *, chunk_size: int, overlap: int, strategy: str
) -> list[tuple[int, int, tuple[str, ...], bool]]:
    """Return exact source spans; headings are locator context, never synthetic text."""
    result: list[tuple[int, int, tuple[str, ...], bool]] = []
    if strategy == "fixed":
        for start in range(0, len(text), chunk_size - overlap):
            end = min(start + chunk_size, len(text))
            left, right = _trim_span(text, start, end)
            if left < right:
                result.append((left, right, (), False))
            if end == len(text):
                break
        return result

    # Blank lines and Markdown headings are safe structural boundaries. Lists and
    # tables without blank lines stay with their surrounding paragraph.
    starts = {0, len(text)}
    for match in re.finditer(r"\n[ \t]*\n+|(?m:^#{1,6}[ \t]+[^\n]+$)", text):
        starts.add(match.start())
        if text[match.start() : match.end()].lstrip().startswith("#"):
            starts.add(match.end())
    boundaries = sorted(starts)
    units = []
    headings: list[str] = []
    for left, right in pairwise(boundaries):
        left, right = _trim_span(text, left, right)
        if left == right:
            continue
        heading = re.fullmatch(r"(#{1,6})[ \t]+(.+)", text[left:right])
        if heading:
            level = len(heading[1])
            headings = [*headings[: level - 1], heading[2].strip()]
        units.append((left, right, tuple(headings), heading is not None))

    result = []
    pending: tuple[int, int, tuple[str, ...], bool] | None = None
    for left, right, path, is_heading in units:
        if pending and (is_heading or path != pending[2] or right - pending[0] > chunk_size):
            result.append(pending)
            pending = None
        if right - left > chunk_size:
            if pending:
                result.append(pending)
                pending = None
            cursor = left
            while cursor < right:
                cut = min(cursor + chunk_size, right)
                if cut < right:
                    matches = list(re.finditer(r"[。！？.!?]\s+|\n", text[cursor:cut]))  # noqa: RUF001
                    safe = [
                        cursor + item.end() for item in matches if item.end() >= chunk_size // 2
                    ]
                    if safe:
                        cut = safe[-1]
                a, b = _trim_span(text, cursor, cut)
                if a < b:
                    result.append((a, b, path, True))
                if cut == right:
                    break
                cursor = max(cursor + 1, cut - overlap)
        elif pending:
            pending = (pending[0], right, path, False)
        else:
            pending = (left, right, path, False)
    if pending:
        result.append(pending)
    return result


def _validate_strategy(
    strategy: str, chunk_size: int, overlap: int, parent_budget: int, child_budget: int
) -> None:
    if strategy not in {"fixed", "structure", "hierarchical"}:
        raise ValueError("strategy must be fixed, structure or hierarchical")
    if chunk_size < 100 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be at least 100 and overlap must be smaller")
    if child_budget < 100 or parent_budget < child_budget:
        raise ValueError(
            "child_budget must be at least 100 and parent_budget at least child_budget"
        )


def _hierarchical_chunks(
    text: str,
    *,
    parent_budget: int,
    child_budget: int,
    page_number: int | None = None,
    canonical_offset: int = 0,
) -> tuple[ParsedChunk, ...]:
    contexts: dict[SourceSpan, ParsedContext] = {}

    def locator(span: SourceSpan, budget: int, level: str) -> EvidenceLocator:
        coordinates: dict[str, object] = {
            "start": span.start,
            "end": span.end,
            "section_path": list(span.section_path),
            "unit_type": span.unit_type,
            "oversized": span.end - span.start > budget,
            "hierarchy_level": level,
            "fallback_window": False,
        }
        if page_number is not None:
            coordinates.update(
                page=page_number,
                canonical_start=canonical_offset + span.start,
                canonical_end=canonical_offset + span.end,
            )
        return EvidenceLocator(
            kind="character_range" if page_number is None else "pdf_page",
            coordinates=coordinates,
        )

    chunks: list[ParsedChunk] = []
    for child, parent in hierarchical_spans(
        text, parent_budget=parent_budget, child_budget=child_budget
    ):
        if parent not in contexts:
            contexts[parent] = ParsedContext(
                content=text[parent.start : parent.end],
                locator=locator(parent, parent_budget, "parent"),
            )
        chunks.append(
            ParsedChunk(
                content=text[child.start : child.end],
                locator=locator(child, child_budget, "child"),
                parent=contexts[parent],
            )
        )
    return tuple(chunks)


class PlainTextParser:
    def __init__(
        self,
        *,
        chunk_size: int = 800,
        overlap: int = 100,
        strategy: str = "fixed",
        parent_budget: int = 3200,
        child_budget: int = 600,
    ) -> None:
        _validate_strategy(strategy, chunk_size, overlap, parent_budget, child_budget)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._strategy = strategy
        self._parent_budget = parent_budget
        self._child_budget = child_budget
        self.parser_version = (
            f"plain-text@3:hierarchical:{parent_budget}:{child_budget}:{algorithm_version()}"
            if strategy == "hierarchical"
            else f"plain-text@2:{strategy}:{chunk_size}:{overlap}"
        )

    def parse(self, content: bytes) -> ParsedDocument:
        text = _canonical_text(content)
        if self._strategy == "hierarchical":
            chunks = _hierarchical_chunks(
                text, parent_budget=self._parent_budget, child_budget=self._child_budget
            )
        else:
            chunks = tuple(
                ParsedChunk(
                    content=text[start:end],
                    locator=EvidenceLocator(
                        kind="character_range",
                        coordinates={
                            "start": start,
                            "end": end,
                            "section_path": list(path),
                            "fallback_window": fallback,
                        },
                    ),
                )
                for start, end, path, fallback in _text_spans(
                    text,
                    chunk_size=self._chunk_size,
                    overlap=self._overlap,
                    strategy=self._strategy,
                )
            )
        return ParsedDocument(
            parser_version=self.parser_version,
            media_type="text/plain",
            canonical_content=text.encode("utf-8"),
            chunks=chunks,
        )


class CsvParser:
    parser_version = "csv@1"

    def __init__(self, *, rows_per_chunk: int = 20) -> None:
        if rows_per_chunk < 1:
            raise ValueError("rows_per_chunk must be positive")
        self._rows_per_chunk = rows_per_chunk

    def parse(self, content: bytes) -> ParsedDocument:
        text = _canonical_text(content)
        try:
            rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
        except csv.Error as exc:
            raise DocumentParseError("invalid CSV content") from exc
        if not rows or not any(cell.strip() for row in rows for cell in row):
            raise DocumentParseError("CSV contains no cells")
        header = rows[0]
        data_rows = rows[1:]
        chunks = []
        if not data_rows:
            chunks.append(
                ParsedChunk(
                    content="Columns: " + " | ".join(header),
                    locator=EvidenceLocator(
                        kind="csv_rows",
                        coordinates={"row_start": 1, "row_end": 1},
                    ),
                )
            )
        for offset in range(0, len(data_rows), self._rows_per_chunk):
            batch = data_rows[offset : offset + self._rows_per_chunk]
            lines = ["Columns: " + " | ".join(header)]
            for index, row in enumerate(batch, start=offset + 2):
                fields = []
                for column, value in enumerate(row):
                    label = (
                        header[column]
                        if column < len(header) and header[column]
                        else f"column_{column + 1}"
                    )
                    fields.append(f"{label}: {value}")
                lines.append(f"Row {index}: " + " | ".join(fields))
            chunks.append(
                ParsedChunk(
                    content="\n".join(lines),
                    locator=EvidenceLocator(
                        kind="csv_rows",
                        coordinates={
                            "row_start": offset + 2,
                            "row_end": offset + len(batch) + 1,
                        },
                    ),
                )
            )
        return ParsedDocument(
            parser_version=self.parser_version,
            media_type="text/csv",
            canonical_content=text.encode("utf-8"),
            chunks=tuple(chunks),
        )


class PdfParser:
    def __init__(
        self,
        *,
        chunk_size: int = 800,
        overlap: int = 100,
        strategy: str = "structure",
        parent_budget: int = 3200,
        child_budget: int = 600,
    ) -> None:
        _validate_strategy(strategy, chunk_size, overlap, parent_budget, child_budget)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._strategy = strategy
        self._parent_budget = parent_budget
        self._child_budget = child_budget
        self.parser_version = (
            f"pdf@3:page-hierarchical:{parent_budget}:{child_budget}:{algorithm_version()}"
            if strategy == "hierarchical"
            else f"pdf@2:page-{strategy}:{chunk_size}:{overlap}"
        )

    def parse(self, content: bytes) -> ParsedDocument:
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as exc:
            raise DocumentParseError("invalid PDF content") from exc
        chunks: list[ParsedChunk] = []
        texts: list[str] = []
        offset = 0
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\r\n", "\n").replace("\r", "\n")
            if not text.strip():
                continue
            if texts:
                offset += 2
            texts.append(text)
            if self._strategy == "hierarchical":
                chunks.extend(
                    _hierarchical_chunks(
                        text,
                        parent_budget=self._parent_budget,
                        child_budget=self._child_budget,
                        page_number=page_number,
                        canonical_offset=offset,
                    )
                )
            else:
                for start, end, path, fallback in _text_spans(
                    text,
                    chunk_size=self._chunk_size,
                    overlap=self._overlap,
                    strategy=self._strategy,
                ):
                    chunks.append(
                        ParsedChunk(
                            content=text[start:end],
                            locator=EvidenceLocator(
                                kind="pdf_page",
                                coordinates={
                                    "page": page_number,
                                    "start": start,
                                    "end": end,
                                    "canonical_start": offset + start,
                                    "canonical_end": offset + end,
                                    "section_path": list(path),
                                    "fallback_window": fallback,
                                },
                            ),
                        )
                    )
            offset += len(text)
        if not chunks:
            raise DocumentParseError("PDF contains no extractable text")
        return ParsedDocument(
            parser_version=self.parser_version,
            media_type="application/pdf",
            canonical_content="\n\n".join(texts).encode("utf-8"),
            chunks=tuple(chunks),
        )


class DocxParser:
    parser_version = "docx@1"

    def parse(self, content: bytes) -> ParsedDocument:
        from docx import Document as DocxDocument

        try:
            document = DocxDocument(io.BytesIO(content))
        except Exception as exc:
            raise DocumentParseError("invalid DOCX content") from exc
        parts: list[str] = [
            paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()
        ]
        for table in document.tables:
            for row in table.rows:
                values = [cell.text.strip() for cell in row.cells]
                if any(values):
                    parts.append("\t".join(values))
        canonical = "\n".join(parts)
        if not canonical.strip():
            raise DocumentParseError("DOCX contains no text")
        return ParsedDocument(
            parser_version=self.parser_version,
            media_type=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            canonical_content=canonical.encode("utf-8"),
            chunks=(
                ParsedChunk(
                    content=canonical,
                    locator=EvidenceLocator(
                        kind="docx_document",
                        coordinates={"parts": len(parts)},
                    ),
                ),
            ),
        )


class XlsxParser:
    parser_version = "xlsx@1"

    def parse(self, content: bytes) -> ParsedDocument:
        from openpyxl import load_workbook

        try:
            workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        except Exception as exc:
            raise DocumentParseError("invalid XLSX content") from exc
        chunks: list[ParsedChunk] = []
        sheet_texts: list[str] = []
        for worksheet in workbook.worksheets:
            lines: list[str] = []
            for row in worksheet.iter_rows(values_only=True):
                if any(value is not None and str(value).strip() for value in row):
                    lines.append(" | ".join("" if value is None else str(value) for value in row))
            text = "\n".join(lines)
            if text.strip():
                chunks.append(
                    ParsedChunk(
                        content=text,
                        locator=EvidenceLocator(
                            kind="xlsx_sheet",
                            coordinates={
                                "sheet": worksheet.title,
                                "rows": len(lines),
                            },
                        ),
                    )
                )
                sheet_texts.append(text)
        if not chunks:
            raise DocumentParseError("XLSX contains no rows")
        return ParsedDocument(
            parser_version=self.parser_version,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            canonical_content="\n\n".join(sheet_texts).encode("utf-8"),
            chunks=tuple(chunks),
        )


class HtmlParser:
    parser_version = "html@1"

    def parse(self, content: bytes) -> ParsedDocument:
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(content, "lxml")
        except Exception as exc:
            raise DocumentParseError("invalid HTML content") from exc
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        if not text:
            raise DocumentParseError("HTML contains no text")
        return ParsedDocument(
            parser_version=self.parser_version,
            media_type="text/html",
            canonical_content=text.encode("utf-8"),
            chunks=(
                ParsedChunk(
                    content=text,
                    locator=EvidenceLocator(
                        kind="html_document",
                        coordinates={"lines": len(text.splitlines())},
                    ),
                ),
            ),
        )


class ParserRegistry:
    def __init__(
        self,
        parsers: Mapping[str, DocumentParser] | None = None,
        *,
        text_strategy: str = "hierarchical",
        parent_budget: int = 3200,
        child_budget: int = 600,
        upstream_backend: str = "native",
        tokenizer_path: Path | None = None,
    ) -> None:
        text_parser = PlainTextParser(
            strategy=text_strategy, parent_budget=parent_budget, child_budget=child_budget
        )
        defaults: dict[str, DocumentParser] = {
            ".txt": text_parser,
            ".md": text_parser,
            ".markdown": text_parser,
            ".csv": CsvParser(),
            ".pdf": PdfParser(
                strategy=text_strategy, parent_budget=parent_budget, child_budget=child_budget
            ),
            ".docx": DocxParser(),
            ".xlsx": XlsxParser(),
            ".html": HtmlParser(),
            ".htm": HtmlParser(),
        }
        if upstream_backend == "docling":
            from aerodiagnosis.adapters.docling_parser import DoclingParser

            if tokenizer_path is None:
                raise DocumentParseError("Docling needs an explicit local tokenizer path")
            for extension in (".pdf", ".html", ".htm", ".docx", ".xlsx", ".csv"):
                defaults[extension] = DoclingParser(extension, tokenizer_path=tokenizer_path)
        elif upstream_backend != "native":
            raise DocumentParseError("Unknown parser backend")
        self._parsers = {key.lower(): value for key, value in (parsers or defaults).items()}

    @property
    def supported_extensions(self) -> Sequence[str]:
        return tuple(sorted(self._parsers))

    def parse(self, display_name: str, content: bytes) -> ParsedDocument:
        suffix = Path(display_name).suffix.lower()
        try:
            parser = self._parsers[suffix]
        except KeyError as exc:
            supported = ", ".join(self.supported_extensions)
            raise DocumentParseError(
                f"unsupported document extension {suffix or '<none>'}; supported: {supported}"
            ) from exc
        return parser.parse(content)
