"""Deterministic, offline parsers for the first v3 ingestion slice."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aerodiagnosis.domain import EvidenceLocator


class DocumentParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedChunk:
    content: str
    locator: EvidenceLocator


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    parser_version: str
    media_type: str
    canonical_content: bytes
    chunks: tuple[ParsedChunk, ...]


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


class PlainTextParser:
    parser_version = "plain-text@1"

    def __init__(self, *, chunk_size: int = 800, overlap: int = 100) -> None:
        if chunk_size < 100 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be at least 100 and overlap must be smaller")
        self._chunk_size = chunk_size
        self._step = chunk_size - overlap

    def parse(self, content: bytes) -> ParsedDocument:
        text = _canonical_text(content)
        chunks = []
        for start in range(0, len(text), self._step):
            end = min(start + self._chunk_size, len(text))
            value = text[start:end].strip()
            if value:
                chunks.append(
                    ParsedChunk(
                        content=value,
                        locator=EvidenceLocator(
                            kind="character_range",
                            coordinates={"start": start, "end": end},
                        ),
                    )
                )
            if end == len(text):
                break
        return ParsedDocument(
            parser_version=self.parser_version,
            media_type="text/plain",
            canonical_content=text.encode("utf-8"),
            chunks=tuple(chunks),
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


class ParserRegistry:
    def __init__(self, parsers: Mapping[str, DocumentParser] | None = None) -> None:
        defaults: dict[str, DocumentParser] = {
            ".txt": PlainTextParser(),
            ".md": PlainTextParser(),
            ".markdown": PlainTextParser(),
            ".csv": CsvParser(),
        }
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
