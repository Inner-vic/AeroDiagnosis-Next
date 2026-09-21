from __future__ import annotations

import pytest

from aerodiagnosis.ingestion import CsvParser, DocumentParseError, ParserRegistry, PlainTextParser


def test_plain_text_parser_is_deterministic_and_normalizes_newlines() -> None:
    parser = PlainTextParser(chunk_size=100, overlap=10)
    windows = parser.parse(("compressor\r\nstall " * 20).encode())
    repeated = parser.parse(("compressor\nstall " * 20).encode())

    assert windows.canonical_content == repeated.canonical_content
    assert windows.chunks == repeated.chunks
    assert len(windows.chunks) > 1
    assert windows.chunks[0].locator.kind == "character_range"


def test_plain_text_accepts_bom_marked_utf16_and_rejects_non_text() -> None:
    parsed = PlainTextParser().parse("涡轮温度异常".encode("utf-16"))

    assert parsed.canonical_content.decode() == "涡轮温度异常"
    with pytest.raises(DocumentParseError, match="valid UTF-8"):
        PlainTextParser().parse(b"\x80\x81")
    with pytest.raises(DocumentParseError, match="empty"):
        PlainTextParser().parse(b"")
    with pytest.raises(DocumentParseError, match="no text"):
        PlainTextParser().parse(b" \r\n ")


def test_csv_parser_preserves_row_locators_and_batches() -> None:
    parsed = CsvParser(rows_per_chunk=2).parse(b"parameter,value\nn1,92\negt,710\np3,130\n")

    assert len(parsed.chunks) == 2
    assert parsed.chunks[0].locator.coordinates == {"row_start": 2, "row_end": 3}
    assert "parameter: egt" in parsed.chunks[0].content
    assert parsed.chunks[1].locator.coordinates == {"row_start": 4, "row_end": 4}


def test_csv_header_only_and_invalid_csv_have_explicit_results() -> None:
    parsed = CsvParser().parse(b"parameter,value\n")

    assert parsed.chunks[0].content == "Columns: parameter | value"
    with pytest.raises(DocumentParseError, match="invalid CSV"):
        CsvParser().parse(b'parameter,value\n"unclosed')


def test_parser_registry_only_advertises_verified_formats() -> None:
    registry = ParserRegistry()

    assert registry.supported_extensions == (
        ".csv",
        ".docx",
        ".htm",
        ".html",
        ".markdown",
        ".md",
        ".pdf",
        ".txt",
        ".xlsx",
    )
    assert registry.parse("notes.MD", b"fault isolation").media_type == "text/plain"
    with pytest.raises(DocumentParseError, match="unsupported document extension"):
        registry.parse("manual.xyz", b"not a supported document")


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: PlainTextParser(chunk_size=99), "chunk_size"),
        (lambda: PlainTextParser(chunk_size=100, overlap=100), "chunk_size"),
        (lambda: CsvParser(rows_per_chunk=0), "positive"),
    ],
)
def test_parser_configuration_rejects_invalid_limits(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]
