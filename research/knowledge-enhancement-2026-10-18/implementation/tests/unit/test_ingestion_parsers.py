# ruff: noqa: RUF001
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


def test_structure_preserves_conditions_headings_and_exact_spans() -> None:
    text = (
        "  # Engine A\n\n在巡航阶段，排气温度升高可能由污染引起。\n\n"
        "建议检查：\n1. 先核对工况。\n2. 再观察趋势。\n\n# Engine B\n\n另一型号的限制。  "
    )
    parser = PlainTextParser(strategy="structure", chunk_size=100, overlap=10)
    parsed = parser.parse(text.encode())
    canonical = parsed.canonical_content.decode()
    assert len(parsed.chunks) >= 2
    assert all(
        chunk.content
        == canonical[chunk.locator.coordinates["start"] : chunk.locator.coordinates["end"]]
        for chunk in parsed.chunks
    )
    assert any("巡航" in chunk.content and "污染" in chunk.content for chunk in parsed.chunks)
    assert any("1. 先" in chunk.content and "2. 再" in chunk.content for chunk in parsed.chunks)
    assert not any(
        "Engine A" in chunk.content and "Engine B" in chunk.content for chunk in parsed.chunks
    )
    assert parsed.chunks[-1].locator.coordinates["section_path"] == ["Engine B"]


def test_structure_keeps_markdown_table_header_units_and_footnotes() -> None:
    text = (
        "# Limits\n\n| parameter | limit (K) |\n| --- | --- |\n"
        "| EGT | 900 |\n* Applies at cruise only.\n"
    )
    parsed = PlainTextParser(strategy="structure").parse(text.encode())
    assert any(
        "limit (K)" in chunk.content and "EGT" in chunk.content and "cruise" in chunk.content
        for chunk in parsed.chunks
    )


def test_long_structure_fallback_is_bounded_and_covers_every_non_whitespace_character() -> None:
    text = "# Long\n\n" + ("在巡航阶段观察排气温度升高。 " * 50)
    parsed = PlainTextParser(strategy="structure", chunk_size=100, overlap=10).parse(text.encode())
    covered: set[int] = set()
    for chunk in parsed.chunks:
        coords = chunk.locator.coordinates
        assert len(chunk.content) <= 100
        covered.update(range(coords["start"], coords["end"]))
    assert all(index in covered for index, value in enumerate(text) if not value.isspace())
    assert any(chunk.locator.coordinates["fallback_window"] for chunk in parsed.chunks)


def test_parser_version_encodes_every_chunking_setting() -> None:
    variants = [
        PlainTextParser(strategy=strategy, chunk_size=size, overlap=overlap).parser_version
        for strategy in ("fixed", "structure")
        for size in (100, 200)
        for overlap in (0, 10)
    ]
    assert len(set(variants)) == 8
    assert "hierarchical" in ParserRegistry().parse("manual.md", b"content").parser_version
    with pytest.raises(ValueError, match="strategy"):
        PlainTextParser(strategy="unknown")
