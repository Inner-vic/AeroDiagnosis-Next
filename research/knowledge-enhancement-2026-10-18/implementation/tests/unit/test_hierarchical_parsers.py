# ruff: noqa: RUF001
from __future__ import annotations

from dataclasses import fields

import pytest

from aerodiagnosis.domain import EvidenceLocator
from aerodiagnosis.ingestion.parsers import (
    CsvParser,
    ParsedChunk,
    ParsedContext,
    ParsedDocument,
    ParserRegistry,
    PdfParser,
    PlainTextParser,
)


def _parser(parent_budget: int = 3200, child_budget: int = 600) -> PlainTextParser:
    return PlainTextParser(
        strategy="hierarchical", parent_budget=parent_budget, child_budget=child_budget
    )


def _assert_exact_hierarchy(parsed: ParsedDocument) -> None:
    canonical = parsed.canonical_content.decode()
    previous_end = -1
    coverage: set[int] = set()
    for chunk in parsed.chunks:
        coords = chunk.locator.coordinates
        assert not coords["fallback_window"]
        left = coords.get("canonical_start", coords["start"])
        right = coords.get("canonical_end", coords["end"])
        assert chunk.content == canonical[left:right]
        assert left >= previous_end
        previous_end = right
        coverage.update(range(left, right))
        assert chunk.parent is not None
        parent = chunk.parent
        parent_coords = parent.locator.coordinates
        parent_left = parent_coords.get("canonical_start", parent_coords["start"])
        parent_right = parent_coords.get("canonical_end", parent_coords["end"])
        assert parent.content == canonical[parent_left:parent_right]
        assert parent_left <= left < right <= parent_right
        assert parent_coords["hierarchy_level"] == "parent"
        assert coords["hierarchy_level"] == "child"
        assert parent_coords["section_path"] == coords["section_path"]
    assert all(index in coverage for index, char in enumerate(canonical) if not char.isspace())


def test_existing_positional_contracts_are_unchanged() -> None:
    locator = EvidenceLocator(kind="character_range", coordinates={"start": 0, "end": 4})
    child = ParsedChunk("text", locator)
    context = ParsedContext("text", locator)
    assert child.parent is None
    assert ParsedChunk("text", locator, context).parent == context
    assert [field.name for field in fields(ParsedDocument) if not field.kw_only] == [
        "parser_version",
        "media_type",
        "canonical_content",
        "chunks",
    ]
    assert ParsedDocument("v", "text/plain", b"text", (child,)).source_artifacts == {}


def test_sections_have_shared_exact_parent_and_distinct_paragraph_children() -> None:
    text = (
        "  # Engine A\n\nAt cruise, observe EGT.\n\nInspect compressor deposits.\n\n"
        "## Restrictions\n\nUse revision 3 only.\n\n# Engine B\n\nUse another limit.  "
    )
    parsed = _parser().parse(text.encode())
    _assert_exact_hierarchy(parsed)
    first_section = [
        chunk
        for chunk in parsed.chunks
        if chunk.locator.coordinates["section_path"] == ["Engine A"]
    ]
    assert len(first_section) == 3
    assert first_section[0].parent is first_section[-1].parent
    assert first_section[0].parent is not None
    assert "Engine B" not in first_section[0].parent.content
    assert parsed.chunks[3].locator.coordinates["section_path"] == ["Engine A", "Restrictions"]
    assert parsed.chunks[-1].locator.coordinates["section_path"] == ["Engine B"]
    assert parsed.chunks[1].content == "At cruise, observe EGT."


def test_soft_budget_never_slices_an_indivisible_long_sentence() -> None:
    sentence = "If the cruise condition applies, " + "inspect carefully " * 80 + "before restart."
    parsed = _parser(parent_budget=200, child_budget=100).parse(sentence.encode())
    _assert_exact_hierarchy(parsed)
    assert len(parsed.chunks) == 1
    chunk = parsed.chunks[0]
    assert chunk.content == sentence
    assert chunk.locator.coordinates["oversized"]
    assert chunk.locator.coordinates["unit_type"] == "sentence"
    assert chunk.parent is not None
    assert chunk.parent.locator.coordinates["oversized"]


def test_complete_sentence_groups_fit_soft_budgets_without_overlap() -> None:
    sentence = "At cruise, review the complete pressure and temperature trend before deciding."
    text = " ".join([sentence] * 24)
    parsed = _parser(parent_budget=320, child_budget=160).parse(text.encode())
    _assert_exact_hierarchy(parsed)
    assert len(parsed.chunks) > 4
    parents = list({id(chunk.parent): chunk.parent for chunk in parsed.chunks}.values())
    assert len(parents) > 1
    assert all(len(chunk.content) <= 160 for chunk in parsed.chunks)
    assert all(
        chunk.content.startswith("At cruise,") and chunk.content.endswith("deciding.")
        for chunk in parsed.chunks
    )
    assert all(parent is not None and len(parent.content) <= 320 for parent in parents)


@pytest.mark.parametrize(
    "sentence",
    [
        "Dr. Smith recorded the pressure with instrument No. 2 at 1.25 bar before departure.",
        "Use Fig. 2 and Eq. 3 with the U.S. reference and A. Smith's observations.",
        "The model rev. 2.1 requires a 0.05 mm tolerance, e.g. when inspecting the liner.",
    ],
)
def test_abbreviations_decimals_and_references_do_not_create_fragment_chunks(
    sentence: str,
) -> None:
    text = (sentence + " ") * 5
    parsed = _parser(parent_budget=600, child_budget=100).parse(text.encode())
    _assert_exact_hierarchy(parsed)
    assert all(chunk.content == sentence for chunk in parsed.chunks)


def test_numbered_list_is_atomic_and_oversized_even_with_punctuation() -> None:
    text = "# Inspection\n\n" + "\n\n".join(
        f"{index}. Dr. Smith checks instrument No. {index} at 1.25 bar. " + "Record the trend. " * 4
        for index in range(1, 6)
    )
    parsed = _parser(parent_budget=200, child_budget=100).parse(text.encode())
    _assert_exact_hierarchy(parsed)
    listed = [chunk for chunk in parsed.chunks if chunk.locator.coordinates["unit_type"] == "list"]
    assert len(listed) == 1
    assert "1. Dr." in listed[0].content and "5. Dr." in listed[0].content
    assert listed[0].locator.coordinates["oversized"]


def test_table_header_rows_and_footnote_stay_together_over_budget() -> None:
    table = (
        "| parameter | cruise limit (K) | revision |\n| --- | --- | --- |\n"
        + "\n".join(f"| EGT-{index} | 900 | 3 |" for index in range(30))
        + "\n* Applies only to the specified engine at cruise."
    )
    text = "# Limits\n\n" + table + "\n\nSeparate paragraph."
    parsed = _parser(parent_budget=200, child_budget=100).parse(text.encode())
    _assert_exact_hierarchy(parsed)
    tables = [chunk for chunk in parsed.chunks if chunk.locator.coordinates["unit_type"] == "table"]
    assert len(tables) == 1
    assert tables[0].content == table
    assert tables[0].locator.coordinates["oversized"]
    assert tables[0].parent is not None and tables[0].parent.locator.coordinates["oversized"]


def test_setext_and_explicit_plain_text_chapters_provide_exact_heading_context() -> None:
    text = (
        "Chapter 1 Inspection\n\nSafety restrictions.\n\n"
        "Cruise limits\n-------------\n\nUse the chart."
    )
    parsed = _parser().parse(text.encode())
    _assert_exact_hierarchy(parsed)
    assert parsed.chunks[-1].locator.coordinates["section_path"] == [
        "Chapter 1 Inspection",
        "Cruise limits",
    ]
    assert parsed.chunks[-1].content == "Use the chart."
    assert parsed.chunks[-2].content == "Cruise limits\n-------------"


def test_chinese_sentences_without_spaces_are_not_fixed_character_windows() -> None:
    sentence = "在巡航阶段观察排气温度和压力，并检查发动机型号对应的限制。"
    text = sentence * 20
    parsed = _parser(parent_budget=300, child_budget=100).parse(text.encode())
    _assert_exact_hierarchy(parsed)
    assert all(
        chunk.content == sentence * (len(chunk.content) // len(sentence)) for chunk in parsed.chunks
    )
    assert all(len(chunk.content) <= 100 for chunk in parsed.chunks)


def test_fenced_block_keeps_long_line_and_missing_closing_fence_intact() -> None:
    for closing in ("\n~~~", ""):
        block = "~~~text\n" + "unpunctuated-data" * 100 + closing
        parsed = _parser(parent_budget=200, child_budget=100).parse(block.encode())
        _assert_exact_hierarchy(parsed)
        assert len(parsed.chunks) == 1
        assert parsed.chunks[0].content == block
        assert parsed.chunks[0].locator.coordinates["unit_type"] == "code"
        assert parsed.chunks[0].locator.coordinates["oversized"]


def test_pdf_hierarchy_is_page_scoped_and_does_not_rewrite_hyphens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = [
        "# Page one\r\n\r\nCompressor pres-\r\nsure trend. " * 8,
        "",
        "# Page three\n\n" + "Third page exact sentence. " * 30,
    ]

    class Page:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class Reader:
        def __init__(self, stream: object) -> None:
            self.pages = [Page(text) for text in texts]

    monkeypatch.setattr("aerodiagnosis.ingestion.parsers.PdfReader", Reader)
    parsed = PdfParser(strategy="hierarchical", parent_budget=300, child_budget=100).parse(b"pdf")
    _assert_exact_hierarchy(parsed)
    assert {chunk.locator.coordinates["page"] for chunk in parsed.chunks} == {1, 3}
    assert "pres-\nsure" in parsed.canonical_content.decode()
    for chunk in parsed.chunks:
        assert chunk.parent is not None
        assert chunk.locator.coordinates["page"] == chunk.parent.locator.coordinates["page"]
        page = texts[chunk.locator.coordinates["page"] - 1].replace("\r\n", "\n")
        coords = chunk.locator.coordinates
        assert chunk.content == page[coords["start"] : coords["end"]]


def test_registry_defaults_to_hierarchy_and_exposes_real_ablation_configuration() -> None:
    registry = ParserRegistry(parent_budget=200, child_budget=100)
    for extension in ("txt", "md", "markdown"):
        parsed = registry.parse("document." + extension, b"Complete source sentence.")
        assert "hierarchical:200:100:semantic@2" in parsed.parser_version
        assert parsed.chunks[0].parent is not None
    for strategy in ("fixed", "structure"):
        parsed = ParserRegistry(text_strategy=strategy).parse("document.md", b"Source text.")
        assert strategy in parsed.parser_version
        assert parsed.chunks[0].parent is None
    custom = ParserRegistry({".txt": PlainTextParser()})
    assert custom.parse("document.txt", b"source").chunks[0].parent is None


def test_parser_identity_tracks_both_soft_budgets_and_preserves_legacy_versions() -> None:
    versions = {
        _parser(parent_budget=parent, child_budget=child).parser_version
        for parent in (400, 800)
        for child in (100, 200)
    }
    assert len(versions) == 4
    assert PlainTextParser().parser_version == "plain-text@2:fixed:800:100"
    assert PdfParser().parser_version == "pdf@2:page-structure:800:100"
    assert PdfParser(strategy="fixed").parser_version == "pdf@2:page-fixed:800:100"
    assert PdfParser(strategy="hierarchical").parser_version != _parser().parser_version


@pytest.mark.parametrize(("parent", "child"), [(0, 100), (100, 200), (3200, 99), (100, 0)])
def test_hierarchical_budget_validation(parent: int, child: int) -> None:
    with pytest.raises(ValueError, match="budget"):
        _parser(parent_budget=parent, child_budget=child)


def test_csv_keeps_complete_multiline_record_semantics_without_parent() -> None:
    parsed = CsvParser(rows_per_chunk=1).parse(
        b'engine,note\nA,"Do not split.\nKeep the whole record."\nB,Next record\n'
    )
    assert len(parsed.chunks) == 2
    assert "Do not split.\nKeep the whole record." in parsed.chunks[0].content
    assert all(chunk.parent is None for chunk in parsed.chunks)
    assert parsed.chunks[0].locator.coordinates == {"row_start": 2, "row_end": 2}


@pytest.mark.parametrize(
    "intro",
    [
        "Section 2 describes the method.",
        "Chapter 3 explains the operating limits.",
        "Part 4 provides the inspection procedure.",
        "第二章介绍了诊断方法。",
    ],
)
def test_section_mentions_in_prose_are_split_as_body_not_heading(intro: str) -> None:
    text = intro + " " + ("Inspect the complete compressor trend before deciding. " * 25)
    parsed = _parser(parent_budget=300, child_budget=100).parse(text.encode())
    _assert_exact_hierarchy(parsed)
    assert len(parsed.chunks) > 10
    assert all(chunk.locator.coordinates["unit_type"] != "heading" for chunk in parsed.chunks)
    assert all(chunk.locator.coordinates["section_path"] == [] for chunk in parsed.chunks)
    assert all(not chunk.locator.coordinates["oversized"] for chunk in parsed.chunks)


def test_short_complete_sentence_section_mention_does_not_replace_real_section_path() -> None:
    text = (
        "# Safety\n\nBefore proceeding, read the instructions.\nSection 2 describes the method.\n"
    )
    parsed = _parser().parse(text.encode())
    _assert_exact_hierarchy(parsed)
    assert len(parsed.chunks) == 2
    assert parsed.chunks[-1].locator.coordinates["unit_type"] == "paragraph"
    assert parsed.chunks[-1].locator.coordinates["section_path"] == ["Safety"]


def test_long_unpunctuated_section_mention_remains_oversized_body() -> None:
    text = "Section 2 discusses " + "the observed operating condition " * 80
    parsed = _parser(parent_budget=300, child_budget=100).parse(text.encode())
    _assert_exact_hierarchy(parsed)
    assert len(parsed.chunks) == 1
    assert parsed.chunks[0].content == text.rstrip()
    assert parsed.chunks[0].locator.coordinates["section_path"] == []
    assert parsed.chunks[0].locator.coordinates["unit_type"] == "sentence"
    assert parsed.chunks[0].locator.coordinates["oversized"]


@pytest.mark.parametrize(
    "title", ["Section 2 Cruise limits", "Chapter 1 Inspection", "第二章诊断方法"]
)
def test_short_explicit_chapter_titles_still_scope_exact_body(title: str) -> None:
    text = title + "\n\nFollow the procedure."
    parsed = _parser().parse(text.encode())
    _assert_exact_hierarchy(parsed)
    assert parsed.chunks[0].content == title
    assert parsed.chunks[0].locator.coordinates["unit_type"] == "heading"
    assert parsed.chunks[-1].content == "Follow the procedure."
    assert parsed.chunks[-1].locator.coordinates["section_path"] == [title]


def test_markdown_heading_with_terminal_punctuation_keeps_explicit_marker_semantics() -> None:
    text = "# Section 2: What does the method require?\n\nRead the operating conditions."
    parsed = _parser().parse(text.encode())
    _assert_exact_hierarchy(parsed)
    assert parsed.chunks[0].locator.coordinates["unit_type"] == "heading"
    assert parsed.chunks[-1].locator.coordinates["section_path"] == [
        "Section 2: What does the method require?"
    ]
