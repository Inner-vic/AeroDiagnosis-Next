# ruff: noqa: RUF001
"""Exact source-span hierarchy with semantic boundaries and soft packing budgets.

This module never cuts at a character budget. Unpunctuated sentences, tables,
lists and fenced blocks stay intact even when they exceed the configured budget.
PDF layout is not reconstructed: the input is the page's extracted canonical text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ALGORITHM_VERSION = "semantic@2"
_HEADING = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)\s*#*\s*$")
_NUMBERED_HEADING = re.compile(
    r"^[ \t]*(?:section|chapter|part)[ \t]+\d+\b.*$|^[ \t]*第[^\s]{1,12}[章节部].*$",
    re.IGNORECASE,
)
_LIST = re.compile(
    r"^[ \t]*(?:[-+*•]|\d+[.)]|[a-zA-Z][.)]|[ivxIVX]+[.)]"
    r"|\((?:[A-Za-z]|[0-9]{1,3}|[ivxlcdm]{2,5})\)|[一二三四五六七八九十]+[、.])[ \t]+"
)
_ABBREVIATIONS = frozenset(
    {
        "no",
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "capt",
        "sr",
        "jr",
        "st",
        "fig",
        "eq",
        "ref",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "approx",
        "dept",
        "vol",
        "rev",
    }
)
_CLOSING = frozenset(chr(34) + chr(39) + "”’»)]】）")


@dataclass(frozen=True, slots=True)
class SourceSpan:
    start: int
    end: int
    section_path: tuple[str, ...]
    unit_type: str


def algorithm_version() -> str:
    return _ALGORITHM_VERSION


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _table_line(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.count("|") >= 2
        or stripped.count("\t") >= 2
        or len(re.split(r" {2,}", stripped)) >= 3
    )


def _fence(line: str) -> str | None:
    match = re.match(r"^[ \t]{0,3}(" + chr(96) + r"{3,}|~{3,})", line)
    return match[1] if match else None


def _plain_heading(line: str) -> bool:
    """Recognize short explicit chapter labels, not prose mentioning a section.

    Plain extracted text lacks Markdown's unambiguous heading markers. Treat
    long lines and complete sentences conservatively as body text rather than
    putting a whole paragraph into every descendant's section_path.
    """
    stripped = line.strip()
    if len(stripped) > 160 or not _NUMBERED_HEADING.match(stripped):
        return False
    without_closing = stripped.rstrip("".join(_CLOSING))
    return not without_closing.endswith((".", "!", "?", "。", "！", "？", ";", "；"))


def _structural_units(text: str) -> list[SourceSpan]:
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    result: list[SourceSpan] = []
    headings: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        first = index
        line = lines[index].rstrip("\n")
        heading = _HEADING.match(line)
        setext = (
            index + 1 < len(lines)
            and bool(re.fullmatch(r"[ \t]*(?:={3,}|-{3,})[ \t]*\n?", lines[index + 1]))
            and not _LIST.match(line)
            and not _table_line(line)
        )
        if heading or setext or _plain_heading(line):
            level = (
                len(heading[1]) if heading else (1 if not setext or "=" in lines[index + 1] else 2)
            )
            title = heading[2].strip() if heading else line.strip()
            headings = [item for item in headings if item[0] < level]
            headings.append((level, title))
            kind = "heading"
            index += 2 if setext else 1
        elif fence := _fence(line):
            kind = "code"
            index += 1
            while index < len(lines):
                close = _fence(lines[index])
                index += 1
                if close and close[0] == fence[0] and len(close) >= len(fence):
                    break
        elif _table_line(line):
            # Retain the complete table and adjacent explanatory footnotes. No
            # attempt to infer PDF columns or reconstruct flattened cell geometry.
            kind = "table"
            index += 1
            while index < len(lines) and lines[index].strip():
                if _HEADING.match(lines[index]) or _fence(lines[index]):
                    break
                index += 1
        elif _LIST.match(line):
            kind = "list"
            index += 1
            while index < len(lines):
                if not lines[index].strip():
                    next_index = index + 1
                    while next_index < len(lines) and not lines[next_index].strip():
                        next_index += 1
                    if next_index < len(lines) and _LIST.match(lines[next_index]):
                        index = next_index
                        continue
                    break
                if (
                    _HEADING.match(lines[index])
                    or _plain_heading(lines[index])
                    or _table_line(lines[index])
                    or _fence(lines[index])
                ):
                    break
                # Wrapped list lines remain part of the same indivisible list.
                index += 1
        else:
            kind = "paragraph"
            index += 1
            while index < len(lines) and lines[index].strip():
                if (
                    _HEADING.match(lines[index])
                    or _plain_heading(lines[index])
                    or _LIST.match(lines[index])
                    or _table_line(lines[index])
                    or _fence(lines[index])
                ):
                    break
                # Let a following Setext underline be parsed with its title.
                if index + 1 < len(lines) and re.fullmatch(
                    r"[ \t]*(?:={3,}|-{3,})[ \t]*\n?", lines[index + 1]
                ):
                    break
                index += 1
        start, end = _trim(text, offsets[first], offsets[index])
        result.append(SourceSpan(start, end, tuple(title for _, title in headings), kind))
    return result


def _protected_period(text: str, position: int, start: int) -> bool:
    if position + 1 < len(text) and text[position + 1].isdigit():
        return True  # decimal, numeric/version component
    token_match = re.search(r"([\w.]+)$", text[max(start, position - 32) : position])
    if not token_match:
        return False
    token = token_match[1]
    return (
        token.isdigit()  # ordinal/list/reference number, never a sentence cut
        or token.lower() in _ABBREVIATIONS
        or (len(token) == 1 and token.isalpha())
        or bool(re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]?", token))
    )


def _sentences(text: str, unit: SourceSpan) -> list[SourceSpan]:
    spans: list[SourceSpan] = []
    cursor = unit.start
    position = unit.start
    while position < unit.end:
        punctuation = text[position]
        if punctuation not in ".!?。！？":
            position += 1
            continue
        if punctuation == "." and _protected_period(text, position, unit.start):
            position += 1
            continue
        end = position + 1
        while end < unit.end and text[end] in ".!?。！？":
            end += 1
        while end < unit.end and text[end] in _CLOSING:
            end += 1
        # Chinese terminators are valid without intervening spaces; English
        # punctuation without whitespace is conservatively kept in the sentence.
        if punctuation in "。！？" or end == unit.end or text[end].isspace():
            start, right = _trim(text, cursor, end)
            if start < right:
                spans.append(SourceSpan(start, right, unit.section_path, "sentence"))
            cursor = end
            position = end
        else:
            position += 1
    start, end = _trim(text, cursor, unit.end)
    if start < end:
        spans.append(SourceSpan(start, end, unit.section_path, "sentence"))
    return spans


def _sentence_groups(text: str, unit: SourceSpan, budget: int) -> list[SourceSpan]:
    if (
        unit.unit_type not in {"paragraph", "sentence_group", "sentence"}
        or unit.end - unit.start <= budget
    ):
        return [unit]
    result: list[SourceSpan] = []
    pending: list[SourceSpan] = []
    for sentence in _sentences(text, unit):
        if pending and sentence.end - pending[0].start > budget:
            result.append(
                SourceSpan(
                    pending[0].start,
                    pending[-1].end,
                    unit.section_path,
                    "sentence" if len(pending) == 1 else "sentence_group",
                )
            )
            pending = []
        pending.append(sentence)
    if pending:
        result.append(
            SourceSpan(
                pending[0].start,
                pending[-1].end,
                unit.section_path,
                "sentence" if len(pending) == 1 else "sentence_group",
            )
        )
    return result


def hierarchical_spans(
    text: str, *, parent_budget: int, child_budget: int
) -> tuple[tuple[SourceSpan, SourceSpan], ...]:
    """Return (fine span, parent span), with exact, non-overlapping fine spans."""
    units = [
        segment
        for unit in _structural_units(text)
        for segment in _sentence_groups(text, unit, parent_budget)
    ]
    groups: list[list[SourceSpan]] = []
    pending: list[SourceSpan] = []
    for unit in units:
        if pending and (
            unit.section_path != pending[0].section_path
            or unit.unit_type == "heading"
            or unit.end - pending[0].start > parent_budget
        ):
            groups.append(pending)
            pending = []
        pending.append(unit)
    if pending:
        groups.append(pending)
    result: list[tuple[SourceSpan, SourceSpan]] = []
    for group in groups:
        parent = SourceSpan(
            group[0].start,
            group[-1].end,
            group[0].section_path,
            "section_group" if group[0].section_path else "paragraph_group",
        )
        for unit in group:
            result.extend((child, parent) for child in _sentence_groups(text, unit, child_budget))
    return tuple(result)
