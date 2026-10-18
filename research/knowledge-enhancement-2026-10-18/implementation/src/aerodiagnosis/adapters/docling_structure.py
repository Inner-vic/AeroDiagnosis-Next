"""Chunk original Docling body items without splitting table rows or sentence continuations."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese list-introduction punctuation.
import re
from typing import Any


def structured_records(
    data: dict[str, Any], tokenizer: Any, budget: int
) -> list[tuple[str, dict[str, Any]]]:
    objects = {
        item["self_ref"]: item
        for key in ("texts", "tables", "groups", "pictures")
        for item in data.get(key, [])
    }
    units: list[tuple[str, dict[str, Any]]] = []
    headings: list[str] = []
    seen: set[str] = set()

    def metadata(item: dict[str, Any], table: bool = False) -> dict[str, Any]:
        prov = item.get("prov", [])
        return {
            "docling_refs": [item["self_ref"]],
            "headings": list(headings),
            "provenance": prov,
            "pages": sorted({p["page_no"] for p in prov}),
            "contains_table": table,
            "contains_list": item.get("label") == "list_item",
        }

    def visit(ref: str) -> None:
        if ref in seen or ref not in objects:
            return
        seen.add(ref)
        item = objects[ref]
        label = item.get("label")
        if label in {"page_header", "page_footer"}:
            return
        if label == "picture":
            for child in item.get("children", []):
                visit(child["$ref"])
            return
        if label in {"section_header", "title"}:
            headings[:] = [item.get("text", "")]
            for child in item.get("children", []):
                visit(child["$ref"])
            return
        if label == "table":
            cells = item["data"]["table_cells"]
            row_count = item["data"]["num_rows"]
            col_count = item["data"]["num_cols"]
            grid = [["" for _ in range(col_count)] for _ in range(row_count)]
            for cell in cells:
                for row in range(cell["start_row_offset_idx"], cell["end_row_offset_idx"]):
                    for col in range(cell["start_col_offset_idx"], cell["end_col_offset_idx"]):
                        grid[row][col] = cell.get("text", "").replace("\n", " ").replace("|", "/")
            header_rows = sorted(
                {c["start_row_offset_idx"] for c in cells if c.get("column_header")}
            )
            headers = [" | ".join(grid[row]) for row in header_rows]
            caption = [
                objects[c["$ref"]].get("text", "")
                for c in item.get("captions", [])
                if c["$ref"] in objects
            ]
            prefix = "\n".join(caption + headers)
            for row in range(row_count):
                if row in header_rows:
                    continue
                # A complete row is never cut at the token budget.
                text = prefix + "\n" + " | ".join(grid[row])
                units.append((text.strip(), metadata(item, True)))
            return
        text = item.get("text")
        if text and label != "caption":
            meta = metadata(item)
            if meta.get("contains_list") and units:
                old_text, old_meta = units[-1]
                if re.search(r"[:：]\s*$", old_text):
                    meta["source_semantic_context_before"] = old_text
                elif old_meta.get("contains_list"):
                    meta["source_semantic_context_before"] = old_meta.get(
                        "source_semantic_context_before", ""
                    )
            if (
                units
                and not units[-1][1]["contains_table"]
                and not units[-1][1].get("contains_list")
                and not meta.get("contains_list")
                and re.match(r"^[a-z(]", text)
                and not re.search(r"[.!?:;。]\s*$", units[-1][0])
            ):
                old, previous = units.pop()
                merged = {
                    **previous,
                    "docling_refs": previous["docling_refs"] + meta["docling_refs"],
                    "provenance": previous["provenance"] + meta["provenance"],
                    "pages": sorted(set(previous["pages"] + meta["pages"])),
                }
                units.append((old.rstrip() + " " + text.lstrip(), merged))
            else:
                units.append((text, meta))
        for child in item.get("children", []):
            visit(child["$ref"])

    for child in data["body"]["children"]:
        visit(child["$ref"])
    records: list[tuple[str, dict[str, Any]]] = []
    for text, meta in units:
        pieces = (
            [text]
            if meta["contains_table"]
            else re.split(r"(?<=[.!?。])\s+(?=[A-Z\u3400-\u9fff])", text)
        )
        for piece in pieces:
            if not piece.strip():
                continue
            if (
                records
                and records[-1][1]["contains_table"] == meta["contains_table"]
                and records[-1][1]["headings"] == meta["headings"]
                and len(tokenizer.encode(records[-1][0] + "\n" + piece, add_special_tokens=False))
                <= budget
            ):
                old, old_meta = records.pop()
                records.append(
                    (
                        old + "\n" + piece,
                        {
                            **old_meta,
                            "docling_refs": list(
                                dict.fromkeys(old_meta["docling_refs"] + meta["docling_refs"])
                            ),
                            "pages": sorted(set(old_meta["pages"] + meta["pages"])),
                            "provenance": old_meta["provenance"] + meta["provenance"],
                        },
                    )
                )
            else:
                records.append((piece, meta))
    return records
