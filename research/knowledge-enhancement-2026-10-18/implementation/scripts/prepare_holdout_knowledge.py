"""Prepare source-only Docling artifacts for an isolated holdout; never call an LLM."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

from aerodiagnosis.adapters.docling_parser import DoclingParser
from aerodiagnosis.config import RuntimeSettings

POLICY = "holdout-source-body@1"
STOP_HEADINGS = {
    "references",
    "参考文献",
    "related articles",
    "相关文章",
    "purpose of safety investigations",
    "publishing information",
    "updates to this page",
    "metrics",
    "本文评价",
    "编辑推荐",
    "download report",
    "download the report",
    "download glossary of abbreviations",
}


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def heading(text: str) -> str:
    plain = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"^[#*\s]+|[*\s]+$", "", plain).casefold()


def select_blocks(
    row: dict[str, Any], blocks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = []
    discarded = []
    images = sum(len(re.findall(r"!\[", b["text"])) for b in blocks)
    if row["completeness"] == "abstract_and_metadata_not_full_paper":
        for b in blocks:
            if re.match(r"^(?:\*\s+)?\*\*摘要[\uFF1A:]?", b["text"]) and len(b["text"]) > 80:
                # A bilingual page may combine both abstracts in the same block.
                text = b["text"].split("**Abstract:")[0].strip()
                selected.append({**b, "text": text})
        if not selected:
            raise ValueError("Declared abstract has no Chinese abstract block")
    else:
        start = 0
        if row["publisher"] == "ATSB":
            start = next((i for i, b in enumerate(blocks) if b["text"].startswith("# ")), 0)
        if row["publisher"] == "NASA":
            start = next((i for i, b in enumerate(blocks) if heading(b["text"]) == "abstract"), 0)
        for i, b in enumerate(blocks):
            text = b["text"]
            if i < start:
                discarded.append(
                    {
                        "block_id": b["block_id"],
                        "reason": "website_navigation_or_publisher_frontmatter",
                    }
                )
                continue
            h = heading(text)
            if text.startswith("#") and any(h == s or h.startswith(s + " ") for s in STOP_HEADINGS):
                discarded.extend(
                    {"block_id": x["block_id"], "reason": "bibliography_or_site_footer"}
                    for x in blocks[i:]
                )
                break
            residue = re.sub(r"!?\[[^\]]*\]\([^)]*\)", "", text)
            if not re.search(r"[A-Za-z\u4e00-\u9fff0-9]", residue):
                discarded.append({"block_id": b["block_id"], "reason": "image_or_link_only_block"})
                continue
            selected.append(b)
    if not selected:
        raise ValueError("No source body selected")
    return selected, {
        "input_blocks": len(blocks),
        "selected_blocks": len(selected),
        "discarded": discarded,
        "image_references_not_loaded": images,
        "private_use_glyph_blocks": sum(
            bool(re.search(r"[\ue000-\uf8ff]", b["text"])) for b in selected
        ),
        "equation_and_figure_recovery": "not_established; no OCR or inferred reconstruction",
    }


def render_source(blocks: list[dict[str, Any]]) -> bytes:
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    elements = []
    for b in blocks:
        text = b["text"]
        # PDF glyph fragments are sometimes mislabelled as huge headings. Keep as text.
        if text.startswith("#") and (len(text) > 180 or re.match(r"^#+\s+[([]", text)):
            text = text.lstrip("# ").strip()
        fragment = BeautifulSoup(renderer.render(text), "html.parser")
        for image in fragment.find_all("img"):
            image.replace_with(image.get("alt", ""))
        for link in fragment.find_all("a"):
            link.unwrap()
        elements.append(
            f'<section data-capture-block="{html.escape(b["block_id"])}">{fragment}</section>'
        )
    return (
        '<!doctype html><html><head><meta charset="utf-8"></head><body><main>'
        + "".join(elements)
        + "</main></body></html>"
    ).encode("utf-8")


def run(source_root: Path, output: Path, profile: Path) -> None:
    manifest = json.loads((source_root / "source_manifest.json").read_text(encoding="utf-8"))
    settings = RuntimeSettings.from_env(profile)
    if settings.knowledge_tokenizer_path is None:
        raise ValueError("Missing local tokenizer")
    parser = DoclingParser(
        ".html", tokenizer_path=settings.knowledge_tokenizer_path, token_budget=400
    )
    output.mkdir(parents=True, exist_ok=True)
    catalog = {
        "schema": "holdout-prepared-corpus@1",
        "preparation_policy": POLICY,
        "source_only": True,
        "llm_calls": 0,
        "documents": [],
    }
    for row in manifest["documents"]:
        if not row.get("candidate_used"):
            continue
        doc = row["document_id"]
        source = (source_root / "sources" / f"{doc}.txt").read_text(encoding="utf-8")
        if sha(source.encode()) != row["canonical_sha256"]:
            raise ValueError("Canonical capture changed: " + doc)
        blocks = json.loads(
            (source_root / "sources" / f"{doc}.blocks.json").read_text(encoding="utf-8")
        )
        chosen, audit = select_blocks(row, blocks)
        content = render_source(chosen)
        folder = output / doc
        folder.mkdir(exist_ok=True)
        html_path = folder / "source.html"
        artifact = folder / "parsed.json"
        preparation = {
            "policy": POLICY,
            "source_canonical_sha256": row["canonical_sha256"],
            "html_sha256": sha(content),
            "parser_version": parser.parser_version,
        }
        if (folder / "preparation.json").exists() and artifact.exists():
            if json.loads((folder / "preparation.json").read_text(encoding="utf-8")) != preparation:
                raise ValueError("Prepared version changed; use a new output directory")
            data = json.loads(artifact.read_text(encoding="utf-8"))
        else:
            html_path.write_bytes(content)
            parsed = parser.parse(content)
            data = {
                "parser_version": parsed.parser_version,
                "media_type": parsed.media_type,
                "canonical_content": parsed.canonical_content.decode("utf-8"),
                "chunks": [
                    {
                        "content": c.content,
                        "locator": c.locator.canonical(),
                        "parent": {
                            "content": c.parent.content,
                            "locator": c.parent.locator.canonical(),
                        }
                        if c.parent
                        else None,
                    }
                    for c in parsed.chunks
                ],
            }
            save(artifact, data)
            for name, value in parsed.source_artifacts.items():
                (folder / name).write_bytes(value)
            save(folder / "preparation.json", preparation)
        save(folder / "body-selection.json", audit)
        canonical = data["canonical_content"]
        for c in data["chunks"]:
            loc = c["locator"]["coordinates"]
            if canonical[loc["start"] : loc["end"]] != c["content"]:
                raise ValueError("Prepared chunk offset invalid")
        lengths = [
            len(parser._tokenizer.encode(c["content"], add_special_tokens=False, verbose=False))
            for c in data["chunks"]
        ]
        entry = {
            "document_id": "holdout-v1-" + doc,
            "original_source_id": doc,
            "path": str(html_path.relative_to(output)),
            "parsed_path": str(artifact.relative_to(output)),
            "sha256": sha(content),
            "canonical_sha256": sha(canonical.encode()),
            "source_capture_sha256": row["canonical_sha256"],
            "display_name": doc + ".html",
            "metadata": {
                "source_url": row["url"],
                "source_type": row["source_type"],
                "source_revision": "captured-2026-10-14:" + row["canonical_sha256"][:12],
                "language": row.get("source_language", "en"),
                "license_or_basis": (
                    "local research source-scoped evidence; no automatic redistribution"
                ),
                "applicability": {
                    "scope_status": "unknown",
                    "engine_models": [],
                    "flight_phases": [],
                },
            },
            "source_family": row["family_id"],
            "completeness": row["completeness"],
            "chunks": len(lengths),
            "input_tokens_e5": sum(lengths),
            "input_utf8_bytes": sum(len(c["content"].encode()) for c in data["chunks"]),
            "max_chunk_tokens_e5": max(lengths),
            "oversize_soft_budget_chunks": sum(x > 400 for x in lengths),
            "structure_audit": audit,
        }
        catalog["documents"].append(entry)
        save(output / "catalog.json", catalog)
        print(
            json.dumps(
                {
                    "source": doc,
                    "chunks": len(lengths),
                    "tokens_e5": sum(lengths),
                    "glyph_warning_blocks": audit["private_use_glyph_blocks"],
                }
            ),
            flush=True,
        )
    catalog["status"] = "prepared_not_translated_not_graph_extracted"
    catalog["totals"] = {
        "documents": len(catalog["documents"]),
        "chunks": sum(d["chunks"] for d in catalog["documents"]),
        "tokens_e5": sum(d["input_tokens_e5"] for d in catalog["documents"]),
        "utf8_bytes": sum(d["input_utf8_bytes"] for d in catalog["documents"]),
        "oversize_soft_budget_chunks": sum(
            d["oversize_soft_budget_chunks"] for d in catalog["documents"]
        ),
    }
    save(output / "catalog.json", catalog)
    print(json.dumps(catalog["totals"]), flush=True)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--source-root", type=Path, default=Path("evaluation/gas_path_holdout_v1_candidate")
    )
    cli.add_argument("--output", type=Path, default=Path(".runtime/holdout-v1/prepared"))
    cli.add_argument("--profile", type=Path, default=Path(".runtime/fusion/profile"))
    args = cli.parse_args()
    run(args.source_root.resolve(), args.output.resolve(), args.profile.resolve())


if __name__ == "__main__":
    main()
