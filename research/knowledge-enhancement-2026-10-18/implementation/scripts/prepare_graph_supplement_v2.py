# ruff: noqa: E501
"""Independent v2 source preparation, reusing mature parsing/paid-call validators."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import prepare_capability_holdout as mature
import prepare_strict_graph_holdout as strict

from aerodiagnosis.domain import make_chunk_id
from aerodiagnosis.ingestion import ParserRegistry
from aerodiagnosis.ports import VectorChunk

ROOT = Path("evaluation/graph_supplement_v2_holdout_candidate")
OUT = Path(".runtime/graph-supplement-v2-holdout")
REVIEW = Path(
    "D:/zhaoweihao3/Desktop/Project/AeroDiagnosis/.runtime/graph-next-source-review/review-candidates.json"
)
read, save, sha, digest = mature.read, mature.save, mature.sha, strict.digest


def configure():
    strict.ROOT, strict.OUT = ROOT, OUT
    mature.ROOT, mature.OUT = ROOT, OUT


def anchor(review=None):
    configure()
    strict.anchor(review)


def prepare():
    assert not (ROOT / "source-catalog.json").exists()
    strategy = read(ROOT / "strategy-freeze.json")
    for p, h in strategy["source_hashes"].items():
        assert digest(Path(p)) == h, p
    review = read(REVIEW)
    entries = review["sources"]
    assert len(entries) <= 16
    selected = []
    seen = set()
    for r in entries:
        if r["decision"] not in {"accept", "accepted"} or r["family"] in seen:
            continue
        if len(selected) >= 10:
            break
        assert r["expert_reviewed"] is False
        selected.append(r)
        seen.add(r["family"])
    parser = ParserRegistry(text_strategy="hierarchical", parent_budget=3200, child_budget=600)
    docs = []
    chunks = []
    origins = []
    for r in selected:
        full = Path(r["source_text_file"]).read_text(encoding="utf8")
        start, end = r.get("technical_start", 0), r.get("technical_end", len(full))
        assert 0 <= start < end <= len(full)
        text = full[start:end]
        assert len(text) >= 1000
        doc = "gs2-" + r["source_id"]
        h = sha(text)
        version = doc + "-" + h[:24]
        saved = ROOT / "technical-sources" / (doc + ".txt")
        saved.parent.mkdir(exist_ok=True)
        saved.write_text(text, encoding="utf8", newline="\n")
        parsed = parser.parse("source.txt", text.encode())
        assert parsed.canonical_content.decode() == text
        ids = []
        for item in parsed.chunks:
            loc = item.locator.canonical()
            co = loc["coordinates"]
            co["canonical_sha256"] = h
            assert text[co["start"] : co["end"]] == item.content
            key = make_chunk_id(version, loc, item.content)
            ids.append(key)
            chunks.append(
                asdict(
                    VectorChunk(
                        key,
                        doc,
                        version,
                        item.content,
                        {
                            "locator": loc,
                            "original_source_id": r["source_id"],
                            "document_metadata": {
                                "source_url": r["url"],
                                "source_type": "public_primary_technical_source",
                                "source_revision": h,
                                "language": "en",
                                "jurisdiction": r["agency"],
                                "source_family": r["family"],
                            },
                        },
                    )
                )
            )
        docs.append(
            dict(
                document_id=doc,
                version_id=version,
                source_key=doc + ":" + version,
                content=text,
                source_sha256=h,
                source_family=r["family"],
                original_source_id=r["source_id"],
                source_url=r["url"],
                agency=r["agency"],
                chunk_ids=ids,
                source_limitations=[r.get("reason", "")],
                event_identity={"title": r["title"]},
                technical_source_path=saved.as_posix(),
            )
        )
        origins.append(
            dict(
                source_id=r["source_id"],
                source_text_file=r["source_text_file"],
                full_sha256=sha(full),
                full_byte_sha256=digest(Path(r["source_text_file"])),
                technical_start=start,
                technical_end=end,
                technical_sha256=h,
            )
        )
    save(
        ROOT / "source-catalog.json",
        dict(
            documents=docs,
            chunks=chunks,
            source_only=True,
            extraction="Mature hierarchical parser child600/parent3200; complete continuous reviewed technical windows, summaries retained",
        ),
    )
    save(ROOT / "source-selection-after-review.json", dict(review=review, items=origins))
    save(
        ROOT / "source-freeze.json",
        dict(
            created_at_utc=datetime.now(UTC).isoformat(),
            before_graph_and_questions=True,
            source_catalog_sha256=digest(ROOT / "source-catalog.json"),
            source_selection_sha256=digest(ROOT / "source-selection-after-review.json"),
            strategy_sha256=digest(ROOT / "strategy-freeze.json"),
            documents=len(docs),
            chunks=len(chunks),
            families=len(seen),
            expert_reviewed=False,
        ),
    )
    print("Prepared", len(docs), len(chunks), sum(len(d["content"]) for d in docs), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "translate", "extract", "anchor", "budget"])
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    configure()
    if a.stage == "prepare":
        prepare()
    elif a.stage == "translate":
        strict.translate(a.limit)
    elif a.stage == "extract":
        strict.extract(a.limit)
    elif a.stage == "anchor":
        strict.anchor()
    else:
        print({k: v for k, v in mature.client().summary().items() if k != "receipts"})
