# ruff: noqa: E501, RUF001
"""Prepare new source-only mechanism holdout. Never reads retrieval predictions."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import prepare_capability_holdout as mature

from aerodiagnosis.domain import make_chunk_id
from aerodiagnosis.ingestion import ParserRegistry
from aerodiagnosis.ports import VectorChunk

ROOT = Path("evaluation/strict_graph_holdout_v1_candidate")
OUT = Path(".runtime/strict-graph-holdout-v1")
REVIEW = Path(
    "D:/zhaoweihao3/Desktop/Project/AeroDiagnosis/.runtime/strict-source-review/source_review_v1.json"
)
read, save, sha = mature.read, mature.save, mature.sha


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if (ROOT / "source-catalog.json").exists():
        raise ValueError("Immutable source catalog already exists")
    review = read(REVIEW)
    assert not any(
        review.get(k, True)
        for k in ["expert_reviewed", "questions_seen", "predictions_seen", "author_gold_seen"]
    )
    assert review["coverage_complete"]
    manifest = (
        read(ROOT / "source_manifest.json")["documents"]
        + read(ROOT / "extra-source-candidates.json")["documents"]
    )
    byid = {r["document_id"]: r for r in manifest}
    accepted = [
        r
        for r in review["items"]
        if r["decision"] == "accept" and r["prior_overlap"]["status"] == "no_identified_overlap"
    ]
    # Selection is the independently reviewed source order, capped at the registered20 families.
    selected = []
    for row in accepted:
        if (
            row["source_family"] not in {r["source_family"] for r in selected}
            and len(selected) < 20
        ):
            selected.append(row)
    if not selected:
        raise ValueError("No independently eligible source families")
    parser = ParserRegistry(text_strategy="hierarchical", parent_budget=3200, child_budget=600)
    docs = []
    chunks = []
    provenance = []
    for row in selected:
        origin = row["document_id"]
        meta = byid[origin]
        path = Path(row["source_text_path"])
        full = path.read_text(encoding="utf8")
        assert sha(full) == row["source_text_sha256"], "Reviewed source canonical hash mismatch"
        lo, hi = row["technical_start"], row["technical_end"]
        assert 0 <= lo < hi <= len(full)
        text = full[lo:hi]
        assert len(text) >= 1000
        h = sha(text)
        doc = "sg1-" + origin
        version = doc + "-" + h[:24]
        technical = ROOT / "technical-sources" / (origin + ".txt")
        technical.parent.mkdir(exist_ok=True)
        technical.write_text(text, encoding="utf8", newline="\n")
        parsed = parser.parse("source.txt", text.encode())
        assert parsed.canonical_content.decode() == text
        ids = []
        for item in parsed.chunks:
            loc = item.locator.canonical()
            co = loc["coordinates"]
            co["canonical_sha256"] = h
            assert text[co["start"] : co["end"]] == item.content
            identifier = make_chunk_id(version, loc, item.content)
            ids.append(identifier)
            chunks.append(
                asdict(
                    VectorChunk(
                        identifier,
                        doc,
                        version,
                        item.content,
                        {
                            "locator": loc,
                            "original_source_id": origin,
                            "document_metadata": {
                                "source_url": meta["url"],
                                "source_type": "official_accident_investigation",
                                "source_revision": h,
                                "language": "en",
                                "jurisdiction": meta["publisher"],
                                "source_family": row["source_family"],
                            },
                        },
                    )
                )
            )
        docs.append(
            {
                "document_id": doc,
                "version_id": version,
                "source_key": doc + ":" + version,
                "content": text,
                "source_sha256": h,
                "source_family": row["source_family"],
                "original_source_id": origin,
                "source_url": meta["url"],
                "agency": meta["publisher"],
                "chunk_ids": ids,
                "source_limitations": row.get("limitations", []),
                "event_identity": row.get("event_identity"),
                "technical_source_path": technical.as_posix(),
            }
        )
        provenance.append(
            {
                "document_id": doc,
                "source_text_path": path.as_posix(),
                "saved_source_sha256": sha(full),
                "saved_source_byte_sha256": digest(path),
                "technical_start": lo,
                "technical_end": hi,
                "technical_sha256": h,
                "offset_semantics": "Unicode characters in read_text utf8 with LF newline normalization",
                "source_family": row["source_family"],
            }
        )
    save(
        ROOT / "source-catalog.json",
        {
            "schema": "strict-mechanism-source-catalog@1",
            "documents": docs,
            "chunks": chunks,
            "source_only": True,
            "extraction": "Mature hierarchical parser child600/parent3200, unchanged natural source chunks; no artificial multi-hop splitting",
        },
    )
    save(
        ROOT / "source-selection-after-review.json",
        {
            "source_only": True,
            "items": provenance,
            "review": review,
            "accepted_not_selected_due_registered_cap": [
                r["document_id"] for r in accepted if r not in selected
            ],
        },
    )
    save(
        ROOT / "source-freeze.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "before_graph_and_questions": True,
            "source_catalog_sha256": digest(ROOT / "source-catalog.json"),
            "source_selection_sha256": digest(ROOT / "source-selection-after-review.json"),
            "strategy_sha256": digest(ROOT / "strategy-freeze.json"),
            "documents": len(docs),
            "chunks": len(chunks),
            "families": len({d["source_family"] for d in docs}),
            "expert_reviewed": False,
        },
    )
    print(
        "Prepared",
        len(docs),
        "families",
        len(chunks),
        "chunks",
        sum(len(d["content"]) for d in docs),
        "characters",
        flush=True,
    )


def translate(limit):
    mature.ROOT, mature.OUT = ROOT, OUT
    mature.translate(limit)


EXTRACT = """仅依据原文抽取最多18条有向技术事实，不读取或生成问题。研究对象为发动机及直接关联的润滑、燃油和机械失效。优先覆盖报告支持的完整因果机理链、不同段落的衔接实体、观测症状、明确排除和不确定性；不要只抄结论或把句子当实体。最多18条是上限，不要求凑数。源/目标实体name为2至16字短中文，surface必须逐字连续存在于原文；同一物理对象跨段落使用完全相同的surface和role，不可为连接路径合并不同轴、轴承、部件。kind=Concept，role稳定中文物理角色。因果方向是原因到结果；症状指向所指示对象只能INDICATES；时间先后不自动是因果。不能把怀疑根因标成确定，也不能把维修预防要求推出确定事故原因。源文明确不确定或未查明必须保留。每条quote完整连续且精确位于某一chunk；实体surface必须位于quote或supporting_quotes，补充引文也必须是同一来源实际连续片段。certainty只用established/possible/suspected，polarity用positive/negative。condition_zh完整保留具体型号/事件/失效级别和限定，禁止外推同系列全部发动机。关系限定CAUSES,CONTRIBUTES_TO,INDICATES,APPLIES_TO,REQUIRES,PREVENTS,HAS_PART,DEFINES,EXEMPTS。返回JSON {assertions:[{source:{name,surface,role,kind},target:{name,surface,role,kind},relation,chunk,quote,certainty,polarity,condition_zh,supporting_quotes:[{chunk,quote}]}]}。只输出所需JSON，无额外说明。"""


def extract(limit):
    # Graph extraction reads only the frozen original; independent of translation and questions.
    data = read(ROOT / "source-catalog.json")
    byid = {c["chunk_id"]: c for c in data["chunks"]}
    model = mature.client()
    called = 0
    for i, doc in enumerate(data["documents"]):
        target = OUT / "extraction" / (doc["document_id"] + ".json")
        if target.exists():
            assert read(target)["source_sha256"] == doc["source_sha256"]
            continue
        cs = [byid[k] for k in doc["chunk_ids"]]
        reply = model.complete(
            task="strict-holdout-source-graph-v1",
            stage="corpus_graph",
            max_tokens=7000,
            instruction=EXTRACT,
            payload={
                "source_document": doc["document_id"],
                "agency": doc["agency"],
                "source_limitations": doc["source_limitations"],
                "source_chunks": [
                    {"key": str(j), "english": c["content"]} for j, c in enumerate(cs)
                ],
            },
        )
        save(
            target,
            {
                "document_id": doc["document_id"],
                "source_sha256": doc["source_sha256"],
                "chunk_key_map": {str(j): c["chunk_id"] for j, c in enumerate(cs)},
                "assertions": reply.value["assertions"],
                "receipt": reply.receipt,
            },
        )
        called += 1
        print("Extracted", i + 1, "/", len(data["documents"]), flush=True)
        if limit and called >= limit:
            break


def anchor(review=None):
    import compile_capability_holdout_graph as compiler

    compiler.ROOT, compiler.OUT = ROOT, OUT
    compiler.compile_graph(Path(review) if review else None)
    if not review:
        save(
            OUT / "raw-graph-anchors/source-graph-candidate.json",
            read(ROOT / "source-graph-candidate.json"),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "translate", "extract", "anchor"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--review")
    a = p.parse_args()
    {
        "prepare": prepare,
        "translate": lambda: translate(a.limit),
        "extract": lambda: extract(a.limit),
        "anchor": lambda: anchor(a.review),
    }[a.stage]()
