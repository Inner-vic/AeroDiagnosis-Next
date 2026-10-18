# ruff: noqa: E501, RUF001
"""Source-only preparation, budgeted translation and graph extraction for frozen-policy holdout."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import httpx
from pypdf import PdfReader

from aerodiagnosis.adapters.llm.research_client import (
    CompletionStructureError,
    CompletionTruncatedError,
    ResearchCallError,
    ResearchClient,
    load_credential,
)
from aerodiagnosis.domain import make_chunk_id
from aerodiagnosis.ingestion import ParserRegistry
from aerodiagnosis.ports import VectorChunk

ROOT = Path("evaluation/retrieval_capability_holdout_v1_candidate")
OUT = Path(".runtime/holdout-capability-v1")


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()


class BoundedFormatClient(ResearchClient):
    def complete(self, **kwargs):
        try:
            return super().complete(**kwargs)
        except (CompletionStructureError, CompletionTruncatedError):
            kwargs["instruction"] += (
                " IMPORTANT: Return one syntactically valid JSON object with quoted keys and escaped string newlines. No markdown or outside text. This is the sole bounded format-repair attempt."
            )
            kwargs["max_tokens"] = min(8192, kwargs.get("max_tokens", 2048) * 2)
            return super().complete(**kwargs)


def client():
    key = load_credential(Path("D:/zhaoweihao3/Desktop/Project/AeroDiagnosis/env.txt"))

    async def request(body):
        async with asyncio.timeout(120):
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60, connect=15), follow_redirects=False
            ) as http:
                response = await http.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": "Bearer " + key},
                    json=body,
                )
                if response.status_code != 200:
                    raise ResearchCallError("Provider HTTP " + str(response.status_code))
                return response.json()

    return BoundedFormatClient(
        api_key=key,
        runtime=Path(".runtime/thesis-research-provider-v1"),
        model="deepseek-flash",
        cap_cny=42,
        transport=lambda body: asyncio.run(request(body)),
    )


def prepare():
    selection = read(ROOT / "source-selection-after-review.json")
    incorporated = {
        r["document_id"]: r for r in read(ROOT / "incorporated-easa-manifest.json")["sources"]
    }
    parser = ParserRegistry(text_strategy="hierarchical", parent_budget=3200, child_budget=600)
    docs = []
    chunks = []

    def add(key, text, url, family, origin, agency):
        doc = "cap1-" + key
        h = sha(text)
        version = doc + "-" + h[:24]
        parsed = parser.parse("source.txt", text.encode())
        assert parsed.canonical_content.decode() == text
        ids = []
        spans = []
        for item in parsed.chunks:
            loc = item.locator.canonical()
            a = loc["coordinates"]
            assert text[a["start"] : a["end"]] == item.content
            current_heading = bool(re.fullmatch(r"\([a-z]\) [A-Z][^\n]{1,75}", item.content))
            if spans:
                previous = spans[-1]["coordinates"]
                prior = text[previous["start"] : previous["end"]]
                prior_heading = bool(re.fullmatch(r"\([a-z]\) [A-Z][^\n]{1,75}", prior))
                # Source-only conservative binding: a truncated natural sentence is never an independently translated evidence block.
                if (
                    not prior_heading
                    and not current_heading
                    and prior.rstrip()[-1] not in ".!?;:\u3002"
                ):
                    previous["end"] = a["end"]
                    previous["unit_type"] = "source_bound_continuation"
                    continue
            spans.append(deepcopy(loc))
        for locator in spans:
            coords = locator["coordinates"]
            content = text[coords["start"] : coords["end"]]
            coords["canonical_sha256"] = h
            key = make_chunk_id(version, locator, content)
            ids.append(key)
            chunks.append(
                asdict(
                    VectorChunk(
                        key,
                        doc,
                        version,
                        content,
                        {
                            "locator": locator,
                            "original_source_id": origin,
                            "document_metadata": {
                                "source_url": url,
                                "source_type": "historical_airworthiness_directive",
                                "source_revision": h,
                                "language": "en",
                                "jurisdiction": agency,
                                "source_family": family,
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
                "source_family": family,
                "original_source_id": origin,
                "source_url": url,
                "agency": agency,
                "chunk_ids": ids,
            }
        )

    for row in selection["sources"]:
        q = Path(row["saved_source_path"])
        q = q if q.exists() else ROOT / q
        text = q.read_text(encoding="utf8")
        assert sha(text) == row["saved_source_sha256"]
        add(row["document_id"], text, row["url"], row["source_family"], row["document_id"], "FAA")
        e = incorporated[row["document_id"]]
        if e["status"] != "captured":
            continue
        pages = [p.extract_text() for p in PdfReader(e["pdf_path"]).pages]
        full = "\n\n".join(pages)
        # Canonical PDF extraction is retained in full. Technical excerpt is a single exact continuous span.
        start = re.search(r"Applicability\s*:", full, re.I)
        end = re.search(r"Ref\.?\s*Publications\s*:", full, re.I)
        if not start or not end or end.start() <= start.start():
            raise ValueError("EASA technical span unavailable " + e["easa_id"])
        text = full[start.start() : end.start()].rstrip()
        target = ROOT / "sources" / ("easa-" + e["easa_id"] + ".plain-full.txt")
        target.write_text(full, encoding="utf8", newline="\n")
        technical = target.with_name("easa-" + e["easa_id"] + ".technical.txt")
        technical.write_text(text, encoding="utf8", newline="\n")
        add("easa-" + e["easa_id"], text, e["pdf_url"], row["source_family"], e["easa_id"], "EASA")
    save(
        ROOT / "source-catalog.json",
        {
            "schema": "capability-sources@1",
            "documents": docs,
            "source_only": True,
            "chunks": chunks,
            "source_family_count": len({d["source_family"] for d in docs}),
            "extraction": "Existing hierarchical parser child600 parent3200; one continuous FAA or EASA technical excerpt per document; separate jurisdictions and canonical hashes.",
        },
    )
    save(
        ROOT / "source-freeze.json",
        {
            "created_before_questions": True,
            "documents": len(docs),
            "chunks": len(chunks),
            "families": len({d["source_family"] for d in docs}),
            "source_only": True,
            "source_catalog_sha256": hashlib.sha256(
                (ROOT / "source-catalog.json").read_bytes()
            ).hexdigest(),
            "selection_sha256": hashlib.sha256(
                (ROOT / "source-selection-after-review.json").read_bytes()
            ).hexdigest(),
            "fusion_policy_sha256": hashlib.sha256(
                (ROOT / "fusion-selection-freeze.json").read_bytes()
            ).hexdigest(),
        },
    )
    print(
        "Prepared",
        len(docs),
        "documents",
        len(chunks),
        "chunks",
        sum(len(d["content"]) for d in docs),
        "characters",
        flush=True,
    )


def translate(limit):
    data = read(ROOT / "source-catalog.json")
    batches = []
    cached = (
        read(OUT / "reused-reviewed-translations.json")
        if (OUT / "reused-reviewed-translations.json").exists()
        else {}
    )
    missing = [c for c in data["chunks"] if c["chunk_id"] not in cached]
    for c in missing:
        if len(c["content"]) > 20000:
            raise ValueError("Unexpected oversized chunk")
        if not batches or sum(len(x["content"]) for x in batches[-1]) + len(c["content"]) > 3200:
            batches.append([])
        batches[-1].append(c)
    model = client()
    called = 0
    for i, batch in enumerate(batches):
        target = OUT / "translation-batches" / (str(i).zfill(3) + ".json")
        request_sha = sha(json.dumps(batch, ensure_ascii=False, sort_keys=True))
        if target.exists():
            assert read(target)["request_source_sha256"] == request_sha
            continue
        reply = model.complete(
            task="capability-holdout-translate-v1",
            stage="supplemental_review",
            max_tokens=(8192 if sum(len(c["content"]) for c in batch) > 4200 else 4096),
            instruction="Translate each item completely to Chinese, preserving every condition, exception, negation, uncertainty, numeric value/unit and identifier and table/list association. Do not add facts or summarize. compressor=压气机, stall=失速, surge=喘振, spoolie=连接套管. Headings and legal technical wording must remain faithful. Return translations:[{key,text_zh}]. Include every supplied key exactly once.",
            payload={"items": [{"key": str(j), "text": c["content"]} for j, c in enumerate(batch)]},
        )
        values = reply.value.get("translations", reply.value.get("items", []))
        mapping = {str(x["key"]): x.get("text_zh", x.get("text")) for x in values}
        if not values and set(reply.value) == set(map(str, range(len(batch)))):
            mapping = reply.value
            values = list(reply.value.values())
        if (
            set(mapping) != set(map(str, range(len(batch))))
            or len(values) != len(batch)
            or any(not isinstance(t, str) or not t.strip() for t in mapping.values())
        ):
            raise ValueError("Translation key mismatch")
        save(
            target,
            {
                "request_source_sha256": request_sha,
                "translations": [
                    {"chunk_id": c["chunk_id"], "text_zh": mapping[str(j)]}
                    for j, c in enumerate(batch)
                ],
                "receipt": reply.receipt,
            },
        )
        called += 1
        print("Translated", i + 1, "/", len(batches), flush=True)
        if limit and called >= limit:
            break
    if all(
        (OUT / "translation-batches" / (str(i).zfill(3) + ".json")).exists()
        for i in range(len(batches))
    ):
        mapping = {
            **cached,
            **{
                x["chunk_id"]: x["text_zh"]
                for q in sorted((OUT / "translation-batches").glob("*.json"))
                for x in read(q)["translations"]
            },
        }
        for c in data["chunks"]:
            c["metadata"].update(
                display_content_zh=mapping[c["chunk_id"]],
                display_translation_source_sha256=sha(c["content"]),
            )
        save(ROOT / "translated-corpus-candidate.json", data)
        print("All translations complete", flush=True)


def extract(limit):
    data = read(ROOT / "translated-corpus-candidate.json")
    byid = {c["chunk_id"]: c for c in data["chunks"]}
    model = client()

    def extract_one(pair):
        i, doc = pair
        target = OUT / "extraction" / (doc["document_id"] + ".json")
        if target.exists():
            return
        chunks = [byid[k] for k in doc["chunk_ids"]]
        reply = model.complete(
            task="capability-holdout-source-graph-v1",
            stage="corpus_graph",
            max_tokens=6000,
            instruction="仅依据本文件原文抽取至多14条重要的有向事实，不能读取或生成题目。涵盖故障机理、部件、症状、适用范围、维修措施、明示例外和定义。每个实体name必须是2至16字的简短中文标签，英文实体名不合格；surface必须是原文中真实连续出现的英文片段，用于身份连接。同一真实对象在多个段落保持完全相同的role和surface，不能因中文名相似合并不同对象。kind固定Concept；role用中文稳定物理角色。不要从预防要求推出已确定因果；不补外部SB数值或条件。每条关系需要一段完整位于一个提供chunk内的原文连续精确quote，必要附加条件可放supporting_quotes。source和target的surface必须能在quote或supporting_quotes中直接找到。关系仅限CAUSES,CONTRIBUTES_TO,INDICATES,APPLIES_TO,REQUIRES,PREVENTS,HAS_PART,DEFINES,EXEMPTS。返回JSON对象 assertions:[{source:{name,surface,role,kind},target:{name,surface,role,kind},relation,chunk,quote,certainty,polarity,condition_zh,supporting_quotes:[{chunk,quote}]}]。certainty仅established/possible/suspected，polarity仅positive/negative，condition_zh必须保留型号/对象/条件/例外和管辖区。FAA和EASA不得混用。",
            payload={
                "source_document": doc["document_id"],
                "jurisdiction": doc["agency"],
                "source_chunks": [
                    {
                        "key": str(j),
                        "english": c["content"],
                    }
                    for j, c in enumerate(chunks)
                ],
            },
        )
        save(
            target,
            {
                "document_id": doc["document_id"],
                "source_sha256": doc["source_sha256"],
                "chunk_key_map": {str(j): c["chunk_id"] for j, c in enumerate(chunks)},
                "assertions": reply.value["assertions"],
                "receipt": reply.receipt,
            },
        )
        print("Extracted", i + 1, "/", len(data["documents"]), flush=True)

    from concurrent.futures import ThreadPoolExecutor

    pending = [
        (i, d)
        for i, d in enumerate(data["documents"])
        if not (OUT / "extraction" / (d["document_id"] + ".json")).exists()
    ]
    if limit:
        pending = pending[:limit]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for start in range(0, len(pending), 2):
            list(pool.map(extract_one, pending[start : start + 2]))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("stage", choices=["prepare", "translate", "extract"])
    args.add_argument("--limit", type=int, default=0)
    a = args.parse_args()
    {
        "prepare": prepare,
        "translate": lambda: translate(a.limit),
        "extract": lambda: extract(a.limit),
    }[a.stage]()
