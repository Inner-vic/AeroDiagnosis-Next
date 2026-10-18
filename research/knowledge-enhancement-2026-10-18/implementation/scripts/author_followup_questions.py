"""Source-only atomic question authoring; no retrieval, graph, predictions or old gold."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese annotation instructions use Chinese punctuation.
import hashlib
import json
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluate_gas_path_holdout import StageClient

from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.domain.atomic_annotation import AtomicAnnotation

INSTRUCTION = """仅依据提供的完整saved_source编写5道不同的中文知识利用题。禁止外部知识。
题目在本次保存正文的范围内，不代表当前维修授权。不要询问缺损公式、图或没有的数字。
5题尽量涵盖：明确机理/原因、来源明示的两步后果、适用条件/对象、尚未确定的原因或可能性、排除某原因或反例。
若来源不支持其中一种题型，用其他不同事实替代，不强行凑事实。不使用统一的泛泛模板。
每题设1–3个题面明确询问的question_slots，每个required_fact只对应1个slot，一个主语、一个谓词、一个宾语；
把题面没问的背景/后果/数字放supplementary_facts，而不是required_facts。事实用中文，型号缩写允许。
source_quote必须连续精确复制saved_source的原文（包含不确定性/否定和决定含义的条件），不可省略号拼接。
找不到源文支持不能编题；本轮优先问来源已经明确回答的“不确定/未知”，而非凭缺少某词推断无答案。
返回JSON {"questions":[{"question":"中文题面（不加来源前缀）","task_type":
"mechanism|chain|condition|uncertainty|counterexample",
"question_slots":[{"slot_id":"s1","requirement":"题面对应的必答范围"}],
"required_facts":[{"slot_id":"s1","subject":"单个主语","predicate":"单个谓词","object":"单个宾语",
"text":"完整原子中文事实","certainty":"finding|possible|undetermined","scope":"对象/事件/条件",
"source_quote":"连续原文精确片段"}],
"supplementary_facts":[]}]}。
每个fact中若有条件必需保留在scope/text；possible必须保留可能，undetermined必须保留未确定。"""


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def locate_quote(text, quote):
    start = text.find(quote)
    if start >= 0:
        return start, quote, "exact"
    normalized = []
    offsets = []
    for index, character in enumerate(text):
        for value in unicodedata.normalize("NFKC", character):
            if not value.isspace():
                normalized.append(value)
                offsets.append(index)
    needle = "".join(c for c in unicodedata.normalize("NFKC", quote) if not c.isspace())
    if len(needle) < 40:
        raise ValueError("Short nonexact source quote")
    found = "".join(normalized).find(needle)
    if found < 0:
        raise ValueError("Quote not exact: " + quote[:80])
    start, end = offsets[found], offsets[found + len(needle) - 1] + 1
    return start, text[start:end], "NFKC_and_whitespace_only_program_resolved"


def main():
    root = Path.cwd()
    dataset = root / "evaluation/gas_path_holdout_v2_candidate"
    if (dataset / "freeze.json").exists():
        raise ValueError("Frozen test annotations cannot be regenerated")
    prepared = root / ".runtime/holdout-v2/prepared"
    catalog = json.loads((prepared / "catalog.json").read_text(encoding="utf-8"))
    manifest = json.loads((dataset / "source_manifest.json").read_text(encoding="utf-8"))
    sources = {d["document_id"]: d for d in manifest["documents"]}
    out = dataset / "authoring"
    out.mkdir(exist_ok=True)
    settings = RuntimeSettings.from_env(root / ".runtime/holdout-v2/profile")
    client = StageClient(settings, "supplemental_review")

    def author(row):
        doc = row["original_source_id"]
        target = out / (doc + ".json")
        if target.exists():
            return json.loads(target.read_text(encoding="utf-8"))
        text = json.loads((prepared / row["parsed_path"]).read_text(encoding="utf-8"))[
            "canonical_content"
        ]
        payload = {
            "source_id": doc,
            "source_scope": sources[doc]["completeness"],
            "saved_source": text,
        }
        reply = client.complete(
            task="atomic-holdout-author-v2",
            instruction=INSTRUCTION,
            payload=payload,
            max_tokens=6000,
        )
        value = reply.value
        for attempt in range(2):
            try:
                if len(value["questions"]) != 5:
                    raise ValueError("Exactly five distinct questions required")
                compiled = []
                for n, question in enumerate(value["questions"]):
                    for category in ["required_facts", "supplementary_facts"]:
                        question.setdefault(category, [])
                        for i, fact in enumerate(question[category]):
                            quote = fact["source_quote"]
                            start, quote, _normalization = locate_quote(text, quote)
                            fact["source_quote"] = quote
                            fact["source_start"], fact["source_end"] = start, start + len(quote)
                            fact["source_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                            fact["fact_key"] = f"{doc}-q{n + 1}-{category}-{i + 1}"
                            if category == "supplementary_facts":
                                fact["slot_id"] = "supplementary"
                    annotation = AtomicAnnotation.model_validate(
                        {
                            k: question[k]
                            for k in ["question_slots", "required_facts", "supplementary_facts"]
                        }
                    )
                    annotation.verify_source(text)
                    compiled.append({**question, "annotation": annotation.model_dump(mode="json")})
                record = {
                    "document_id": doc,
                    "questions": compiled,
                    "receipt": reply.receipt,
                    "status": "author_source_checked_pending_independent_review",
                    "expert_reviewed": False,
                }
                save(target, record)
                print("Authored", doc, flush=True)
                return record
            except (ValueError, KeyError, TypeError) as exc:
                if attempt:
                    raise
                reply = client.complete(
                    task="atomic-holdout-author-repair-v2",
                    instruction=INSTRUCTION
                    + " supplementary_facts必须为空数组[]；不要输出字符串数组。",
                    payload={**payload, "invalid_output": value, "error": str(exc)},
                    max_tokens=6000,
                )
                value = reply.value

    with ThreadPoolExecutor(max_workers=3) as pool:
        authored = list(pool.map(author, catalog["documents"]))
    rows = []
    for doc in authored:
        source = sources[doc["document_id"]]
        for q in doc["questions"]:
            identity = f"followup-q{len(rows) + 1:03d}"
            annotation = q["annotation"]
            rows.append(
                {
                    "sample_id": identity,
                    "query": f"依据资料「{doc['document_id']}」：{q['question']}",
                    "query_language": "zh",
                    "task_type": q["task_type"],
                    **annotation,
                    "reference_answer": " ".join(f["text"] for f in annotation["required_facts"]),
                    "minimal_facts": annotation["required_facts"],
                    "answerability": "answerable",
                    "document_id": doc["document_id"],
                    "statistical_cluster": source["family_id"],
                    "correlation_group": "nasa-engine-monitoring"
                    if source["publisher"] == "NASA"
                    else source["family_id"],
                    "source_layer": source["source_layer"],
                    "stratum": source["source_type"],
                    "review_status": "pending_independent_answer_hidden_ai_review",
                    "human_expert_reviewed": False,
                    "annotation_policy": "question-bound-atomic@1",
                }
            )
    if len(rows) != 120 or len({r["query"] for r in rows}) != 120:
        raise ValueError("120 distinct source-bound tasks required")
    (dataset / "queries.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    (dataset / "generation_queries.jsonl").write_text(
        "".join(
            json.dumps(
                {k: r[k] for k in ["sample_id", "query", "query_language"]}, ensure_ascii=False
            )
            + "\n"
            for r in rows
        ),
        encoding="utf-8",
    )
    print("Authored 120 candidates; no predictions; independent review pending", flush=True)


if __name__ == "__main__":
    main()
