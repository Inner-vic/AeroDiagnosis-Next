"""Source-scoped paired holdout: audit, freeze, then generate without gold access."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese prompts deliberately use Chinese punctuation.
import argparse
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluate_chinese_graph_ablation import (
    DisabledGraph,
    NoCases,
    checked_review,
    citation_checks,
    digest,
    pack_context,
)
from prebuild_holdout_knowledge import RetryJSONClient
from transformers import AutoTokenizer

from aerodiagnosis.adapters.llm.research_client import CompletionTruncatedError
from aerodiagnosis.adapters.persistence.neo4j_graph import Neo4jGraphStore
from aerodiagnosis.adapters.persistence.sqlite_context import SQLiteParentContextStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.domain import DiagnosisCommand, RetrievalStrategy
from aerodiagnosis.ingestion import DocumentManifest
from aerodiagnosis.retrieval.manual import ManualEvidenceRetriever, ManualRetrievalPolicy
from aerodiagnosis.retrieval.reranker import LexicalReranker
from aerodiagnosis.tools.diagnostic import DiagnosticToolset

JUDGE_MODEL = "deepseek-v4-pro"

GEN_INSTRUCTION = (
    "只根据evidence回答question，中文输出。无法确定的原因保留不确定性，错误前提应纠正。"
    "资料不足时明确说明，不能用模型记忆补全。每个可核查事实需引文。历史事件不变成现行维修授权。"
    '返回 {"answer":"中文答案","insufficient_evidence":true或false,'
    '"claims":[{"text":"事实","evidence_ids":["提供的id"],'
    '"support_quote":"evidence.text内精确连续片段"}]}。'
)
JUDGE_INSTRUCTION = (
    "按参考必要事实和各组提供的证据审核中文回答，不知道X/Y的检索方法。"
    "事实必须有证据，因果方向、可能性、型号范围、历史限制和否定不能错误。"
    "补充细节不必须覆盖，但所有补充陈述不得与证据矛盾或臆造。"
    "引用字符串检查由独立程序负责。不能因文风和篇幅而偏好。"
    '返回 {"X":{"correct":true或false,"fact_coverage":0到1,'
    '"unsupported_claim":true或false,"reason":"中文具体说明"},"Y":{同字段}}。'
)


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


class StageClient(RetryJSONClient):
    def __init__(self, settings: RuntimeSettings, stage: str, *, model: str | None = None) -> None:
        assert settings.default_llm_api_key is not None and settings.default_llm_model is not None
        super().__init__(
            api_key=settings.default_llm_api_key.get_secret_value(),
            runtime=settings.knowledge_budget_runtime,
            model=model or settings.default_llm_model,
            cap_cny=settings.knowledge_api_cap_cny,
        )
        self.stage = stage

    def complete(self, **kwargs: Any) -> Any:
        kwargs.setdefault("stage", self.stage)
        try:
            return super().complete(**kwargs)
        except CompletionTruncatedError:
            if kwargs.get("max_tokens", 2048) >= 8192:
                raise
            kwargs["max_tokens"] = min(kwargs.get("max_tokens", 2048) * 2, 8192)
            return super().complete(**kwargs)


def snapshot(settings: RuntimeSettings, out: Path) -> tuple[dict[str, Any], Neo4jGraphStore]:
    manifest = DocumentManifest(settings.database_path)
    versions = sorted(manifest.active_version_ids())
    graph = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )
    conn = sqlite3.connect("file:" + settings.database_path.as_posix() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    docs = [
        dict(r)
        for r in conn.execute("select document_id,active_version_id from documents")
        if r["active_version_id"] in versions
    ]
    chunks = [
        dict(r)
        for r in conn.execute(
            "select chunk_id,document_id,version_id,content,metadata_json from vector_chunks"
        )
        if r["version_id"] in versions
    ]
    conn.close()
    if len(docs) != 28:
        raise ValueError("All 28 source documents must be active before audit or freeze")
    for c in chunks:
        c["metadata"] = json.loads(c.pop("metadata_json"))
    nodes = {}
    edges = {}
    for v in versions:
        ns, es = graph.snapshot(active_version_ids=frozenset([v]), limit=2000)
        if len(ns) >= 2000:
            raise ValueError("Per-document graph snapshot may be truncated")
        nodes.update({n.node_id: asdict(n) for n in ns})
        edges.update({e.edge_id: asdict(e) for e in es})
    data = {
        "active_versions": versions,
        "documents": sorted(docs, key=lambda x: x["document_id"]),
        "chunks": sorted(chunks, key=lambda x: x["chunk_id"]),
        "nodes": sorted(nodes.values(), key=lambda x: x["node_id"]),
        "edges": sorted(edges.values(), key=lambda x: x["edge_id"]),
    }
    p = out / "snapshot.json"
    if p.exists() and digest(read(p)) != digest(data):
        raise ValueError("Knowledge snapshot changed; use new experiment directory")
    save(p, data)
    return data, graph


def audit(data, client, out, workers=3):
    by_chunk = {c["chunk_id"]: c for c in data["chunks"]}
    by_node = {n["node_id"]: n for n in data["nodes"]}
    checks = []
    for e in data["edges"]:
        c = by_chunk.get(e["properties"].get("chunk_id"))
        quote = e["properties"].get("evidence_text", "")
        p = e["properties"]
        coordinate = c["metadata"].get("locator", {}).get("coordinates", {}) if c else {}
        start = p.get("canonical_start", 0) - coordinate.get("start", 0)
        checks.append(
            {
                "id": e["edge_id"],
                "endpoints_valid": e["source_id"] in by_node and e["target_id"] in by_node,
                "source_chunk_valid": bool(c and c["version_id"] == e["version_id"]),
                "quote_in_chunk": bool(c and quote and quote in c["content"]),
                "exact_offset_valid": bool(
                    c and quote and c["content"][start : start + len(quote)] == quote
                ),
                "translation_present": bool(p.get("evidence_text_zh")),
            }
        )
    jobs = []
    translations = []
    instruction = (
        "审核英文资料到中文的技术翻译，或中文原文的保存质量。仅依据original。重点检查否定、可能性、条件、数值单位、对象和步骤是否漏"
        '译或强化。不要因保留型号/缩写而报错。返回 {"items":[{"id":"原ID","status":"pass|minor|'
        'major","issues":[{"original_quote":"原文精确片段","chinese_quote":"对应译文'
        '片段","reason":"中文原因","suggested_chinese":"建议修改"}]}]}。major=改变技术含义、'
        "漏关键条件或无法理解；minor=格式或用词不自然但含义可辨；无问题issues为空。不要总结替代逐条审核。"
    )
    for i in range(0, len(data["chunks"]), 3):
        batch = [
            {
                "id": c["chunk_id"],
                "original": c["content"],
                "chinese": c["metadata"].get("display_content_zh", ""),
            }
            for c in data["chunks"][i : i + 3]
        ]
        jobs.append(("translation", "zh-quality-translation-v1", instruction, batch))
    relations = []
    instruction = (
        "依据source_original审核中文三元组，实体缩短后的语义和中文source_chinese。检查实体类型、方向、关系含义"
        "、否定、可能性和条件。CAUSES为原因到结果，CAUSED_BY为结果到原因；CHECKED_BY、MITIGATED_BY等按"
        "通常含义，不把一般训练资料变成普适维修授权。qualifiers承载条件，短实体不用重复所有条件。source_original唯"
        '一依据。返回 {"items":[{"id":"原ID","status":"pass|minor|major","source_'
        'entity_status":"pass|minor|major","target_entity_status":"pass|mi'
        'nor|major","relation_status":"pass|minor|major","issues":[{"origi'
        'nal_quote":"原文精确片段","reason":"中文说明","suggested_change":"中文具体修改"}]'
        "}]}。major表示错误技术含义、关系方向错误、无原文支持、漏掉影响结论的条件；minor为含义仍可辨的小问题。"
    )
    for i in range(0, len(data["edges"]), 4):
        batch = []
        for e in data["edges"][i : i + 4]:
            p = e["properties"]
            batch.append(
                {
                    "id": e["edge_id"],
                    "source_original": p.get("evidence_text"),
                    "source_chinese": p.get("evidence_text_zh"),
                    "source_entity": {
                        k: by_node[e["source_id"]][k] for k in ["name", "kind", "description"]
                    },
                    "target_entity": {
                        k: by_node[e["target_id"]][k] for k in ["name", "kind", "description"]
                    },
                    "relation": e["relation"],
                    "certainty": p.get("certainty"),
                    "polarity": p.get("polarity"),
                    "qualifiers": p.get("qualifiers"),
                }
            )
        jobs.append(("relation", "zh-quality-relations-v1", instruction, batch))

    def review(job):
        kind, task, instruction, batch = job
        result = checked_review(client, task, instruction, batch, out)
        print("Source-only audit", kind, len(batch), flush=True)
        return kind, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for kind, result_rows in pool.map(review, jobs):
            (translations if kind == "translation" else relations).extend(result_rows)
    for rows in [translations, relations]:
        if any(x["status"] not in ["pass", "minor", "major"] for x in rows):
            raise ValueError("Invalid review status")
    result = {
        "snapshot_sha256": digest(data),
        "review_basis": (
            "same-provider AI source review; not expert gold; flags require adjudication"
        ),
        "structural_checks": checks,
        "translations": translations,
        "entity_relations": relations,
        "translation_counts": dict(Counter(x["status"] for x in translations)),
        "relation_counts": dict(Counter(x["relation_status"] for x in relations)),
        "entity_counts": dict(
            Counter(
                s for x in relations for s in [x["source_entity_status"], x["target_entity_status"]]
            )
        ),
        "structural_pass": {k: sum(x[k] for x in checks) for k in checks[0] if k != "id"},
    }
    save(out / "quality-audit.json", result)


def policy(settings: RuntimeSettings) -> ManualRetrievalPolicy:
    return ManualRetrievalPolicy(
        strategy=settings.manual_retrieval_strategy,
        candidate_k=settings.manual_candidate_k,
        rrf_k=settings.manual_rrf_k,
        max_per_document=settings.manual_max_per_document,
        expand_context=settings.manual_expand_context,
    )


def resolve_scope(question: str, documents: list[dict[str, Any]]) -> tuple[str, frozenset[str]]:
    match = re.match(r"^依据资料「([^」]+)」[：:]\s*(.+)$", question)
    if not match:
        raise ValueError("Expected explicit user-supplied source reference")
    source, body = match.groups()
    target = "holdout-v1-" + source
    versions = frozenset(d["active_version_id"] for d in documents if d["document_id"] == target)
    if not versions:
        raise ValueError("Referenced source is unavailable")
    return body, versions


def configuration(settings: RuntimeSettings) -> dict[str, Any]:
    return {
        "manual_policy": asdict(policy(settings)),
        "requested_model": settings.default_llm_model,
        "judge_model": JUDGE_MODEL,
        "tokenizer_path": str(settings.knowledge_tokenizer_path.resolve()),
        "source_guided_graph": True,
        "embedding_backend": settings.embedding_backend,
        "embedding_dimensions": settings.embedding_dimensions,
        "embedding_model": settings.embedding_model,
        "vector_backend": settings.vector_backend,
        "graph_backend": settings.graph_backend,
        "retrieval_strategy": "weighted_rrf",
    }


def citation_gate(arm: dict[str, Any]) -> bool:
    checks = arm["citation_exact_checks"]
    answer = arm["answer"]
    # Empty citations only make sense for an explicit abstention; the blind judge
    # still verifies that the abstention is appropriate against gold.
    return bool(all(checks) and (checks or answer.get("insufficient_evidence") is True))


def freeze(
    root: Path, dataset: Path, settings: RuntimeSettings, data: dict[str, Any], out: Path
) -> None:
    if not (out / "quality-adjudication.json").exists():
        raise ValueError("Complete source-only quality adjudication before freeze")
    adjudication = read(out / "quality-adjudication.json")
    if adjudication["snapshot_sha256"] != digest(data) or adjudication["unresolved_major"] != 0:
        raise ValueError("Unresolved source quality issues")
    qs = rows(dataset / "queries.jsonl")
    reconciliation = read(dataset / "reconciliation.json")
    if len(qs) != 120 or reconciliation["ai_source_accepted"] != 120:
        raise ValueError("Review incomplete")
    receipt = read(root / ".runtime/holdout-v1/application/prebuild-receipt.json")
    if receipt["status"] != "complete":
        raise ValueError("Ingestion incomplete")
    file_names = [
        "queries.jsonl",
        "generation_queries.jsonl",
        "source_manifest.json",
        "reconciliation.json",
        "protocol.json",
    ]
    code_files = [
        *(root / "src/aerodiagnosis").rglob("*.py"),
        Path(__file__).resolve(),
        root / "scripts/prepare_holdout_knowledge.py",
        root / "scripts/prebuild_holdout_knowledge.py",
        root / "scripts/evaluate_chinese_graph_ablation.py",
        root / "scripts/check_holdout_preparation.py",
        root / "scripts/adjudicate_holdout_quality.py",
        root / "scripts/summarize_holdout_comparison.py",
        root / "scripts/review_holdout_corrections.py",
        root / "scripts/finalize_holdout_quality.py",
    ]
    record = {
        "date": "2026-10-14",
        "status": "frozen_before_system_predictions",
        "queries_sha256": hashlib.sha256(
            (dataset / "queries.jsonl").read_text(encoding="utf-8").encode()
        ).hexdigest(),
        "file_sha256_bytes": {
            f: hashlib.sha256((dataset / f).read_bytes()).hexdigest() for f in file_names
        },
        "snapshot_sha256": digest(data),
        "code_sha256_bytes": {
            str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in code_files
        },
        "knowledge_coverage_verified": True,
        "unresolved_reviews": 0,
        "quality_adjudication_sha256": digest(adjudication),
        "configuration": configuration(settings),
        "protocol": {
            "query_mode": "natural_chinese_only",
            "source_scope": (
                "explicit source reference parsed from question only; identical in both arms"
            ),
            "top_k": 5,
            "evidence_token_budget": 2048,
            "manual_policy": asdict(policy(settings)),
            "ranking_policy": "grounded_source_rank@2",
            "case_route": False,
            "requested_model": settings.default_llm_model,
            "judge_model": JUDGE_MODEL,
            "temperature": 0,
            "known_billed_truncation_retry": "one retry with double max tokens in each arm",
            "transport_unknown_charge_retry": False,
            "generation_instruction_sha256": hashlib.sha256(GEN_INSTRUCTION.encode()).hexdigest(),
            "judge_instruction_sha256": hashlib.sha256(JUDGE_INSTRUCTION.encode()).hexdigest(),
            "primary_subset": "97 source-text questions; 23 summary auxiliary separately",
            "interval_method": (
                "paired source-cluster bootstrap; conservative NASA correlation-group "
                "sensitivity; conditional to curated sources"
            ),
            "bootstrap_repetitions": 10000,
            "bootstrap_seed": 20261014,
            "primary_effect": "difference in strictly correct answers (graph minus text)",
            "test_tuning_prohibited": True,
            "not_actual_repair_diagnostic_accuracy": True,
        },
    }
    p = dataset / "freeze.json"
    if p.exists() and read(p) != record:
        raise ValueError("Frozen protocol differs")
    save(p, record)
    save(out / "freeze.json", record)
    print("Frozen", len(qs), "questions; no predictions yet", flush=True)


def check_freeze(
    root: Path, dataset: Path, data: dict[str, Any], settings: RuntimeSettings
) -> dict[str, Any]:
    frozen = read(dataset / "freeze.json")
    if frozen["configuration"] != configuration(settings):
        raise ValueError("Frozen runtime configuration changed")
    if frozen["snapshot_sha256"] != digest(data):
        raise ValueError("Frozen knowledge changed")
    for path, sha in frozen["file_sha256_bytes"].items():
        if hashlib.sha256((dataset / path).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen dataset changed")
    for path, sha in frozen["code_sha256_bytes"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen implementation changed")
    return frozen


def compare(
    root: Path,
    dataset: Path,
    settings: RuntimeSettings,
    data: dict[str, Any],
    graph: Neo4jGraphStore,
    out: Path,
    workers: int,
) -> None:
    frozen = check_freeze(root, dataset, data, settings)
    # The generation input contains no answers, evidence ranges, task type, or answerability.
    inputs = rows(dataset / "generation_queries.jsonl")
    if any(set(x) != {"sample_id", "query", "query_language"} for x in inputs):
        raise ValueError("Generation metadata leakage")
    tokenizer = AutoTokenizer.from_pretrained(
        str(settings.knowledge_tokenizer_path), local_files_only=True
    )
    manifest = DocumentManifest(settings.database_path)

    def metadata(v: str) -> dict[str, Any] | None:
        rec = manifest.get_version(v)
        return rec.metadata.model_dump(mode="json") if rec else None

    retriever = ManualEvidenceRetriever(
        vector_store=SQLiteVectorStore(settings.database_path),
        lexical_store=SQLiteLexicalStore(settings.database_path),
        context_store=SQLiteParentContextStore(settings.database_path),
        policy=policy(settings),
        reranker=LexicalReranker(),
    )
    tools = {
        name: DiagnosticToolset(
            vector_store=retriever,
            graph_store=g,
            case_store=NoCases(),
            version_metadata=metadata,
            source_guided_graph=True,
        )
        for name, g in [("text", DisabledGraph()), ("text_graph", graph)]
    }
    client = StageClient(settings, "generation_evaluation")
    answers = out / "answers"
    answers.mkdir(exist_ok=True)

    def answer(q: dict[str, Any]) -> dict[str, Any]:
        p = answers / (q["sample_id"] + ".json")
        if p.exists():
            old = read(p)
            if old["snapshot_sha256"] != digest(data) or old["query"] != q["query"]:
                raise ValueError("Stale answer cache")
            return old
        query, versions = resolve_scope(q["query"], data["documents"])
        cmd = DiagnosisCommand(session_id=q["sample_id"], question=q["query"], top_k=5)
        arms = {}
        for name in ["text", "text_graph"]:
            start = time.perf_counter()
            result = tools[name].hybrid_search(
                command=cmd,
                query=query,
                active_version_ids=versions,
                strategy=RetrievalStrategy.WEIGHTED_RRF,
            )
            elapsed = time.perf_counter() - start
            evidence = [h.evidence for h in result.hits]
            packed, count = pack_context(evidence, tokenizer, budget=2048)
            reply = client.complete(
                task="holdout-answer-v1",
                instruction=GEN_INSTRUCTION,
                payload={"question": q["query"], "evidence": packed},
                max_tokens=1600,
            )
            checks = citation_checks(reply.value.get("claims"), packed)
            arms[name] = {
                "answer": reply.value,
                "packed": packed,
                "retrieved": [x.model_dump(mode="json") for x in evidence],
                "context_tokens": count,
                "citation_exact_checks": checks,
                "retrieval_seconds": elapsed,
                "receipt": reply.receipt,
            }
        row = {
            "id": q["sample_id"],
            "query": q["query"],
            "snapshot_sha256": digest(data),
            "arms": arms,
        }
        save(p, row)
        print("Generated paired", q["sample_id"], flush=True)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(answer, inputs))
    # Gold is read only after paired generation finishes.
    gold = {x["sample_id"]: x for x in rows(dataset / "queries.jsonl")}
    judge_client = StageClient(settings, "generation_evaluation", model=JUDGE_MODEL)
    graded = out / "graded"
    graded.mkdir(exist_ok=True)

    def judge(index_result: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, result = index_result
        p = graded / (result["id"] + ".json")
        if p.exists():
            return read(p)
        q = gold[result["id"]]
        order = ["text", "text_graph"] if index % 2 == 0 else ["text_graph", "text"]
        arms = result["arms"]
        response = judge_client.complete(
            task="holdout-judge-v1",
            instruction=JUDGE_INSTRUCTION,
            payload={
                "question": result["query"],
                "minimal_facts": q["minimal_facts"],
                "reference_answer": q["reference_answer"],
                "answerability": q["answerability"],
                "X": {"answer": arms[order[0]]["answer"], "evidence": arms[order[0]]["packed"]},
                "Y": {"answer": arms[order[1]]["answer"], "evidence": arms[order[1]]["packed"]},
            },
            max_tokens=1600,
        )
        if set(response.value) != {"X", "Y"}:
            raise ValueError("Invalid anonymous judge output")
        scores = {}
        for letter, name in zip(["X", "Y"], order, strict=True):
            score = response.value[letter]
            if not isinstance(score.get("correct"), bool) or not isinstance(
                score.get("unsupported_claim"), bool
            ):
                raise ValueError("Invalid judge schema")
            score["strict_correct"] = (
                score["correct"] and not score["unsupported_claim"] and citation_gate(arms[name])
            )
            scores[name] = score
        row = {
            **result,
            "judge": scores,
            "source_layer": q["source_layer"],
            "stratum": q["stratum"],
            "source_cluster": q["statistical_cluster"],
            "correlation_group": q["correlation_group"],
            "task_type": q["task_type"],
            "judge_receipt": response.receipt,
        }
        save(p, row)
        print("Graded paired", result["id"], flush=True)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(judge, enumerate(results)))
    save(
        out / "comparison-complete.json",
        {
            "questions": len(inputs),
            "snapshot_sha256": digest(data),
            "freeze_sha256": digest(frozen),
            "budget": {k: v for k, v in client.summary().items() if k != "receipts"},
            "review_basis": (
                "same-provider separate-model blinded AI judge; source adjudication required"
            ),
        },
    )


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--phase", choices=["audit", "freeze", "compare"], required=True)
    cli.add_argument("--workers", type=int, default=3, choices=[1, 2, 3, 4])
    cli.add_argument(
        "--output", type=Path, default=Path(".runtime/experiments/holdout-v1-paired-20261014")
    )
    args = cli.parse_args()
    root = Path.cwd().resolve()
    dataset = root / "evaluation/gas_path_holdout_v1_candidate"
    settings = RuntimeSettings.from_env(root / ".runtime/holdout-v1/profile")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if settings.knowledge_api_cap_cny != 20:
        raise ValueError("Explicit cumulative cap 20 required")
    data, graph = snapshot(settings, out)
    if args.phase == "audit":
        audit(data, StageClient(settings, "supplemental_review"), out, args.workers)
    elif args.phase == "freeze":
        freeze(root, dataset, settings, data, out)
    else:
        compare(root, dataset, settings, data, graph, out, args.workers)


if __name__ == "__main__":
    main()
