"""Read-only Chinese quality audit and paired graph ablation on a frozen live snapshot."""

from __future__ import annotations

# ruff: noqa: RUF001 -- These prompts deliberately use Chinese punctuation.
import argparse
import hashlib
import json
import sqlite3
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from transformers import AutoTokenizer

from aerodiagnosis.adapters.llm.research_client import ResearchClient
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


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


class NoCases:
    backend_name = "disabled_for_ablation"

    def search(self, query, *, limit=20):
        return []


class DisabledGraph:
    """The text arm must never contact Neo4j, even through hybrid route collection."""

    backend_name = "graph_disabled_for_ablation"

    def snapshot(self, **kwargs):
        return [], []

    def search_nodes(self, keyword, **kwargs):
        return []

    def neighbors(self, node_id, **kwargs):
        return []


def snapshot(root, settings, out):
    conn = sqlite3.connect("file:" + settings.database_path.as_posix() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    active = [
        r[0] for r in conn.execute("SELECT version_id FROM document_versions WHERE status='active'")
    ]
    chunks = [
        dict(r)
        for r in conn.execute(
            "SELECT chunk_id,document_id,version_id,content,metadata_json FROM vector_chunks"
        )
        if r["version_id"] in active
    ]
    conn.close()
    for c in chunks:
        c["metadata"] = json.loads(c.pop("metadata_json"))
    graph = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )
    nodes, edges = graph.snapshot(active_version_ids=frozenset(active), limit=2000)
    data = {
        "active_versions": sorted(active),
        "chunks": chunks,
        "nodes": [asdict(x) for x in nodes],
        "edges": [asdict(x) for x in edges],
    }
    if (out / "snapshot.json").exists():
        if digest(read(out / "snapshot.json")) != digest(data):
            raise ValueError("Live knowledge changed; never mix snapshots in this run")
    else:
        save(out / "snapshot.json", data)
    return data, graph


def checked_review(client, task, instruction, items, out):
    key = digest(items)
    target = out / (task + "-" + key + ".json")
    if target.exists():
        return read(target)["items"]
    result = client.complete(
        task=task, instruction=instruction, payload={"items": items}, max_tokens=4096
    ).value
    rows = result.get("items")
    if (
        not isinstance(rows, list)
        or len(rows) != len(items)
        or {x.get("id") for x in rows} != {x["id"] for x in items}
    ):
        raise ValueError("Review response omitted or changed IDs")
    save(target, result)
    return rows


def audit(data, client, out):
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
        translations.extend(
            checked_review(client, "zh-quality-translation-v1", instruction, batch, out)
        )
        print("translation audit", min(i + 3, len(data["chunks"])), flush=True)
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
        relations.extend(checked_review(client, "zh-quality-relations-v1", instruction, batch, out))
        print("entity/relation audit", min(i + 4, len(data["edges"])), flush=True)
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


def pack_context(items, tokenizer, budget=2048):
    """Never cross the evidence token budget or give one arm extra context."""
    packed = []
    used = 0
    for evidence in items:
        text = evidence.locator.get("display_text_zh") or evidence.excerpt
        entry = {"id": evidence.evidence_id, "text": text}
        metadata = getattr(evidence, "document_metadata", {})
        if metadata:
            entry["source"] = {
                k: metadata.get(k) for k in ["source_url", "source_revision", "source_type"]
            }

        size = len(
            tokenizer.encode(
                json.dumps(entry, ensure_ascii=False), add_special_tokens=False, verbose=False
            )
        )
        if used + size > budget:
            break
        packed.append(entry)
        used += size
    return packed, used


def citation_checks(claims, packed):
    """Unknown IDs and unsupported exact quotations fail rather than disappear."""
    if not isinstance(claims, list):
        raise ValueError("Answer claims must be a list")
    texts = {item["id"]: item["text"] for item in packed}
    results = []
    for claim in claims:
        refs = claim.get("evidence_ids") if isinstance(claim, dict) else None
        quote = claim.get("support_quote") if isinstance(claim, dict) else None
        results.append(
            bool(
                isinstance(refs, list)
                and refs
                and all(isinstance(ref, str) and ref in texts for ref in refs)
                and isinstance(quote, str)
                and quote
                and any(quote in texts[ref] for ref in refs)
            )
        )
    return results


def compare(root, settings, data, graph, client, out, query_mode="natural", quality_policy=False):
    if not (out / "quality-audit.json").exists():
        raise ValueError("Complete quality audit before ablation")
    tokenizer = AutoTokenizer.from_pretrained(
        root / ".runtime/models/multilingual-e5-small", local_files_only=True
    )
    policy = ManualRetrievalPolicy(
        strategy=settings.manual_retrieval_strategy,
        candidate_k=settings.manual_candidate_k,
        rrf_k=settings.manual_rrf_k,
        max_per_document=settings.manual_max_per_document,
        expand_context=settings.manual_expand_context,
    )
    retriever = ManualEvidenceRetriever(
        vector_store=SQLiteVectorStore(settings.database_path),
        lexical_store=SQLiteLexicalStore(settings.database_path),
        context_store=SQLiteParentContextStore(settings.database_path),
        policy=policy,
        reranker=LexicalReranker(),
    )
    manifest = DocumentManifest(settings.database_path)

    def metadata(version):
        r = manifest.get_version(version)
        return r.metadata.model_dump(mode="json") if r else None

    tools = DiagnosticToolset(
        vector_store=retriever,
        graph_store=graph,
        case_store=NoCases(),
        version_metadata=metadata,
        source_guided_graph=quality_policy,
    )
    text_tools = DiagnosticToolset(
        vector_store=retriever,
        graph_store=DisabledGraph(),
        case_store=NoCases(),
        version_metadata=metadata,
        source_guided_graph=quality_policy,
    )
    queries = lines(root / "evaluation/gas_path_pilot_v1/queries.jsonl") + lines(
        root / "evaluation/gas_path_benchmark_v2_supplement/tasks.jsonl"
    )
    # Human-readable Chinese wording frozen before retrieval; translated from questions only.
    if (out / "queries-zh.json").exists():
        translations = read(out / "queries-zh.json")
    else:
        translations = []
        for i in range(0, len(queries), 10):
            batch = [{"id": q["sample_id"], "question": q["query"]} for q in queries[i : i + 10]]
            translations.extend(
                checked_review(
                    client,
                    "zh-evaluation-query-v1",
                    (
                        "仅将question忠实译成中文。不要回答，不增加提示，不改写问题前提。另从中文问题提取2至5个原问题中已有的短检索关键词，"
                        "不用答"
                        '案知识。返回 {"items":[{"id":"原ID","query_zh":"中文问题'
                        '","keywords":["关键词"]'
                        "}]}。"
                    ),
                    batch,
                    out,
                )
            )
        save(out / "queries-zh.json", translations)
    zh = {x["id"]: x for x in translations}
    catalog = read(root / ".runtime/fusion/source-catalog/catalog.json")
    derivations = {d["document_id"]: d["derivation"] for d in catalog["documents"]}

    def pages(e):
        doc = e.document_id.removeprefix("document:")
        coords = e.locator.get("coordinates", {})
        if e.source_kind == "knowledge_graph_path":
            props = coords.get("properties", {})
            coords = props.get("source_locator", {}).get("coordinates", coords)
        ds = derivations.get(doc)
        ps = coords.get("pages", [])
        if ds:
            return ds["original_source_id"], [
                ds["derived_to_original_page"].get(str(p), p) for p in ps
            ]
        if doc == "fusion-faa-ch10-page45":
            return "faa-amtp-ch10", [45]
        return doc, ps

    protocol = {
        "snapshot_sha256": digest(data),
        "date": "2026-10-14",
        "dataset": "25 used dev + 5 supplemental; never formal test",
        "top_k": 5,
        "context_token_budget": 2048,
        "tokenizer": "local multilingual-e5-small",
        "model": client.model,
        "temperature": 0,
        "quality_policy": quality_policy,
        "manual_policy": asdict(policy),
        "fusion": (
            "grounded_source_rank@2 in BOTH arms; 0.65 lexical + 0.25 manual rank + "
            "0.02 lexical graph support; no forced route coverage"
            if quality_policy
            else ("same weighted_rrf@2 in BOTH arms; text arm graph disabled; "
                  "cases disabled; graph weight 0.88")
        ),
        "query": query_mode + "; keywords derived from question only, never reference",
        "generation": "identical prompt; no reference answer in retrieval or generation",
        "judge": (
            "same-model source/reference review, blinded X/Y balanced determin"
            "istic order; not expert; no significance claim"
        ),
    }
    target = out / "protocol.json"
    if target.exists() and read(target) != protocol:
        raise ValueError("Protocol changed")
    save(target, protocol)
    results = []
    gen_instruction = (
        "只根据evidence回答question，中文输出。资料不足时明确说无法据此确定，不用模型记忆补全。不将培训示例视作任何型号的授"
        '权。每个事实需引文。返回 {"answer":"中文答案","insufficient_evidence":true或false,'
        '"claims":[{"text":"事实","evidence_ids":["提供的id"],"support_quote":"'
        'evidence.text内精确连续片段"}]}。'
    )
    for index, q in enumerate(queries):
        file = out / (q["sample_id"] + ".json")
        if file.exists():
            results.append(read(file))
            continue
        cmd = DiagnosisCommand(
            session_id="graph-ablation-" + q["sample_id"],
            question=zh[q["sample_id"]]["query_zh"],
            top_k=5,
        )
        arms = {}
        sensitivity = {}
        for name, strategy in [
            ("text", RetrievalStrategy.WEIGHTED_RRF),
            ("text_graph", RetrievalStrategy.WEIGHTED_RRF),
        ]:
            start = time.perf_counter()
            selected_tools = text_tools if name == "text" else tools
            retrieval_query = (
                cmd.question
                if query_mode == "natural"
                else " ".join(zh[q["sample_id"]]["keywords"])
            )
            retrieved = selected_tools.hybrid_search(
                command=cmd,
                query=retrieval_query,
                active_version_ids=frozenset(data["active_versions"]),
                strategy=strategy,
            )
            elapsed = time.perf_counter() - start
            evidence = [h.evidence for h in retrieved.hits]
            packed, tokens = pack_context(evidence, tokenizer)
            output = client.complete(
                task="zh-graph-ablation-answer-v1",
                instruction=gen_instruction,
                payload={"question": cmd.question, "evidence": packed},
                max_tokens=1600,
            )
            ids = {e["id"] for e in packed}
            checks = citation_checks(output.value.get("claims"), packed)
            arms[name] = {
                "retrieved": [e.model_dump(mode="json") for e in evidence],
                "packed": packed,
                "context_tokens": tokens,
                "retrieval_seconds": elapsed,
                "answer": output.value,
                "citation_exact_checks": checks,
                "included_source_pages": [pages(e) for e in evidence if e.evidence_id in ids],
                "receipt": output.receipt,
                "routes": [x.model_dump(mode="json") for x in retrieved.routes],
            }
            kr = selected_tools.hybrid_search(
                command=cmd,
                query=" ".join(zh[q["sample_id"]]["keywords"]),
                active_version_ids=frozenset(data["active_versions"]),
                strategy=strategy,
            )
            sensitivity[name] = {
                "graph_hits": sum(
                    (
                        h.evidence.source_kind == "knowledge_graph_path"
                        or "graph_guidance" in h.evidence.locator
                    )
                    for h in kr.hits
                ),
                "evidence_ids": [h.evidence.evidence_id for h in kr.hits],
            }
        gold = [
            (e["document_id"], e.get("source_locator", {}).get("page"))
            for e in q.get("evidence", [])
        ]
        available = {
            (d["original_source_id"], p)
            for d in derivations.values()
            for p in d["original_physical_pages"]
        }
        coverage = bool(gold and all(pair in available for pair in gold))
        if not gold:
            coverage = q["sample_id"] in ["gp-q028", "gp-q029"]
        for arm in arms.values():
            offered = {(doc, p) for doc, ps in arm["included_source_pages"] for p in ps}
            arm["all_gold_source_pages_hit"] = (
                all(pair in offered for pair in gold) if gold else None
            )
        order = ["text", "text_graph"] if index % 2 == 0 else ["text_graph", "text"]
        judged = client.complete(
            task="zh-graph-ablation-judge-v1",
            instruction=(
                "依据reference_answer与minimal_facts判断答案是否覆盖必要事实；根据每组提供的evidence检查答案是"
                "否有无依据陈述、错误条件、错误否定或臆造数值。不要因为流畅、篇幅长而偏好。X/Y是匿名顺序，不知道检索方法。引用准确性由独立精确检"
                "查负责。missing_source=true时正确行为是拒答/说明无法确定，即使碰巧答案符合reference也不能算正确。an"
                'swerability非answerable时应纠正错误前提或拒绝编造。返回 {"X":{"correct":true或'
                "false"
                ',"fact_coverage":0到1,"unsupported_claim":true或false,"reason":"中文具'
                '体解释"},"Y":{同字段}}。仅输出客观判断，不给总排名。'
            ),
            payload={
                "question": cmd.question,
                "reference_answer": q.get("reference_answer"),
                "minimal_facts": q.get("minimal_facts"),
                "answerability": q.get("answerability"),
                "missing_source": not coverage,
                "X": {"answer": arms[order[0]]["answer"], "evidence": arms[order[0]]["packed"]},
                "Y": {"answer": arms[order[1]]["answer"], "evidence": arms[order[1]]["packed"]},
            },
            max_tokens=1600,
        )
        if set(judged.value) != {"X", "Y"}:
            raise ValueError("Invalid judge arms")
        result = {
            "id": q["sample_id"],
            "query_zh": cmd.question,
            "answerability": q.get("answerability"),
            "source_covered": coverage,
            "gold_pages": gold,
            "arms": arms,
            "keyword_sensitivity": sensitivity,
            "judge": {order[0]: judged.value["X"], order[1]: judged.value["Y"]},
        }
        save(file, result)
        results.append(result)
        print(
            "paired comparison",
            index + 1,
            "/30",
            q["sample_id"],
            "graph",
            sum(
                (x["source_kind"] == "knowledge_graph_path" or "graph_guidance" in x["locator"])
                for x in arms["text_graph"]["retrieved"]
            ),
            flush=True,
        )
    primary = [r for r in results if r["source_covered"]]
    summary = {
        "primary_n": len(primary),
        "missing_source_n": len(results) - len(primary),
        "arms": {},
        "snapshot_sha256": digest(data),
    }
    for name in ["text", "text_graph"]:
        claims = [b for r in primary for b in r["arms"][name]["citation_exact_checks"]]
        pg = [r for r in primary if r["gold_pages"]]
        summary["arms"][name] = {
            "correct": sum(r["judge"][name]["correct"] for r in primary),
            "mean_fact_coverage": sum(r["judge"][name]["fact_coverage"] for r in primary)
            / len(primary),
            "unsupported_answer_count": sum(r["judge"][name]["unsupported_claim"] for r in primary),
            "exact_citation_pass": sum(claims),
            "citation_claim_count": len(claims),
            "all_gold_source_pages_hit": sum(
                r["arms"][name]["all_gold_source_pages_hit"] for r in pg
            ),
            "page_metric_n": len(pg),
            "graph_returned_questions": sum(
                any(
                    (x["source_kind"] == "knowledge_graph_path" or "graph_guidance" in x["locator"])
                    for x in r["arms"][name]["retrieved"]
                )
                for r in primary
            ),
            "keyword_graph_returned_questions": sum(
                r["keyword_sensitivity"][name]["graph_hits"] > 0 for r in primary
            ),
            "mean_retrieval_seconds": sum(r["arms"][name]["retrieval_seconds"] for r in primary)
            / len(primary),
        }
    summary["paired"] = {
        "graph_wins": sum(
            r["judge"]["text_graph"]["correct"] and not r["judge"]["text"]["correct"]
            for r in primary
        ),
        "graph_losses": sum(
            r["judge"]["text"]["correct"] and not r["judge"]["text_graph"]["correct"]
            for r in primary
        ),
        "same_evidence_questions": sum(
            r["arms"]["text"]["packed"] == r["arms"]["text_graph"]["packed"] for r in primary
        ),
    }
    summary["budget"] = {k: v for k, v in client.summary().items() if k != "receipts"}
    save(out / "comparison-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["audit", "compare"], required=True)
    parser.add_argument("--query-mode", choices=["natural", "keywords"], default="natural")
    parser.add_argument("--quality-policy", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(".runtime/experiments/zh-graph-audit-20261014")
    )
    args = parser.parse_args()
    root = Path.cwd()
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    settings = RuntimeSettings.from_env(root / ".runtime/fusion/profile")
    client = ResearchClient(
        api_key=settings.default_llm_api_key.get_secret_value(),
        runtime=settings.knowledge_budget_runtime,
        model=settings.default_llm_model,
        cap_cny=5,
        timeout=60,
    )
    data, graph = snapshot(root, settings, out)
    if args.phase == "audit":
        audit(data, client, out)
    else:
        compare(root, settings, data, graph, client, out, args.query_mode, args.quality_policy)


if __name__ == "__main__":
    main()
