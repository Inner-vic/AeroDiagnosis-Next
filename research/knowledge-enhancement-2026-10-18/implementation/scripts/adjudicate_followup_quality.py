"""Second-model source-only adjudication; keep originals and quarantine bad assertions."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese source review instructions.
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from evaluate_followup_holdout import StageClient, digest, read, save, snapshot

from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.ports import GraphEdge

INSTRUCTION = """仅依据source_original复核技术翻译/断言是否正确，不能读问题、参考答案或预测。
第一轮审核可能误报，必须自己核对原文。字段category=translation时current_chinese是完整中文文本；
graph时检查实体类型、方向、事件/比较角色、certainty、polarity和条件，不把历史动作变成通用要求。
返回 {"items":[{"id":"原ID","status":"pass|minor|major","source_corruption":false,
"reason":"中文来源核对理由","revised_chinese":null}]}。
pass=含义正确；minor=格式等不影响技术含义；major=技术含义/条件/可能性/否定/对象错误。
source_corruption仅用于源文本公式/图示缺损，不能用来豁免可读文字的翻译错误。
若translation为major且source_corruption=false，revised_chinese必须给依据完整source_original重新修正的全文中文；
不得只返回问题片段或总结，型号/数字必须原样保留。graph不需改写，major会保守退出检索。"""


def main():
    root = Path.cwd()
    dataset = root / "evaluation/gas_path_holdout_v2_candidate"
    if (dataset / "freeze.json").exists():
        raise ValueError("Quality changes must precede all predictions")
    prior = root / ".runtime/experiments/holdout-v2-quality-20261014"
    out = root / ".runtime/experiments/holdout-v2-paired-20261014"
    out.mkdir(parents=True, exist_ok=True)
    data = read(prior / "snapshot.json")
    audit = read(prior / "quality-audit.json")
    if audit["snapshot_sha256"] != digest(data):
        raise ValueError("Audit snapshot changed")
    by_chunk = {r["chunk_id"]: r for r in data["chunks"]}
    by_node = {r["node_id"]: r for r in data["nodes"]}
    by_edge = {r["edge_id"]: r for r in data["edges"]}
    jobs = []
    for row in audit["translations"]:
        if row["status"] == "pass":
            continue
        chunk = by_chunk[row["id"]]
        jobs.append(
            {
                "id": row["id"],
                "category": "translation",
                "source_original": chunk["content"],
                "current_chinese": chunk["metadata"]["display_content_zh"],
                "first_review": row,
            }
        )
    for row in audit["entity_relations"]:
        if row["status"] == "pass":
            continue
        edge = by_edge[row["id"]]
        jobs.append(
            {
                "id": row["id"],
                "category": "graph",
                "source_original": edge["properties"]["evidence_text"],
                "current_chinese": edge["properties"].get("evidence_text_zh"),
                "source_entity": {k: by_node[edge["source_id"]][k] for k in ["name", "kind"]},
                "target_entity": {k: by_node[edge["target_id"]][k] for k in ["name", "kind"]},
                "relation": edge["relation"],
                "certainty": edge["properties"].get("certainty"),
                "polarity": edge["properties"].get("polarity"),
                "qualifiers": edge["properties"].get("qualifiers"),
                "first_review": row,
            }
        )
    settings = RuntimeSettings.from_env(root / ".runtime/holdout-v2/profile")
    client = StageClient(settings, "supplemental_review", model="deepseek-v4-pro")
    reviews = prior / "second-model"
    reviews.mkdir(exist_ok=True)

    def review(batch):
        key = digest(batch)
        target = reviews / (key + ".json")
        if target.exists():
            result = read(target)
        else:
            reply = client.complete(
                task="followup-source-adjudication-v2",
                instruction=INSTRUCTION,
                payload={"items": batch},
                max_tokens=6000,
            )
            result = reply.value
            result["receipt"] = reply.receipt
            save(target, result)
        if len(result["items"]) != len(batch) or {r["id"] for r in result["items"]} != {
            r["id"] for r in batch
        }:
            raise ValueError("Incomplete second-model source review")
        for row in result["items"]:
            if (
                row["status"] not in ["pass", "minor", "major"]
                or type(row["source_corruption"]) is not bool
            ):
                raise ValueError("Invalid source review status")
        return result["items"]

    batches = [jobs[i : i + 2] for i in range(0, len(jobs), 2)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        checked = [r for batch in pool.map(review, batches) for r in batch]
    categories = {r["id"]: r["category"] for r in jobs}
    confirmation = {
        "snapshot_sha256": digest(data),
        "reviews": checked,
        "expert_reviewed": False,
        "basis": "second-model source-only AI review; no gold or predictions",
    }
    save(prior / "second-model-source-confirmation.json", confirmation)
    # Preserve database before applying any source-only changes.
    backup = out / "before-quality-adjudication.sqlite"
    if not backup.exists():
        with sqlite3.connect(settings.database_path) as src, sqlite3.connect(backup) as dst:
            src.backup(dst)
    vector = SQLiteVectorStore(settings.database_path)
    lexical = SQLiteLexicalStore(settings.database_path)
    changed = []
    unresolved = []
    for row in checked:
        if categories[row["id"]] != "translation" or row["status"] != "major":
            continue
        chunk = vector.get_chunk(row["id"])
        if chunk is None:
            raise ValueError("Source chunk unavailable")
        if row["source_corruption"]:
            revised = (
                "【来源质量提示：原始资料的部分数学表达或图示文字不完整，仅使用可读定性文字。】\n"
                + chunk.metadata["display_content_zh"]
            )
        else:
            revised = row.get("revised_chinese")
            if not isinstance(revised, str) or len(revised) < 20:
                unresolved.append(row["id"])
                continue
            recheck = client.complete(
                task="followup-translation-repair-check-v2",
                instruction=INSTRUCTION,
                payload={
                    "items": [
                        {
                            "id": row["id"],
                            "category": "translation",
                            "source_original": chunk.content,
                            "current_chinese": revised,
                            "first_review": {"status": "repair_not_yet_checked"},
                        }
                    ]
                },
                max_tokens=1500,
            ).value
            if len(recheck.get("items", [])) != 1 or recheck["items"][0].get("status") not in [
                "pass",
                "minor",
            ]:
                save(prior / ("unresolved-" + row["id"] + ".json"), recheck)
                unresolved.append(row["id"])
                continue
        changed.append(
            replace(
                chunk,
                metadata={
                    **chunk.metadata,
                    "display_content_zh": revised,
                    "quality_review_status": "source_reviewed_before_predictions",
                    "quality_review_sha256": digest(confirmation),
                },
            )
        )
    vector.upsert(changed)
    lexical.upsert(changed)
    graph_updates = []
    for row in checked:
        if categories[row["id"]] == "graph" and row["status"] == "major":
            edge = GraphEdge(**by_edge[row["id"]])
            graph_updates.append(
                replace(
                    edge,
                    properties={
                        **edge.properties,
                        "deprecated": True,
                        "quality_review_status": "rejected_source_semantics",
                        "quality_review_reason": row["reason"],
                        "quality_review_sha256": digest(confirmation),
                    },
                )
            )
    # Open Neo4j from the final snapshot helper without writing old audit snapshot.
    from aerodiagnosis.adapters.persistence.neo4j_graph import Neo4jGraphStore

    graph = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )
    graph.upsert_edges(graph_updates)
    if settings.external_store_mode != "external-primary":
        SQLiteGraphStore(settings.database_path).upsert_edges(graph_updates)
    final, _ = snapshot(settings, out)
    record = {
        "snapshot_sha256": digest(final),
        "initial_audit_sha256": digest(audit),
        "second_review_sha256": digest(confirmation),
        "translation_changes": len(changed),
        "quarantined_assertions": len(graph_updates),
        "unresolved_major": len(unresolved),
        "unresolved_translation_ids": unresolved,
        "human_expert_reviewed": False,
        "gold_access": False,
        "system_predictions": 0,
        "policy": "source-only-audit-and-second-model-conservative-quarantine@1",
    }
    save(out / "quality-adjudication.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if unresolved:
        raise ValueError("Source quality issues need explicit adjudication before freeze")


if __name__ == "__main__":
    main()
