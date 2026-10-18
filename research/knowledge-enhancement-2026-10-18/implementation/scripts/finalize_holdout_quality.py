"""Freeze-ready source quality constraints and conservative final adjudication."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese source-quality notices.
import sqlite3
from dataclasses import replace
from pathlib import Path

from evaluate_gas_path_holdout import digest, read, save, snapshot

from aerodiagnosis.adapters.persistence.neo4j_graph import Neo4jGraphStore
from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.ports import GraphEdge


def main():
    root = Path.cwd().resolve()
    if (root / "evaluation/gas_path_holdout_v1_candidate/freeze.json").exists():
        raise ValueError("No post-prediction quality changes")
    prior = root / ".runtime/experiments/holdout-v1-20261014"
    out = root / ".runtime/experiments/holdout-v1-paired-20261014"
    out.mkdir(parents=True, exist_ok=True)
    data = read(prior / "snapshot.json")
    confirmation = read(prior / "second-model-source-confirmation.json")
    if confirmation["snapshot_sha256"] != digest(data):
        raise ValueError("Review snapshot changed")
    unknown_trans = [
        r
        for r in confirmation["translations"]
        if r["status"] == "major" and not r["source_corruption"]
    ]
    majors = [r for r in confirmation["graph"] if r["status"] == "major"]
    if unknown_trans or len(majors) != 1 or not majors[0]["id"].startswith("d50eeb418d"):
        raise ValueError("Unadjudicated second-model findings")
    affected = {
        c["document_id"]
        for c in data["chunks"]
        if c["chunk_id"].startswith(
            ("49aa0a831e", "49c36da3bf", "50db299118", "814775b0b6", "e29df304d2")
        )
    }
    notice = (
        "【来源质量提示：该资料的部分公式、上下标或图示未完整解析。"
        "请仅使用可读文字；不能据此补写或推算缺损公式。】"
    )
    plan = {
        "before_snapshot_sha256": digest(data),
        "second_model_review_sha256": digest(confirmation),
        "expert_reviewed": False,
        "gold_access": False,
        "deprecated_edge": majors[0]["id"],
        "reason": "affecting并不等于独立的直接CAUSES；保守退出图谱检索，来源正文仍保留。",
        "source_warning_documents": sorted(affected),
        "notice": notice,
    }
    save(out / "supplemental-source-adjudication.json", plan)
    backup = out / "before-supplement.sqlite"
    settings = RuntimeSettings.from_env(root / ".runtime/holdout-v1/profile")
    if not backup.exists():
        with sqlite3.connect(settings.database_path) as src, sqlite3.connect(backup) as dst:
            src.backup(dst)
    vector = SQLiteVectorStore(settings.database_path)
    lexical = SQLiteLexicalStore(settings.database_path)
    changed = []
    for c in data["chunks"]:
        if c["document_id"] not in affected:
            continue
        chunk = vector.get_chunk(c["chunk_id"])
        text = c["metadata"]["display_content_zh"]
        changed.append(
            replace(
                chunk,
                metadata={
                    **chunk.metadata,
                    "display_content_zh": notice + "\n" + text,
                    "source_integrity_warning": notice,
                    "source_integrity_adjudication_sha256": digest(plan),
                },
            )
        )
    vector.upsert(changed)
    lexical.upsert(changed)
    graph = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )
    raw = next(e for e in data["edges"] if e["edge_id"] == majors[0]["id"])
    edge = GraphEdge(**raw)
    updated = replace(
        edge,
        properties={
            **edge.properties,
            "deprecated": True,
            "quality_review_status": "rejected_source_semantics",
            "supplemental_adjudication_sha256": digest(plan),
        },
    )
    graph.upsert_edges([updated])
    SQLiteGraphStore(settings.database_path).upsert_edges([updated])
    final, _ = snapshot(settings, out)
    checks = []
    chunks = {c["chunk_id"]: c for c in final["chunks"]}
    for e in final["edges"]:
        p = e["properties"]
        c = chunks[p["chunk_id"]]
        quote = p["evidence_text"]
        start = p["canonical_start"] - c["metadata"]["locator"]["coordinates"]["start"]
        checks.append(
            c["content"][start : start + len(quote)] == quote and c["version_id"] == e["version_id"]
        )
    if not all(checks):
        raise ValueError("Source evidence changed")
    record = {
        **read(prior / "quality-adjudication.json"),
        "snapshot_sha256": digest(final),
        "second_model_review_sha256": digest(confirmation),
        "supplemental_plan_sha256": digest(plan),
        "graph_assertions_quarantined": 43,
        "graph_assertions_repaired": 8,
        "graph_active_positive_assertions": sum(
            e["properties"].get("deprecated") is not True
            and e["properties"].get("polarity") != "negative"
            for e in final["edges"]
        ),
        "source_warning_documents": sorted(affected),
        "warning_chunks": len(changed),
        "known_source_corruption_major": 3,
        "source_constraint": (
            "damaged formulas/figures are unavailable; readable qualitative "
            "source text remains available, with explicit notice in generation evidence"
        ),
        "post_structural_checks_pass": sum(checks),
        "post_structural_checks_total": len(checks),
        "unresolved_major": 0,
    }
    save(out / "quality-adjudication.json", record)
    print(record)


if __name__ == "__main__":
    main()
