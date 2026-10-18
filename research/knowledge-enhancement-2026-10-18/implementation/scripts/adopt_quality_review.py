"""Adopt independent source-only v3 quality decisions before question authoring."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from evaluate_quality_holdout import digest, read, save, snapshot

from aerodiagnosis.adapters.persistence import create_embedding_provider
from aerodiagnosis.adapters.persistence.neo4j_graph import Neo4jGraphStore
from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.application.evidence_semantics import translation_issues
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.ports import GraphEdge


def main():
    root = Path.cwd()
    dataset = root / "evaluation/gas_path_holdout_v3_candidate"
    if (dataset / "queries.jsonl").exists() or (dataset / "freeze.json").exists():
        raise ValueError("Source decisions must precede v3 authoring and predictions")
    prior = root / ".runtime/experiments/holdout-v3-source-audit-20261014"
    out = root / ".runtime/experiments/holdout-v3-final-20261014"
    out.mkdir(exist_ok=True)
    data, audit = read(prior / "snapshot.json"), read(prior / "quality-audit.json")
    packet = read(root / ".runtime/holdout-v3/source-quality-review-packet.json")
    review = read(root / ".runtime/holdout-v3/source-quality-independent-review.json")
    decisions = review["items"]
    jobs = {r["id"]: r for r in packet["items"]}
    if len(decisions) != len(jobs) or {r["id"] for r in decisions} != set(jobs):
        raise ValueError("Incomplete independent source review")
    if audit["snapshot_sha256"] != digest(data):
        raise ValueError("Original quality snapshot changed")
    settings = RuntimeSettings.from_env(root / ".runtime/holdout-v3/profile")
    backup = out / "before-source-quality.sqlite"
    if not backup.exists():
        with sqlite3.connect(settings.database_path) as src, sqlite3.connect(backup) as dst:
            src.backup(dst)
    vector = SQLiteVectorStore(
        settings.database_path, embedding_provider=create_embedding_provider(settings)
    )
    lexical = SQLiteLexicalStore(settings.database_path)
    cs = {c["chunk_id"]: c for c in data["chunks"]}
    es = {e["edge_id"]: e for e in data["edges"]}
    chunks, edges, records = [], [], []
    for r in decisions:
        if r["status"] not in {"pass", "minor", "major"} or not r["reason"]:
            raise ValueError("Invalid independent decision")
        job = jobs[r["id"]]
        action = "retained"
        if job["category"] == "translation":
            revised = r.get("revised_chinese")
            if r["status"] == "major" and not isinstance(revised, str):
                raise ValueError("Major translation requires complete Chinese repair")
            if isinstance(revised, str):
                original = cs[r["id"]]
                if len(revised) < 20:
                    raise ValueError("Full translation required")
                context = (
                    original["metadata"]
                    .get("locator", {})
                    .get("coordinates", {})
                    .get("source_semantic_context_before", "")
                )
                if translation_issues(original["content"], revised, context):
                    raise ValueError("Repair fails source semantics: " + r["id"])
                chunk = vector.get_chunk(r["id"])
                chunks.append(
                    replace(
                        chunk,
                        metadata={
                            **chunk.metadata,
                            "display_content_zh": revised,
                            "quality_review_status": "independent_source_reviewed_before_questions",
                            "quality_review_sha256": digest(review),
                            "quality_review_reason": r["reason"],
                            "human_expert_reviewed": False,
                        },
                    )
                )
                action = "full_chinese_revised"
        elif r["status"] == "major":
            edge = GraphEdge(**es[r["id"]])
            edges.append(
                replace(
                    edge,
                    properties={
                        **edge.properties,
                        "deprecated": True,
                        "quality_review_status": "rejected_source_semantics",
                        "quality_review_reason": r["reason"],
                        "quality_review_sha256": digest(review),
                    },
                )
            )
            action = "assertion_quarantined"
        records.append(
            {
                "id": r["id"],
                "category": job["category"],
                "status": r["status"],
                "action": action,
                "reason": r["reason"],
            }
        )
    vector.upsert(chunks)
    lexical.upsert(chunks)
    graph = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )
    graph.upsert_edges(edges)
    if settings.external_store_mode != "external-primary":
        SQLiteGraphStore(settings.database_path).upsert_edges(edges)
    graph.close()
    final, graph = snapshot(settings, out)
    save(
        out / "quality-adjudication.json",
        {
            "snapshot_sha256": digest(final),
            "initial_audit_sha256": digest(audit),
            "independent_source_review_sha256": digest(review),
            "translation_changes": len(chunks),
            "quarantined_assertions": len(edges),
            "unresolved_major": 0,
            "human_expert_reviewed": False,
            "gold_access": False,
            "questions_seen": False,
            "system_predictions": 0,
            "decisions": records,
            "policy": "source-only-independent-ai-adjudication-before-questions@1",
        },
    )
    graph.close()
    print("Source-only Chinese revisions", len(chunks), "quarantined assertions", len(edges))


if __name__ == "__main__":
    main()
