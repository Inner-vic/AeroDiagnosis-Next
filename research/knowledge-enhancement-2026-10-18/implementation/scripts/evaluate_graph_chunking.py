"""Compare conservative KG extraction over a frozen canonical-text snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from aerodiagnosis.application.knowledge_graph_extractor import KnowledgeGraphExtractor
from aerodiagnosis.domain import make_chunk_id
from aerodiagnosis.ingestion.parsers import PlainTextParser


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    extractor = KnowledgeGraphExtractor()
    summaries = []
    for strategy in ("fixed", "structure", "hierarchical"):
        parser = PlainTextParser(
            strategy=strategy, chunk_size=800, overlap=100, parent_budget=3200, child_budget=800
        )
        documents = []
        for source in snapshot["documents"]:
            path = args.snapshot.parent / source["processed_path"]
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != source["sha256"]:
                raise ValueError("snapshot byte hash mismatch")
            parsed = parser.parse(raw)
            version = hashlib.sha256(
                parsed.canonical_content + parsed.parser_version.encode()
            ).hexdigest()
            nodes, edges = extractor.extract(
                source_ref="document:" + source["document_id"],
                version_id=version,
                chunks=tuple(c.content for c in parsed.chunks),
                chunk_ids=tuple(
                    make_chunk_id(version, c.locator.canonical(), c.content) for c in parsed.chunks
                ),
                chunk_locators=tuple(c.locator.canonical() for c in parsed.chunks),
            )
            relations = Counter(e.relation for e in edges)
            documents.append(
                {
                    "document_id": source["document_id"],
                    "cohort": source["cohort"],
                    "chunks": len(parsed.chunks),
                    "nodes": len(nodes),
                    "edges": len(edges),
                    "relations": dict(relations),
                    "diagnostic_edges": sum(
                        v for k, v in relations.items() if k != "HAS_COMPONENT"
                    ),
                }
            )
        summaries.append(
            {
                "strategy": strategy,
                "parser_identity": parsed.parser_version,
                "documents": documents,
                "chunk_count": sum(d["chunks"] for d in documents),
                "edge_count": sum(d["edges"] for d in documents),
                "diagnostic_edges": sum(d["diagnostic_edges"] for d in documents),
            }
        )
    report = {
        "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "extractor_identity": extractor.identity,
        "chunk_size": 800,
        "parent_budget": 3200,
        "child_budget": 800,
        "overlap": 100,
        "strategies": summaries,
        "interpretation": "Coverage observation only; no human-reviewed relation gold. "
        "Zero diagnostic output cannot establish extraction precision.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            [
                {k: s[k] for k in ("strategy", "chunk_count", "edge_count", "diagnostic_edges")}
                for s in summaries
            ]
        )
    )


if __name__ == "__main__":
    main()
