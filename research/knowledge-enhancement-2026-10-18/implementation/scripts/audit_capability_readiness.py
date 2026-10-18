"""Read-only source-topology and label-readiness audit; never runs retrieval or API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    graph = Path(".runtime/experiments/live-chinese-graph-dev-20261015/graph-snapshot.json")
    questions = Path("evaluation/text_hybrid_holdout_v5_candidate/queries.jsonl")
    d = json.loads(graph.read_text(encoding="utf-8"))
    nodes = {n["node_id"]: n for n in d["nodes"]}
    adjacent = defaultdict(set)
    outgoing = defaultdict(list)
    degree = Counter()
    for e in d["edges"]:
        a, b = e["source_id"], e["target_id"]
        assert a in nodes and b in nodes
        adjacent[a].add(b)
        adjacent[b].add(a)
        degree[a] += 1
        degree[b] += 1
        outgoing[a].append(e)
    sizes = []
    seen = set()
    for node in nodes:
        if node in seen:
            continue
        queue = [node]
        size = 0
        while queue:
            current = queue.pop()
            if current in seen:
                continue
            seen.add(current)
            size += 1
            queue.extend(adjacent[current] - seen)
        sizes.append(size)
    two_hops = sum(
        1
        for e in d["edges"]
        for f in outgoing[e["target_id"]]
        if f["target_id"] != e["source_id"] and f["version_id"] == e["version_id"]
    )
    qs = [json.loads(line) for line in questions.read_text(encoding="utf-8").splitlines()]
    result = {
        "scope": "current application active snapshot; no production mutation",
        "snapshot_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
        "nodes": len(nodes),
        "edges": len(d["edges"]),
        "node_degree_distribution": dict(Counter(degree[n] for n in nodes)),
        "connected_component_size_distribution": dict(Counter(sizes)),
        "directed_nonbacktracking_two_hop_paths": two_hops,
        "nodes_with_explicit_role": sum(
            n["properties"].get("role", "unknown") != "unknown" for n in nodes.values()
        ),
        "interpretation": (
            "Raw assertion graph is disconnected pairs; same-name concept grouping "
            "is not proof of physical identity or a reasoning chain"
        ),
        "v5_questions": len(qs),
        "v5_necessary_fact_count_distribution": dict(Counter(len(q["required_facts"]) for q in qs)),
        "v5_questions_with_explicit_support_paths": sum(bool(q.get("support_paths")) for q in qs),
        "metric_limit": (
            "Graph structural readiness and annotation shape, not retrieval or answer accuracy"
        ),
        "api_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
