"""Independent source/metric audit for the six protected development strategies."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path("evaluation/retrieval_capability_v1_candidate")
OLD = Path(".runtime/experiments/graph-gating-dev60-20261016")
OUT = Path(".runtime/experiments/graph-gating-dev60-20261016-final")


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    protocol = read(OUT / "protocol.json")
    maintenance = read(OUT / "source-maintenance.json")
    for p, h in protocol["hashes"].items():
        if p in maintenance:
            note = maintenance[p]
            assert digest(Path(note["registered_source"])) == h == note["registered_sha256"]
            assert digest(Path(p)) == note["current_sha256"]
            assert ast.dump(ast.parse(Path(p).read_text(encoding="utf8"))) == ast.dump(
                ast.parse(Path(note["registered_source"]).read_text(encoding="utf8"))
            )
        else:
            assert digest(Path(p)) == h
    first = read(OLD / "protocol.json")
    for p, h in first["hashes"].items():
        current = Path(p)
        relative = current.relative_to(Path.cwd()) if current.is_absolute() else current
        saved = OLD / "registered-source-snapshots" / relative
        assert digest(saved if saved.exists() else current) == h, p
    # Compare every recorded value, including source bundles, scores and gate traces.
    previous = [
        json.loads(s) for s in (OLD / "predictions.jsonl").read_text(encoding="utf8").splitlines()
    ]
    rows = [
        json.loads(s) for s in (OUT / "predictions.jsonl").read_text(encoding="utf8").splitlines()
    ]
    assert rows == previous
    questions = {
        q["sample_id"]: q
        for q in map(
            json.loads, (ROOT / "questions-final.jsonl").read_text(encoding="utf8").splitlines()
        )
    }
    corpus = read(ROOT / "corpus.json")
    chunks = {c["chunk_id"]: c for c in corpus["chunks"]}
    graph = read(ROOT / "source-graph-final.json")
    edges = {e["edge_id"]: e for e in graph["edges"]}
    nodes = {n["node_id"]: n for n in graph["nodes"]}
    baseline = {r["sample_id"]: r for r in rows if r["method"] == "text_baseline"}
    tokenizer = Tokenizer.from_file(".runtime/models/multilingual-e5-small/tokenizer.json")
    tokenizer.no_padding()
    tokenizer.no_truncation()
    costs = {
        k: len(tokenizer.encode(c["metadata"].get("display_content_zh", c["content"])).ids)
        for k, c in chunks.items()
    }

    def proof(p):
        c = chunks[p["chunk_id"]]
        co = c["metadata"]["locator"]["coordinates"]
        a, b = p["canonical_start"], p["canonical_end"]
        start = co["start"]
        assert start <= a < b <= start + len(c["content"])
        assert c["content"][a - start : b - start] == p["evidence_text"]
        assert co["canonical_sha256"] == p["canonical_sha256"]
        return c

    seen = set()
    path_proofs = 0
    new_cases = []
    for row in rows:
        identity = (row["method"], row["sample_id"])
        assert identity not in seen
        seen.add(identity)
        base = baseline[row["sample_id"]]
        q = questions[row["sample_id"]]
        hs = row["hits"]
        assert hs[:5] == base["hits"][:5]
        assert len(hs) <= 10 and len({h["chunk_id"] for h in hs}) == len(hs)
        assert sum(costs[h["chunk_id"]] for h in hs) <= 4096
        assert (
            max(Counter(chunks[h["chunk_id"]]["document_id"] for h in hs).values(), default=0) <= 5
        )
        for h in hs:
            c = chunks[h["chunk_id"]]
            co = c["metadata"]["locator"]["coordinates"]
            assert (h["source_key"], h["start"], h["end"], h["content"]) == (
                c["document_id"] + ":" + c["version_id"],
                co["start"],
                co["end"],
                c["content"],
            )
            if h["graph_supplement"]:
                scope = tuple(row["trace"]["source_scope"])
                assert (c["document_id"], c["version_id"]) == scope
                path = h["graph_supplement"]["path"]
                es = [edges[k] for k in path["edge_ids"]]
                assert path["direction"] == "forward" and path["hops"] == len(es)
                assert path["certainty"] == [e["properties"]["certainty"] for e in es]
                assert path["qualifiers"] == [e["properties"].get("qualifiers", {}) for e in es]
                assert set(path["source_chunks"]) <= {x["chunk_id"] for x in hs}
                for e in es:
                    p = e["properties"]
                    assert e["version_id"] == scope[1] and e["source_ref"] == "document:" + scope[0]
                    assert p["polarity"] == "positive" and p["independent_source_review"] in {
                        "accepted",
                        "revise",
                        "revised",
                        "qualified",
                    }
                    assert p["validation_status"] == "necessary_source_checks_passed"
                    for pp in [p, *p.get("supporting_evidence", [])]:
                        c2 = proof(pp)
                        assert (c2["document_id"], c2["version_id"]) == scope
                        path_proofs += 1
                    for node in [nodes[e["source_id"]], nodes[e["target_id"]]]:
                        np = node["properties"]["entity_link"]
                        nc = proof(np)
                        assert np["review_status"] == "source_verified_nonexpert" and np[
                            "role"
                        ] not in {"", "unknown"}
                        assert np["surface"].casefold() in np["evidence_text"].casefold()
                        assert (
                            node["version_id"] == scope[1]
                            and node["source_ref"] == "document:" + scope[0]
                        )
                        assert np["document_id"] == nc["document_id"] and np["kind"] == node["kind"]
                    if row["method"] in {"strict_gate", "no_eviction_gate"}:
                        assert p["certainty"] in {"established", "observed"}

        # An independent per-character source-range union, not the runtime/scorer masks.
        def complete(k, q=q, hs=hs):
            covered = set()
            for u in q["support_units"]:
                spans = [
                    (h["start"], h["end"]) for h in hs[:k] if h["source_key"] == u["source_key"]
                ]
                if all(
                    any(a <= u["start"] + i < b for a, b in spans)
                    for i, ch in enumerate(u["quote"])
                    if not ch.isspace()
                ):
                    covered.add(u["unit_id"])
            return any(set(s) <= covered for s in q["acceptable_support_sets"])

        assert complete(5) == bool(row["complete5"]) and complete(10) == bool(row["complete10"])
        rank = next((k for k in range(1, len(hs) + 1) if complete(k)), None)
        assert abs(row["mrr10"] - (1 / rank if rank else 0)) < 1e-12
        if row["method"] == "strict_gate" and row["complete10"] and not base["complete10"]:
            new_cases.append(
                dict(
                    sample_id=row["sample_id"],
                    query=q["query"],
                    kind=q["kind"],
                    strict_directed_chain=q["strict_directed_chain"],
                    new_chunks=row["trace"]["added"],
                )
            )
    summary = read(OUT / "summary.json")
    for name, stats in summary.items():
        rs = [r for r in rows if r["method"] == name]
        assert stats["ALL"]["complete10"] == sum(r["complete10"] for r in rs)
        assert stats["ALL"]["complete5"] == sum(r["complete5"] for r in rs)
    selection = read(OUT / "selection.json")
    assert selection["selected"] == "strict_gate"
    receipt = dict(
        rows_verified=len(rows),
        all_360_rows_equal_before_after_provenance_metadata_hardening=True,
        all_final_freeze_hashes_valid=True,
        original_registered_code_retained=True,
        all_prefixes_scores_and_source_ranges_valid=True,
        all_atomic_graph_bundles_complete=True,
        all_metrics_independent_position_union_recomputed=True,
        graph_evidence_checks=path_proofs,
        strict_gain_cases=new_cases,
        api_calls=0,
    )
    (OUT / "audit.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
