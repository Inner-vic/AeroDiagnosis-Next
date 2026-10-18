"""Read-only independent range, budget and graph-path audit of frozen holdout outputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evaluate_strict_graph_holdout import BASE, BASE_INDEX, BASE_SHA, EXP, OLD
from prepare_strict_graph_holdout import OUT, ROOT, digest, read, save
from tokenizers import Tokenizer

from aerodiagnosis.ports import GraphNode, VectorChunk
from aerodiagnosis.retrieval.graph_paths import entity_identity


def units_covered(q, hits):
    units = set()
    for u in q["support_units"]:
        quote = u["quote"]
        wanted = {u["start"] + i for i, ch in enumerate(quote) if not ch.isspace()}
        got = set()
        for h in hits:
            if h["source_key"] == u["source_key"]:
                got |= set(range(h["start"], h["end"]))
        if wanted <= got:
            units.add(u["unit_id"])
    return units


def main():
    freeze = read(ROOT / "evaluation-freeze.json")
    assert all(digest(Path(p)) == h for p, h in freeze["hashes"].items())
    assert digest(BASE_INDEX) == BASE_SHA and digest(OUT / "index.sqlite") == freeze["index_sha256"]
    corpus = read(ROOT / "corpus-final.json")
    chunks = {c["chunk_id"]: VectorChunk(**c) for c in corpus["chunks"]}
    questions = {
        q["sample_id"]: q
        for q in map(
            json.loads, (ROOT / "questions-final.jsonl").read_text(encoding="utf8").splitlines()
        )
    }
    rows = list(
        map(json.loads, (EXP / "predictions.jsonl").read_text(encoding="utf8").splitlines())
    )
    assert len(rows) == 4 * len(questions)
    assert len({(r["sample_id"], r["method"]) for r in rows}) == len(rows)
    baseline = {r["sample_id"]: r for r in rows if r["method"] == "text_baseline"}
    graphs = [read(p / "source-graph-final.json") for p in [BASE, OLD, ROOT]]
    edges = {e["edge_id"]: e for g in graphs for e in g["edges"]}
    nodes = {n["node_id"]: GraphNode(**n) for g in graphs for n in g["nodes"]}
    tokenizer = Tokenizer.from_file(".runtime/models/multilingual-e5-small/tokenizer.json")
    tokenizer.no_padding()
    tokenizer.no_truncation()
    costs = {
        k: len(tokenizer.encode(c.metadata.get("display_content_zh", c.content)).ids)
        for k, c in chunks.items()
    }

    def proof(p):
        c = chunks[p["chunk_id"]]
        co = c.metadata["locator"]["coordinates"]
        lo, hi = p["canonical_start"], p["canonical_end"]
        assert co["start"] <= lo < hi <= co["end"]
        assert c.content[lo - co["start"] : hi - co["start"]] == p["evidence_text"]
        assert co["canonical_sha256"] == p["canonical_sha256"]
        return c

    path_proofs = 0
    cases = []
    novel_units = []
    for r in rows:
        q = questions[r["sample_id"]]
        base = baseline[r["sample_id"]]
        hs = r["hits"]
        assert set(r["method"] for r in rows if r["sample_id"] == q["sample_id"]) == {
            "text_baseline",
            "strict_gate",
            "source_gate",
            "no_eviction_gate",
        }
        assert len(hs) <= 10 and len({h["chunk_id"] for h in hs}) == len(hs)
        assert sum(costs[h["chunk_id"]] for h in hs) <= 4096
        assert max(Counter(chunks[h["chunk_id"]].document_id for h in hs).values(), default=0) <= 5
        assert [(h["chunk_id"], h["content"], h["score"]) for h in hs[:5]] == [
            (h["chunk_id"], h["content"], h["score"]) for h in base["hits"][:5]
        ]
        if r["method"] == "no_eviction_gate":
            assert [(h["chunk_id"], h["content"], h["score"]) for h in hs[: len(base["hits"])]] == [
                (h["chunk_id"], h["content"], h["score"]) for h in base["hits"]
            ]
        for h in hs:
            c = chunks[h["chunk_id"]]
            co = c.metadata["locator"]["coordinates"]
            assert (h["source_key"], h["start"], h["end"], h["content"]) == (
                c.document_id + ":" + c.version_id,
                co["start"],
                co["end"],
                c.content,
            )
        covered = units_covered(q, hs)
        bc = units_covered(q, base["hits"])
        assert (
            sorted(bc - covered) == r["lost_gold_units"]
            and sorted(covered - bc) == r["new_gold_units"]
        )
        for k in [5, 10]:
            current = units_covered(q, hs[:k])
            complete = any(set(s) <= current for s in q["acceptable_support_sets"])
            assert complete == r["complete" + str(k)]
            assert (
                abs(
                    max(len(set(s) & current) / len(set(s)) for s in q["acceptable_support_sets"])
                    - r["coverage" + str(k)]
                )
                < 1e-12
            )
        rank = next(
            (
                k
                for k in range(1, len(hs) + 1)
                if any(set(s) <= units_covered(q, hs[:k]) for s in q["acceptable_support_sets"])
            ),
            None,
        )
        assert abs(r["mrr10"] - (1 / rank if rank else 0)) < 1e-12
        final_ids = {h["chunk_id"] for h in hs}
        for h in hs:
            p = h.get("graph_supplement")
            if not p:
                continue
            path = p["path"]
            assert path["direction"] == "forward" and 1 <= len(path["edge_ids"]) <= 3
            assert set(path["source_chunks"]) <= final_ids
            scope = r["trace"]["source_scope"]
            agreement = Counter(
                (chunks[h["chunk_id"]].document_id, chunks[h["chunk_id"]].version_id)
                for h in base["hits"][:3]
            )
            assert agreement[tuple(scope)] >= 2
            previous = None
            for eid in path["edge_ids"]:
                e = edges[eid]
                props = e["properties"]
                assert e["version_id"] == scope[1] and e["source_ref"] == "document:" + scope[0]
                assert (
                    props["independent_source_review"]
                    in {"accepted", "revise", "revised", "qualified"}
                    and props["polarity"] == "positive"
                )
                if r["method"] == "strict_gate":
                    assert props["certainty"] in {"established", "observed"}
                for pr in [props, *props.get("supporting_evidence", [])]:
                    c = proof(pr)
                    assert [c.document_id, c.version_id] == scope
                ids = []
                for nid in [e["source_id"], e["target_id"]]:
                    node = nodes[nid]
                    link = node.properties["entity_link"]
                    proof(link)
                    ident = entity_identity(node, chunks)
                    assert ident.startswith("entity:")
                    ids.append(ident)
                if previous is not None:
                    assert previous == ids[0]
                previous = ids[1]
            path_proofs += 1
        if r["method"] == "strict_gate":
            if r["complete10"] != base["complete10"]:
                cases.append(
                    {
                        "sample_id": q["sample_id"],
                        "cluster": q["cluster"],
                        "kind": q["kind"],
                        "baseline": base["complete10"],
                        "strict": r["complete10"],
                        "query": q["query"],
                    }
                )
            if covered - bc:
                novel_units.append(
                    {
                        "sample_id": q["sample_id"],
                        "new_units": sorted(covered - bc),
                        "lost_units": sorted(bc - covered),
                        "strict_chain": q["strict_directed_chain"],
                    }
                )
    correct = sum(r["complete10"] for r in baseline.values())
    losses = sum(
        baseline[r["sample_id"]]["complete10"] and not r["complete10"]
        for r in rows
        if r["method"] == "strict_gate"
    )
    save(
        EXP / "independent-audit.json",
        {
            "questions": len(questions),
            "rows": len(rows),
            "canonical_ranges_and_independent_union_metrics_verified": True,
            "all_top5_prefixes_scores_identical": True,
            "all_context_budgets_verified": True,
            "graph_path_proofs_verified": path_proofs,
            "baseline_complete10": correct,
            "strict_complete10_regressions": losses,
            "conditional_regression_rate": losses / correct if correct else None,
            "changed_completion_cases": cases,
            "novel_necessary_unit_cases": novel_units,
            "input_hashes_unchanged": True,
            "base1935_index_unchanged": True,
            "api_calls": 0,
            "no_future_zero_regression_guarantee": True,
        },
    )
    print(
        "Independent audit verified",
        len(rows),
        "rows;",
        path_proofs,
        "path proofs;",
        len(cases),
        "changed cases",
        flush=True,
    )


if __name__ == "__main__":
    main()
