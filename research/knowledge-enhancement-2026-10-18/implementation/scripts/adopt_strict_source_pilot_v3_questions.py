"""Adopt source-reconstructed labels before any rankings; author gold remains unused."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

from author_followup_questions import locate_quote
from capability_metrics import evidence_metrics
from prepare_graph_supplement_v2 import OUT, ROOT, digest, read, save

REVIEW = Path(
    "D:/zhaoweihao3/Desktop/Project/AeroDiagnosis/.runtime/strict-question-review/blind_question_review_v1.json"
)


def adopt(final, review_path=None):
    review = read(review_path or REVIEW)
    assert review["coverage_complete"] if final else True
    assert (
        not review["gold_answers_seen"]
        and not review["predictions_seen"]
        and not review["expert_reviewed"]
    )
    mapping = read(OUT / "blind-question-private-map.json")
    pack = read(OUT / "blind-question-review-pack.json")
    queries = {q["blind_id"]: q["query"] for f in pack["families"] for q in f["questions"]}
    data = read(ROOT / "translated-corpus-final.json")
    chunks = {c["chunk_id"]: c for c in data["chunks"]}
    docs = {d["document_id"]: d for d in data["documents"]}
    sources = {d["source_key"]: d["content"] for d in docs.values()}
    if final:
        assert {i["blind_id"] for i in review["items"]} == set(mapping)
    accepted = []
    rejected = []
    errors = []
    reclassified = []
    for item in review["items"]:
        key = item["blind_id"]
        parent = mapping[key]
        if item["query_quality"] == "reject":
            rejected.append({"blind_id": key, "family": parent["family"], "reason": item["reason"]})
            continue
        try:
            query = item["query_revision"] if item["query_quality"] == "revise" else queries[key]
            assert query.strip() and "sg1-" not in query
            units = []
            ids = []
            align = []
            for j, u in enumerate(item["supports"]):
                assert u["quote"].strip()
                c = chunks[u["chunk_id"]]
                d = docs[c["document_id"]]
                assert d["source_family"] == parent["family"]
                offset, quote, mode = locate_quote(c["content"], u["quote"])
                lo = offset + c["metadata"]["locator"]["coordinates"]["start"]
                assert d["content"][lo : lo + len(quote)] == quote
                units.append(
                    {
                        "unit_id": str(j),
                        "source_key": d["source_key"],
                        "start": lo,
                        "end": lo + len(quote),
                        "quote": quote,
                        "source_sha256": d["source_sha256"],
                        "requirement_zh": u["requirement_zh"],
                    }
                )
                ids.append(c["chunk_id"])
                align.append(mode)
            sets = [[str(v) for v in s] for s in item["sufficient_sets"]]
            assert sets and all(s and set(s) <= {u["unit_id"] for u in units} for s in sets)
            minimum = min(len({ids[int(j)] for j in s}) for s in sets)
            assert minimum == item["minimum_original_chunks"]
            chain = []
            for link in item.get("chain_links", []):
                c = chunks[link["chunk_id"]]
                assert docs[c["document_id"]]["source_family"] == parent["family"]
                off, exact, mode = locate_quote(c["content"], link["quote"])
                chain.append(
                    {
                        **link,
                        "quote": exact,
                        "alignment": mode,
                        "canonical_start": c["metadata"]["locator"]["coordinates"]["start"] + off,
                    }
                )
            if chain:
                assert all(a["to_zh"] == b["from_zh"] for a, b in itertools.pairwise(chain))
            hops = item.get("semantic_hops", 0)
            assert hops == len(chain)
            hidden = item.get("hidden_bridge_zh", [])
            strict = bool(
                item["strict_directed_chain"]
                and minimum >= 2
                and item["minimum_natural_blocks"] >= 2
                and hops >= 2
                and hidden
                and all(h not in query for h in hidden)
                and all(x["certainty"] in {"established", "observed"} for x in chain)
            )
            if item["strict_directed_chain"] and not strict:
                raise ValueError("Claimed strict chain fails registered explicit eligibility")
            planned = parent["planned_kind"]
            kind = planned
            if planned in {"S4_two_support", "S5_three_support", "S7_mechanism"}:
                kind = (
                    ("STRICT_3plus" if hops >= 3 else "STRICT_2")
                    if strict
                    else (
                        "MODEL_DEPENDENCY_CHAIN"
                        if hops >= 2
                        and minimum >= 2
                        and item.get("chain_scope", item.get("scope"))
                        in {"model_dependency", "model_computation"}
                        else "UNCERTAIN_CHAIN"
                        if hops >= 2
                        and minimum >= 2
                        and any(x["certainty"] not in {"established", "observed"} for x in chain)
                        else "NONSTRICT_DIRECTED_CHAIN"
                        if hops >= 2 and minimum >= 2
                        else "S_multiblock_synthesis"
                        if minimum >= 2
                        else "S_singleblock_synthesis"
                    )
                )
            if kind != planned:
                reclassified.append({"blind_id": key, "planned": planned, "actual": kind})
            q = {
                "sample_id": "sp3:" + key,
                "blind_id": key,
                "cluster": parent["family"],
                "kind": kind,
                "planned_kind": planned,
                "query": query,
                "reference_answer": item["answer_zh"],
                "support_units": units,
                "acceptable_support_sets": sets,
                "necessary_original_chunks": minimum,
                "necessary_natural_blocks": item["minimum_natural_blocks"],
                "natural_block_minimality": item.get("minimality_audit_zh", []),
                "single_block_alternative_audit": item.get("single_block_alternative_audit_zh", ""),
                "provided_chunk_ids": ids,
                "strict_directed_chain": strict,
                "semantic_hops": hops,
                "chain_links": chain,
                "hidden_bridge_zh": hidden,
                "evidence_structure": item["evidence_structure"],
                "chain_scope": item.get(
                    "chain_scope", item.get("scope", "source_physical_or_causal")
                ),
                "answerable": True,
                "expert_reviewed": False,
                "annotation_status": "independent_answer_hidden_nonexpert_source_reconstruction",
                "source_review_reason": item["reason"],
                "quote_alignment": align,
                "query_revised_before_predictions": item["query_quality"] == "revise",
            }
            evidence_metrics(
                sources=sources, support_units=units, acceptable_support_sets=sets, hits=[]
            )
            assert any("\u3400" <= ch <= "\u9fff" for ch in q["reference_answer"])
            accepted.append(q)
        except (AssertionError, KeyError, ValueError, TypeError) as e:
            errors.append({"blind_id": key, "reason": type(e).__name__ + ":" + str(e)[:180]})
    audit = {
        "candidate_queries": len(mapping),
        "reviewed": len(review["items"]),
        "accepted": len(accepted),
        "rejected": rejected,
        "errors": errors,
        "reclassified": reclassified,
        "by_kind": dict(Counter(q["kind"] for q in accepted)),
        "by_minimum_chunks": dict(Counter(q["necessary_original_chunks"] for q in accepted)),
        "strict_chains": sum(q["strict_directed_chain"] for q in accepted),
        "expert_reviewed": False,
        "predictions_seen": False,
        "final": final,
    }
    save(OUT / "question-adoption-preview.json", audit)
    if final:
        assert not errors, "Resolve all source annotation errors before any predictions"
        assert not (ROOT / "evaluation-freeze.json").exists()
        (ROOT / "questions-final.jsonl").write_text(
            "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in accepted), encoding="utf8"
        )
        save(ROOT / "blind-question-independent-review.json", review)
        save(ROOT / "question-final-adoption-audit.json", audit)
        save(
            ROOT / "question-freeze.json",
            {
                "questions_sha256": digest(ROOT / "questions-final.jsonl"),
                "graph_freeze_sha256": digest(ROOT / "graph-freeze.json"),
                "author_gold_not_used": True,
                "all_labels_validated_before_predictions": True,
                "questions": len(accepted),
                "clusters": len({q["cluster"] for q in accepted}),
            },
        )
    print(
        json.dumps(
            {k: v for k, v in audit.items() if k not in {"rejected", "errors", "reclassified"}},
            ensure_ascii=False,
        ),
        flush=True,
    )
    if errors:
        print("Errors", errors, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--final", action="store_true")
    p.add_argument("--review", type=Path)
    a = p.parse_args()
    adopt(a.final, a.review)
