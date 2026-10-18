# ruff: noqa: E501
"""Validate source quote anchors; no evaluation questions are read."""

from __future__ import annotations

from pathlib import Path

from author_followup_questions import locate_quote
from prepare_capability_holdout import OUT, ROOT, read, save


def compile_graph(review_path=None):
    data = read(
        ROOT
        / ("translated-corpus-final.json" if review_path else "translated-corpus-candidate.json")
    )
    decisions = {}
    if review_path:
        review = read(review_path)
        assert (
            not review["expert_reviewed"]
            and not review["questions_seen"]
            and not review["predictions_seen"]
        )
        assert set(review["reviewed_document_ids"]) == {d["document_id"] for d in data["documents"]}
        decisions = {
            (x["document_id"], x["assertion_index"]): x
            for x in review["items"]
            if x.get("in_anchored_candidate", False)
        }
        original = read(OUT / "raw-graph-anchors/source-graph-candidate.json")
        necessary = {
            (e["source_ref"].removeprefix("document:"), int(e["edge_id"].rsplit(":", 1)[1]))
            for e in original["edges"]
        }
        assert necessary <= decisions.keys(), "Raw anchor-valid edge was not semantically reviewed"
    chunks = {c["chunk_id"]: c for c in data["chunks"]}
    nodes = []
    edges = []
    rejected = []
    alignment = []
    for doc in data["documents"]:
        f = OUT / "extraction" / (doc["document_id"] + ".json")
        if not f.exists():
            raise ValueError("Source extraction incomplete")
        row = read(f)
        keymap = row["chunk_key_map"]
        assert row["source_sha256"] == doc["source_sha256"]

        def proof(key, quote, keymap=keymap, doc=doc):
            c = chunks[keymap[str(key)]]
            if review_path and c["metadata"].get("source_annotation_restriction"):
                raise ValueError(
                    "Unresolved source ambiguity cannot anchor a final graph assertion"
                )
            offset, exact, mode = locate_quote(c["content"], quote)
            lo = c["metadata"]["locator"]["coordinates"]["start"] + offset
            if mode != "exact":
                alignment.append(
                    {"document_id": doc["document_id"], "chunk_id": c["chunk_id"], "mode": mode}
                )
            return {
                "chunk_id": c["chunk_id"],
                "document_id": doc["document_id"],
                "canonical_start": lo,
                "canonical_end": lo + len(exact),
                "canonical_sha256": doc["source_sha256"],
                "evidence_text": exact,
            }

        for i, a in enumerate(row["assertions"]):
            if review_path:
                decision = decisions.get((doc["document_id"], i))
                if decision is None or decision["decision"] == "reject":
                    rejected.append(
                        {
                            "document_id": doc["document_id"],
                            "assertion_index": i,
                            "reason": "not_reviewed_or_rejected_by_independent_source_review",
                        }
                    )
                    continue
                if decision["decision"] == "revise":
                    a = decision["assertion"]
            try:
                primary = proof(a["chunk"], a["quote"])
                extras = [proof(x["chunk"], x["quote"]) for x in a.get("supporting_quotes", [])]
                both = []
                for side in ["source", "target"]:
                    e = a[side]
                    surface = e["surface"]
                    options = [
                        p
                        for p in [primary, *extras]
                        if surface.casefold() in p["evidence_text"].casefold()
                    ]
                    if not options or not e["role"].strip() or not 1 <= len(e["name"]) <= 40:
                        raise ValueError("Invalid entity source or label")
                    link = {
                        **options[0],
                        "kind": "Concept",
                        "role": doc["agency"] + ":" + e["role"],
                        "surface": surface,
                        "review_status": "source_verified_nonexpert"
                        if review_path
                        else "automatic_anchor_only",
                    }
                    key = doc["document_id"] + ":mention:" + str(i) + ":" + side
                    both.append(
                        {
                            "node_id": key,
                            "name": e["name"],
                            "kind": "Concept",
                            "description": "Source-only model extraction with exact evidence; nonexpert reviewed research snapshot",
                            "source_ref": "document:" + doc["document_id"],
                            "version_id": doc["version_id"],
                            "properties": {"entity_link": link, "mention_source": options[0]},
                        }
                    )
                if a["relation"] not in {
                    "CAUSES",
                    "CONTRIBUTES_TO",
                    "INDICATES",
                    "APPLIES_TO",
                    "REQUIRES",
                    "PREVENTS",
                    "HAS_PART",
                    "DEFINES",
                    "EXEMPTS",
                }:
                    raise ValueError("Invalid relation")
                if a["certainty"] not in {"established", "possible", "suspected"} or a[
                    "polarity"
                ] not in {"positive", "negative"}:
                    raise ValueError("Invalid modality")
                eid = doc["document_id"] + ":assertion:" + str(i)
                nodes.extend(both)
                edges.append(
                    {
                        "edge_id": eid,
                        "source_id": both[0]["node_id"],
                        "target_id": both[1]["node_id"],
                        "relation": a["relation"],
                        "source_ref": "document:" + doc["document_id"],
                        "version_id": doc["version_id"],
                        "confidence": 1.0,
                        "properties": {
                            **primary,
                            "supporting_evidence": extras,
                            "validation_status": "necessary_source_checks_passed",
                            "polarity": a["polarity"],
                            "certainty": a["certainty"],
                            "qualifiers": {
                                "condition": a["condition_zh"],
                                "jurisdiction": doc["agency"],
                                "scope": "this document only",
                                "curation": "source_only_model_nonexpert_reviewed",
                            },
                            "independent_source_review": decisions[(doc["document_id"], i)][
                                "decision"
                            ]
                            if review_path
                            else "pending",
                        },
                    }
                )
            except (KeyError, ValueError, TypeError) as e:
                rejected.append(
                    {
                        "document_id": doc["document_id"],
                        "assertion_index": i,
                        "reason": str(e)[:160],
                    }
                )
    save(
        ROOT / ("source-graph-final.json" if review_path else "source-graph-candidate.json"),
        {
            "schema": "source-only-model-graph@1",
            "nodes": nodes,
            "edges": edges,
            "nonexpert_semantic_review_pending": not bool(review_path),
            "automatic_source_anchor_validation_only": not bool(review_path),
        },
    )
    save(
        ROOT
        / (
            "source-graph-review-adoption-audit.json"
            if review_path
            else "source-graph-anchor-audit.json"
        ),
        {
            "accepted": len(edges),
            "rejected": rejected,
            "quote_alignment": alignment,
            "before_questions": True,
        },
    )
    print("Graph anchored", len(nodes), len(edges), "rejected", len(rejected), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    compile_graph(args.review)
