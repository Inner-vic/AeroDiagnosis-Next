# ruff: noqa: E501, RUF001
"""Source-check independently reconstructed labels; no retrieval outputs are available here."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from author_followup_questions import locate_quote
from prepare_capability_holdout import OUT, ROOT, read, save

REVIEW = Path(
    "D:/zhaoweihao3/Desktop/Project/AeroDiagnosis/.runtime/capability-holdout-blind-question-review.json"
)


def compile_review(final=False):
    review = read(REVIEW)
    assert (
        not review["gold_answers_seen"]
        and not review["predictions_seen"]
        and not review["expert_reviewed"]
    )
    mapping = read(OUT / "blind-question-private-map.json")
    data = read(ROOT / "translated-corpus-final.json")
    chunks = {c["chunk_id"]: c for c in data["chunks"]}
    docs = {d["document_id"]: d for d in data["documents"]}
    if final:
        assert review.get("coverage_complete"), "Independent source review incomplete"
        assert {x["blind_id"] for x in review["items"]} == set(mapping)
    accepted = []
    rejected = []
    errors = []
    reclassified = []
    for item in review["items"]:
        identity = item["blind_id"]
        parent = mapping[identity]
        if identity in set(
            read(ROOT / "annotation-errata.json")["excluded_blind_ids"]
            if (ROOT / "annotation-errata.json").exists()
            else []
        ):
            assert any(not support["quote"].strip() for support in item["supports"])
            rejected.append(
                {
                    "blind_id": identity,
                    "family": parent["family"],
                    "reason": "empty_gold_quote_unscorable_uniform_nonempty_validation",
                }
            )
            continue
        if item["query_quality"] == "reject":
            rejected.append(
                {
                    "blind_id": identity,
                    "family": parent["family"],
                    "reason": item.get("reason", "source reviewer rejected"),
                }
            )
            continue
        try:
            query = (
                item.get("query_revision") if item["query_quality"] == "revise" else item["query"]
            )
            if not isinstance(query, str) or not query.strip() or "cap1-" in query:
                raise ValueError("Invalid/internal-ID query")
            if re.search(r"FAA\s*(?:AD|适航指令)\s*20\d{2}-\d{5}\b", query):
                raise ValueError("Federal Register document number misused as formal AD number")
            units = []
            identifiers = []
            alignment = []
            for i, support in enumerate(item["supports"]):
                if not isinstance(support["quote"], str) or not support["quote"].strip():
                    raise ValueError("Empty gold quote")
                c = chunks[support["chunk_id"]]
                d = docs[c["document_id"]]
                assert d["source_family"] == parent["family"]
                if c["metadata"].get("source_annotation_restriction"):
                    raise ValueError("Unresolved source ambiguity used as gold")
                offset, quote, mode = locate_quote(c["content"], support["quote"])
                start = c["metadata"]["locator"]["coordinates"]["start"] + offset
                assert d["content"][start : start + len(quote)] == quote
                units.append(
                    {
                        "unit_id": str(i),
                        "source_key": d["source_key"],
                        "start": start,
                        "end": start + len(quote),
                        "quote": quote,
                        "source_sha256": d["source_sha256"],
                        "requirement_zh": support["requirement_zh"],
                    }
                )
                identifiers.append(c["chunk_id"])
                alignment.append(mode)
            sets = [[str(i) for i in s] for s in item["sufficient_sets"]]
            if not sets or any(not s or not set(s) <= {u["unit_id"] for u in units} for s in sets):
                raise ValueError("Invalid sufficient sets")
            minimum = min(len({identifiers[int(i)] for i in s}) for s in sets)
            if minimum != item["minimum_original_chunks"]:
                raise ValueError("Review minimum count disagrees with its own sufficient sets")
            planned = parent["planned_kind"]
            kind = planned
            if planned in {"S4_two_support", "S5_three_support"}:
                kind = (
                    "S5_three_support"
                    if minimum >= 3
                    else "S4_two_support"
                    if minimum == 2
                    else "S_single_support_synthesis"
                )
            structure = item["evidence_structure"]
            if "one_substantive_unit" in structure or "context_binding" in structure:
                kind = "S_context_binding"
            elif "two_substantive_units" in structure and planned in {
                "S4_two_support",
                "S5_three_support",
            }:
                kind = "S4_two_support"
            if kind != planned:
                reclassified.append(
                    {
                        "blind_id": identity,
                        "planned": planned,
                        "actual": kind,
                        "minimum_original_chunks": minimum,
                    }
                )
            answer = item["answer_zh"]
            for original, chinese in {
                "Rolls-Royce Deutschland Ltd & Co KG": "罗尔斯罗伊斯德国有限公司",
                "International Aero Engines, LLC": "国际航空发动机公司（IAE）",
                "International Aero Engines": "国际航空发动机公司（IAE）",
                "CFM International, S.A.": "CFM国际公司",
                "Rolls-Royce Alert NMSB": "罗尔斯罗伊斯警报类非改装服务通告",
                "Group1": "第1组",
            }.items():
                answer = answer.replace(original, chinese)
            answer = re.sub(r"\bengines\b", "发动机", answer)
            answer = re.sub(r"\band\b", "及", answer)
            if not re.search(r"[\u3400-\u9fff]", answer):
                # Identifiers remain verbatim; only add a Chinese display label, never a new factual clause.
                if re.fullmatch(r"P/N\s*[A-Z0-9/-]+[.。]?", answer):
                    answer = "部件号为" + answer
                elif re.fullmatch(r"[A-Za-z0-9,、 /.-]+[.。]?", answer):
                    answer = "对应标识为" + answer
                else:
                    raise ValueError("Reference answer is not Chinese")
            root_term_adjustment = None
            if identity == "541546f541d6704f":
                answer = answer.replace(
                    "下次飞行前退出发动机，放行前联系RR并完成修理",
                    "下次飞行前将发动机停用，放行前联系RR取得适用修理指令并按指令执行",
                )
                root_term_adjustment = (
                    "remove engine from service is 停用; accomplish repair instructions retained"
                )
            elif identity == "0dc81cb2a06ad784":
                query = (
                    query.replace("在何时将发动机拆下送修", "在何时停用发动机")
                    .replace("又应在何时拆下", "又应在何时停用发动机")
                    .replace("拆下后更换", "停用后更换")
                )
                root_term_adjustment = "Remove from service does not independently require physical engine removal/shipping"
            accepted.append(
                {
                    "sample_id": "cap1:" + identity,
                    "blind_id": identity,
                    "cluster": parent["family"],
                    "kind": kind,
                    "planned_kind": planned,
                    "query": query.strip(),
                    "reference_answer": answer,
                    "answerable": True,
                    "support_units": units,
                    "acceptable_support_sets": sets,
                    "necessary_original_chunks": minimum,
                    "provided_chunk_ids": identifiers,
                    "strict_directed_chain": item["strict_directed_chain"],
                    "multi_block_chain_required": bool(
                        item["strict_directed_chain"] and minimum >= 2
                    ),
                    "evidence_structure": item["evidence_structure"],
                    "annotation_status": "independent_answer_hidden_nonexpert_source_reconstruction_adjudicated_before_predictions",
                    "expert_reviewed": False,
                    "quote_alignment": alignment,
                    "source_review_reason": item.get("reason"),
                    "query_revised_before_predictions": item["query_quality"] == "revise",
                    "root_source_term_adjustment": root_term_adjustment,
                }
            )
        except (KeyError, ValueError, TypeError, AssertionError) as error:
            errors.append({"blind_id": identity, "reason": str(error)[:200]})
    audit = {
        "candidate_queries": len(mapping),
        "blind_reconstructed": len(review["items"]),
        "accepted": len(accepted),
        "source_rejected": rejected,
        "anchor_or_schema_errors": errors,
        "reclassified_before_predictions": reclassified,
        "by_kind": dict(Counter(q["kind"] for q in accepted)),
        "by_necessary_chunks": dict(Counter(q["necessary_original_chunks"] for q in accepted)),
        "expert_reviewed": False,
        "predictions_seen": False,
        "final": final,
    }
    save(
        OUT
        / (
            "question-final-adoption-audit.json"
            if final
            else "question-adoption-preview-audit.json"
        ),
        audit,
    )
    if final:
        if errors:
            raise ValueError("Unresolved gold source anchors; do not freeze")
        save(ROOT / "blind-question-independent-review.json", review)
        (ROOT / "questions-final.jsonl").write_text(
            "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in accepted), encoding="utf8"
        )
        save(ROOT / "question-final-adoption-audit.json", audit)
    print(
        "Reconstructed",
        len(review["items"]),
        "accepted",
        len(accepted),
        "source rejected",
        len(rejected),
        "anchor errors",
        len(errors),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--final", action="store_true")
    a = p.parse_args()
    compile_review(a.final)
