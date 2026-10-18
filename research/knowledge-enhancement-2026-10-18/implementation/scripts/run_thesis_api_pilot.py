"""Budget-capped developer smoke run; outputs are not formal system quality scores."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import build_academic_benchmark as builder

from aerodiagnosis.adapters.llm.research_client import (
    ResearchCallError,
    ResearchClient,
    load_credential,
)
from aerodiagnosis.application.knowledge_applicability import assess_applicability
from aerodiagnosis.application.knowledge_graph_extractor import KnowledgeGraphExtractor
from aerodiagnosis.application.qualified_graph_extraction import (
    extraction_instruction,
    extraction_schema,
    parse_candidates,
    validate_candidates,
)
from aerodiagnosis.domain.diagnosis import EvidenceItem
from aerodiagnosis.domain.evidence import EvidenceLocator, SourceKind, make_evidence_id
from aerodiagnosis.domain.evidence_bundle import EvidenceBundle
from aerodiagnosis.domain.knowledge_query import KnowledgeQueryContext
from aerodiagnosis.domain.qualified_claim import SourceUnit

QA_IDS = ("gp-q001", "gp-q018", "gp-q022")
GRAPH_RANGES = ("gp-r001", "gp-r025", "gp-r026", "gp-r029")


def cited_snapshot_texts(bundle: EvidenceBundle, evidence_ids: list[str]) -> list[str]:
    """Join only overlapping/adjacent intervals from the same original snapshot."""
    citations = {e.evidence_id: e for e in bundle.evidence}
    groups: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
    for identifier in evidence_ids:
        evidence = citations[identifier]
        if evidence.excerpt_truncated:
            raise ValueError("Truncated evidence cannot establish a complete quote")
        if builder.digest(evidence.excerpt.encode()) != evidence.content_hash:
            raise ValueError("Evidence excerpt fingerprint mismatch")
        start, end = evidence.locator.get("start"), evidence.locator.get("end")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or isinstance(start, bool)
            or isinstance(end, bool)
            or start < 0
            or end <= start
            or end - start != len(evidence.excerpt)
        ):
            raise ValueError("Citation lacks an exact canonical interval")
        key = (evidence.document_id, evidence.version_id)
        groups.setdefault(key, []).append((start, end, evidence.excerpt))
    texts = []
    for spans in groups.values():
        current_start, current_end, current_text = sorted(spans)[0]
        for start, end, text in sorted(spans)[1:]:
            if start > current_end:
                texts.append(current_text)
                current_start, current_end, current_text = start, end, text
                continue
            overlap = min(current_end, end) - start
            if (
                current_text[start - current_start : start - current_start + overlap]
                != text[:overlap]
            ):
                raise ValueError("Overlapping citation text disagrees inside one snapshot")
            if end > current_end:
                current_text += text[current_end - start :]
                current_end = end
        texts.append(current_text)
    return texts


def validate_claim_quotes(
    bundle: EvidenceBundle,
    payload: dict[str, Any],
    expected_claims: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Necessary citation checks; semantic judgments remain explicitly AI-only."""
    citations = {e.evidence_id: e for e in bundle.evidence}
    accepted, rejected = [], []
    seen = set()
    rows = payload.get("claims")
    if not isinstance(rows, list):
        return {
            "accepted": [],
            "rejected": [{"reasons": ["missing_claim_list"]}],
            "semantic_gold": False,
        }
    expected = {r["claim_id"]: r for r in expected_claims or []}
    for row in rows:
        reasons = []
        if not isinstance(row, dict):
            rejected.append({"raw": row, "reasons": ["not_an_object"]})
            continue
        identifier = row.get("claim_id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            reasons.append("invalid_or_duplicate_claim_id")
        if isinstance(identifier, str):
            seen.add(identifier)
        if expected_claims is not None and identifier not in expected:
            reasons.append("verifier_added_unknown_claim")
        if identifier in expected and row.get("text") != expected[identifier].get("text"):
            reasons.append("verifier_changed_claim_text")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            reasons.append("missing_claim_text")
        evidence_ids = row.get("evidence_ids", [])
        valid_refs = (
            isinstance(evidence_ids, list)
            and bool(evidence_ids)
            and all(isinstance(e, str) and e in citations for e in evidence_ids)
        )
        if not valid_refs:
            reasons.append("missing_or_unknown_citation")
        quote = row.get("support_quote")
        texts = []
        if valid_refs:
            try:
                texts = cited_snapshot_texts(bundle, evidence_ids)
            except ValueError:
                reasons.append("citation_snapshot_inconsistent")
        if not isinstance(quote, str) or not quote.strip() or not any(quote in t for t in texts):
            reasons.append("quote_not_in_cited_snapshot")
        if row.get("support") != "supported":
            reasons.append("ai_support_not_confirmed")
        if reasons:
            rejected.append({"candidate": row, "reasons": reasons})
        else:
            accepted.append(
                {**row, "support_basis": "ai_judgment_plus_exact_quote_check_not_expert_truth"}
            )
    for identifier, original in expected.items():
        if identifier not in seen:
            rejected.append({"candidate": original, "reasons": ["verifier_omitted_claim"]})
    return {"accepted": accepted, "rejected": rejected, "semantic_gold": False}


def assemble_traceable_report(bundle: EvidenceBundle, checks: dict[str, Any]) -> dict[str, Any]:
    """Render checked statements without letting a draft reintroduce rejected facts."""
    used = {identifier for claim in checks["accepted"] for identifier in claim["evidence_ids"]}
    return {
        "bundle_id": bundle.bundle_id,
        "query": bundle.query,
        "known_conditions": bundle.known_conditions.model_dump(mode="json"),
        "source_statements": checks["accepted"],
        "conflicts": bundle.conflicts,
        "unresolved_claims": [
            {
                "claim_id": r.get("candidate", {}).get("claim_id"),
                "status": "unverified",
                "reasons": r["reasons"],
            }
            for r in checks["rejected"]
        ],
        "citations": [e.model_dump(mode="json") for e in bundle.evidence if e.evidence_id in used],
        "status": "source_checked_ai_report_not_expert_validated",
        "limitations": [
            "AI semantic judgments are not expert truth",
            "No current maintenance applicability or diagnostic accuracy established",
        ],
    }


def make_bundle(
    query: dict[str, Any], prediction: dict[str, Any], sources: dict[str, Any]
) -> EvidenceBundle:
    context = KnowledgeQueryContext()
    evidence = []
    for hit in prediction["chunks"]:
        source = sources[hit["document_id"]]
        if hit["canonical_sha256"] != source["canonical_sha256"]:
            raise ValueError("Retrieval snapshot changed")
        start, end = hit["canonical_start"], hit["canonical_end"]
        if not 0 <= start < end <= len(source["text"]):
            raise ValueError("Retrieval interval outside source")
        text = source["text"][start:end]
        locator = EvidenceLocator(
            kind="canonical_span",
            coordinates={
                "start": start,
                "end": end,
                "canonical_sha256": source["canonical_sha256"],
            },
        )
        source_ref = f"{source['canonical_sha256']}:{start}:{end}"
        record = {
            "source_type": source["source_type"],
            "source_revision": source["document_description"].get("revision"),
        }
        evidence.append(
            EvidenceItem(
                evidence_id=make_evidence_id(
                    SourceKind.DOCUMENT_CHUNK, source_ref, locator.canonical()
                ),
                source_kind="document_chunk",
                source_ref=source_ref,
                document_id=hit["document_id"],
                version_id=source["canonical_sha256"],
                content_hash=builder.digest(text.encode()),
                excerpt=text,
                locator=dict(locator.coordinates),
                score=0,
                applicability_status=assess_applicability(record, context),
                document_metadata={
                    **record,
                    "source_url": source["source_url"],
                    "publisher_id": source["publisher_id"],
                    "source_family_id": source["family_id"],
                    "raw_sha256": source["raw_sha256"],
                },
            )
        )
    identity = builder.annotation_hash(
        {"query": query["query"], "evidence": [e.model_dump(mode="json") for e in evidence]}
    )
    return EvidenceBundle(
        bundle_id=identity,
        query=query["query"],
        known_conditions=context,
        evidence=tuple(evidence),
        retrieval_trace={
            "configuration": "preselected fixed BM25 baseline",
            "mode": "development smoke; no gold evidence packing",
            "sample_id": query["sample_id"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/gas_path_pilot_v1"))
    parser.add_argument("--retrieval-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--provider-runtime", type=Path, default=Path(".runtime/thesis-research-provider-v1")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh immutable run output")
    builder.validate_release(args.dataset, args.dataset / "protocol.json")
    sources = builder.load_sources(args.dataset / "corpus_manifest.json")
    # Registration precedes requests; the extraction never loads task gold.
    ranges = {
        r["range_id"]: r
        for r in builder.read_jsonl(Path("evaluation/gas_path_scope_v1/range_annotations.jsonl"))
    }
    args.output.mkdir(parents=True)
    run_id = uuid.uuid4().hex
    config = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "graph_units": GRAPH_RANGES,
        "qa_ids": QA_IDS,
        "model": "deepseek-flash",
        "stage": "pilot",
        "cap_cny": 5,
        "output_token_limits": {
            "G1": 1024,
            "G2": 2048,
            "QA": 1200,
            "verification": 1400,
            "report": 1200,
        },
        "source_dataset_release_sha256": builder.digest(
            (args.dataset / "release.json").read_bytes()
        ),
        "script_sha256": builder.digest(Path(__file__).read_bytes()),
        "paid_retries": 0,
        "production_ingestion": False,
        "limitations": [
            "developer smoke, not formal quality evaluation",
            "no exhaustive extraction gold",
            "AI support judgment, not human/expert validation",
            "three preselected developer examples, not statistical sample",
        ],
    }
    config["shared_provider_runtime"] = str(args.provider_runtime)
    (args.output / "source_script.py").write_bytes(Path(__file__).read_bytes())
    builder.write_json(args.output / "configuration.json", config)
    builder.write_json(args.output / "run_state.json", {"status": "running", **config})
    client = ResearchClient(
        api_key=load_credential(args.credential_file),
        runtime=args.provider_runtime,
        cap_cny=5,
        timeout=60,
    )
    results = []
    failures = []
    try:
        for range_id in GRAPH_RANGES:
            row = ranges[range_id]
            span = row["evidence"]
            source = sources[span["document_id"]]
            unit = SourceUnit(
                unit_id=range_id,
                document_id=span["document_id"],
                canonical_sha256=source["canonical_sha256"],
                canonical_start=span["canonical_start"],
                text=source["text"][span["canonical_start"] : span["canonical_end"]],
                source_locator=span["source_locator"],
                source_kind=source["source_type"],
            )
            nodes, edges = KnowledgeGraphExtractor().extract(
                source_ref=unit.document_id,
                version_id=unit.canonical_sha256,
                chunks=[unit.text],
                chunk_locators=[{"start": unit.canonical_start, "end": unit.canonical_end}],
            )
            results.append(
                {
                    "task": range_id,
                    "method": "G0_existing_lexical_observation",
                    "nodes": [asdict(n) for n in nodes],
                    "edges": [asdict(e) for e in edges],
                    "limits": (
                        "Existing lexical scope differs from qualified schema; not a matched G0 "
                        "quality baseline."
                    ),
                }
            )
            g1 = client.complete(
                task=range_id + "-G1",
                instruction=(
                    "Extract source-supported entities and relationships freely. Return "
                    '{"claims": [...]} with source quotes. Preserve negation and uncertainty.'
                ),
                payload={"source_unit": unit.model_dump(mode="json")},
                max_tokens=1024,
            )
            results.append(
                {"task": range_id, "method": "G1", "prediction": g1.value, "receipt": g1.receipt}
            )
            g2 = client.complete(
                task=range_id + "-G2",
                instruction=extraction_instruction(),
                payload={
                    "source_unit": unit.model_dump(mode="json"),
                    "schema": extraction_schema(),
                },
                max_tokens=2048,
            )
            candidates, invalid = parse_candidates(g2.value)
            g3 = validate_candidates(unit, candidates)
            results.append(
                {
                    "task": range_id,
                    "method": "G2",
                    "prediction": g2.value,
                    "receipt": g2.receipt,
                    "schema_rejections": [r.model_dump(mode="json") for r in invalid],
                }
            )
            results.append(
                {
                    "task": range_id,
                    "method": "G3_necessary_checks",
                    "prediction": g3.model_dump(mode="json"),
                }
            )
            builder.write_json(args.output / "graph_outputs.json", results)
            print("graph", range_id, flush=True)
        queries = {q["sample_id"]: q for q in builder.read_jsonl(args.dataset / "queries.jsonl")}
        predictions = builder.read_jsonl(
            args.retrieval_run / "fixed-fine-bm25-740d1b6135bf-dev-native.predictions.jsonl"
        )
        hit_map = {p["sample_id"]: p for p in predictions}
        qa = []
        for sid in QA_IDS:
            bundle = make_bundle(queries[sid], hit_map[sid], sources)
            builder.write_json(args.output / (sid + ".bundle.json"), bundle.model_dump(mode="json"))
            answer = client.complete(
                task=sid + "-QA",
                instruction=(
                    "Answer the question using only this EvidenceBundle. Return claims with "
                    "claim_id,text,statement_type "
                    "(fact/hypothesis/unknown/check),evidence_ids. Cite exact evidence IDs; "
                    "state missing evidence and applicability. Do not invent limits, "
                    "probabilities or maintenance steps."
                ),
                payload={"bundle": bundle.model_dump(mode="json")},
                max_tokens=1200,
            )
            checked = client.complete(
                task=sid + "-verification",
                instruction=(
                    "Independently check every proposed claim against the supplied evidence. "
                    'Return {"claims":[{"claim_id":...,"text":...,"evidence_ids":[...],"suppo'
                    'rt":"supported" or "unsupported" or "uncertain","support_quote": exact '
                    "source substring}]}. Check negation, numbers, units, "
                    "model/phase/revision and causal certainty. A quote must actually support"
                    " the whole claim, not just share keywords. Preserve all claims including"
                    " unsupported ones."
                ),
                payload={"bundle": bundle.model_dump(mode="json"), "proposed_answer": answer.value},
                max_tokens=1400,
            )
            checks = validate_claim_quotes(bundle, checked.value, answer.value.get("claims", []))
            report = client.complete(
                task=sid + "-report",
                instruction=(
                    "Organize a traceable short report using only the supplied accepted "
                    "facts. Return JSON with input_facts,applicability,candidate_explanations"
                    ",unknowns,check_leads,citations. Every factual sentence requires "
                    "evidence_ids and text. Preserve supported wording and uncertainty. Leave"
                    " unsupported sections empty; do not add operational maintenance steps."
                ),
                payload={
                    "query": bundle.query,
                    "known_conditions": bundle.known_conditions.model_dump(mode="json"),
                    "accepted_facts": checks["accepted"],
                    "unresolved_claims": checks["rejected"],
                },
                max_tokens=1200,
            )
            qa.append(
                {
                    "sample_id": sid,
                    "bundle_id": bundle.bundle_id,
                    "answer": answer.value,
                    "verification": checked.value,
                    "necessary_checks": checks,
                    "report_draft": report.value,
                    "traceable_report": assemble_traceable_report(bundle, checks),
                    "report_status": "draft_requires_final_claim_review",
                    "receipts": [answer.receipt, checked.receipt, report.receipt],
                }
            )
            builder.write_json(args.output / "qa_report_outputs.json", qa)
            print("qa-report", sid, flush=True)
        builder.write_json(
            args.output / "run_state.json",
            {
                "status": "completed_smoke",
                "run_id": run_id,
                "graph_units": len(GRAPH_RANGES),
                "qa_report_cases": len(QA_IDS),
                "finished_at": datetime.now(UTC).isoformat(),
                "formal_quality_scores": False,
            },
        )
    except (ResearchCallError, ValueError, KeyError, TypeError) as exc:
        failures.append({"error_type": type(exc).__name__, "response_or_secret_not_logged": True})
        builder.write_json(args.output / "failures.json", failures)
        builder.write_json(
            args.output / "run_state.json",
            {
                "status": "failed_preserved",
                "run_id": run_id,
                "failures": failures,
                "finished_at": datetime.now(UTC).isoformat(),
            },
        )
        raise SystemExit(1) from None
    finally:
        summary = client.summary()
        builder.write_json(args.output / "budget_receipt.json", summary)
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "provider_call_count",
                        "budget_upper_estimate_cny",
                        "pending_or_unknown_calls",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
