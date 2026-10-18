from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE = Path(__file__).resolve().parents[2] / "scripts/build_academic_benchmark.py"
spec = importlib.util.spec_from_file_location("academic_benchmark_test", MODULE)
assert spec and spec.loader
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


def source(
    family: str = "F1", text: str = "No stall at 12 kg. repeated repeated."
) -> dict[str, Any]:
    return {
        "family_id": family,
        "text": text,
        "canonical_sha256": builder.digest(text.encode()),
        "raw_sha256": builder.digest(b"fixture"),
        "blocks": [
            {
                "text": text,
                "canonical_start": 0,
                "canonical_end": len(text),
                "source_locator": {"page": 2},
            }
        ],
    }


def anchor(docid: str = "D1", text: str = "No stall at 12 kg.") -> dict[str, Any]:
    return {"document_id": docid, "block_index": 0, "anchor_text": text}


def query(sid: str = "q1", docid: str = "D1", **kwargs: Any) -> dict[str, Any]:
    return {
        "sample_id": sid,
        "query": "What happened at 12 kg?",
        "query_language": "en",
        "task_type": "mechanism",
        "scope": {},
        "answerability": "answerable",
        "reference_answer": "No stall.",
        "evidence": [anchor(docid)],
        "evidence_requirement": "all",
        "hard_negatives": [],
        "ambiguity_notes": [],
        "fact_id": sid,
        "paired_variant_of": None,
        "author": "agent:author",
        **kwargs,
    }


def test_unicode_span_preserves_negation_and_units() -> None:
    text = "标题:No stall at 12 kg."
    span = builder.materialize_span(anchor(text="No stall at 12 kg."), {"D1": source(text=text)})
    assert span["canonical_start"] == 3
    assert span["canonical_end"] == len(text)
    assert span["text_sha256"] == builder.digest(b"No stall at 12 kg.")
    assert span["source_locator"] == {"page": 2}


@pytest.mark.parametrize(
    "bad",
    [
        anchor(text="repeated"),
        anchor(text="stall at 13 kg"),
        anchor(text=""),
        {**anchor(), "block_index": -1},
    ],
)
def test_ambiguous_missing_empty_or_negative_anchor_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(builder.AnnotationError):
        builder.materialize_span(bad, {"D1": source()})


def test_no_answer_needs_audit_and_family() -> None:
    raw = query(answerability="insufficient_evidence", evidence=[])
    with pytest.raises(builder.AnnotationError, match="audit"):
        builder.compile_samples([raw], {"D1": source()}, set())
    raw["no_answer_audit"] = {
        "reason": "Not specified",
        "search_terms": ["limit"],
        "reviewed_document_ids": ["D1"],
    }
    with pytest.raises(builder.AnnotationError, match="evaluation family"):
        builder.compile_samples([raw], {"D1": source()}, set())
    raw["evaluation_document_ids"] = ["D1"]
    result = builder.compile_samples([raw], {"D1": source()}, set())
    assert result[0]["evidence"] == [] and result[0]["family_ids"] == ["F1"]


def test_required_evidence_cross_split_rejected() -> None:
    raw = query(evidence=[anchor(), anchor("D2")])
    with pytest.raises(builder.AnnotationError, match="crosses dev/test"):
        builder.compile_samples([raw], {"D1": source(), "D2": source("F2")}, {"F2"})


def test_same_fact_cannot_cross_splits() -> None:
    q2 = query("q2", "D2", query="A paraphrased question", fact_id="q1")
    with pytest.raises(builder.AnnotationError, match="Fact leaks"):
        builder.compile_samples([query(), q2], {"D1": source(), "D2": source("F2")}, {"F2"})


def test_translated_pair_needs_same_fact() -> None:
    q2 = query("q2", query="十二公斤时发生什么?", query_language="zh", paired_variant_of="q1")
    with pytest.raises(builder.AnnotationError, match="translated pair"):
        builder.compile_samples([query(), q2], {"D1": source()}, set())
    q2["fact_id"] = "q1"
    samples = builder.compile_samples([query(), q2], {"D1": source()}, set())
    assert samples[1]["fact_id"] == samples[0]["fact_id"]


def test_duplicate_query_and_duplicate_span_rejected() -> None:
    with pytest.raises(builder.AnnotationError, match="Duplicate query"):
        builder.compile_samples([query(), query("q2")], {"D1": source()}, set())
    with pytest.raises(builder.AnnotationError, match="Duplicate spans"):
        builder.compile_samples([query(evidence=[anchor(), anchor()])], {"D1": source()}, set())


def test_blind_packet_omits_author_answers_evidence_and_verdict() -> None:
    raw = query()
    view = builder.blind_view(raw)
    assert (
        not {
            "answerability",
            "reference_answer",
            "evidence",
            "author",
            "hard_negatives",
            "task_type",
        }
        & view.keys()
    )
    packet = builder.blind_packets([raw], {"D1": source()})[0]
    assert packet["blind_sha256"] == builder.annotation_hash(view)
    assert packet["pool_is_exhaustive"] is False


def test_correlated_families_form_transitive_clusters() -> None:
    clusters = builder.family_cluster_map(
        [{"family_ids": ["F1", "F2"]}, {"family_ids": ["F2", "F3"]}, {"family_ids": ["F4"]}],
        [["F3", "F5"]],
    )
    assert clusters["F1"] == clusters["F3"] == clusters["F5"]
    assert clusters["F4"] != clusters["F1"]


def review_fixture() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw = query()
    samples = builder.compile_samples([raw], {"D1": source()}, set())
    review = {
        "sample_id": "q1",
        "reviewer": "agent:reviewer",
        "blind_sha256": builder.annotation_hash(builder.blind_view(raw)),
        "reconstructed_answerability": "answerable",
        "evidence": [anchor()],
    }
    reconciliation = {
        "sample_id": "q1",
        "decision": "accept",
        "reason": "Same finding",
        "disposition": "agreed",
        "final_annotation_sha256": samples[0]["annotation_sha256"],
    }
    return samples, raw, review, reconciliation


def test_review_cannot_be_self_review_or_stale() -> None:
    samples, raw, review, reconciliation = review_fixture()
    review["reviewer"] = raw["author"]
    with pytest.raises(builder.AnnotationError, match="Self review"):
        builder.apply_reviews(samples, {"q1": raw}, [review], [reconciliation], {"D1": source()})
    review["reviewer"] = "agent:other"
    review["blind_sha256"] = "0" * 64
    with pytest.raises(builder.AnnotationError, match="no longer binds"):
        builder.apply_reviews(samples, {"q1": raw}, [review], [reconciliation], {"D1": source()})


def test_disagreement_needs_explicit_adjudication() -> None:
    samples, raw, review, reconciliation = review_fixture()
    review["reconstructed_answerability"] = "insufficient_evidence"
    with pytest.raises(builder.AnnotationError, match="Unresolved"):
        builder.apply_reviews(samples, {"q1": raw}, [review], [reconciliation], {"D1": source()})
    reconciliation["disposition"] = "adjudicated_disagreement"
    kept, _ = builder.apply_reviews(
        samples, {"q1": raw}, [review], [reconciliation], {"D1": source()}
    )
    assert kept[0]["review_status"] == "ai_cross_source_verified"
    assert kept[0]["human_expert_reviewed"] is False


def test_graph_unsupported_needs_specific_verdict_kind() -> None:
    raw = {
        "sample_id": "g1",
        "subject": "stall",
        "predicate": "CAUSES",
        "object": "failure",
        "label": "unsupported",
        "polarity": "positive",
        "certainty": "finding",
        "qualifiers": {},
        "evidence": [anchor()],
        "rationale": "No stall",
        "author": "agent:author",
    }
    with pytest.raises(builder.AnnotationError, match="Unsupported must distinguish"):
        builder.compile_samples([raw], {"D1": source()}, set())
    raw["unsupported_type"] = "contradicted"
    assert builder.compile_samples([raw], {"D1": source()}, set())[0]["label"] == "unsupported"


@pytest.mark.parametrize("file_kind", ["raw", "blocks", "canonical"])
def test_portable_source_fingerprints_reject_tampering(tmp_path: Path, file_kind: str) -> None:
    text = "Evidence at 12 kg."
    raw_path = tmp_path / "raw.pdf"
    text_path = tmp_path / "canonical.txt"
    block_path = tmp_path / "blocks.jsonl"
    raw_path.write_bytes(b"original raw fixture")
    text_path.write_text(text, encoding="utf-8")
    builder.write_jsonl(
        block_path,
        [{"canonical_start": 0, "canonical_end": len(text), "text": text}],
    )
    catalog = tmp_path / "catalog.json"
    builder.write_json(
        catalog,
        {
            "documents": [
                {
                    "document_id": "D1",
                    "family_id": "F1",
                    "processed_path": str(text_path),
                    "blocks_path": str(block_path),
                    "raw_path": str(raw_path),
                    "canonical_sha256": builder.digest(text_path.read_bytes()),
                    "blocks_sha256": builder.digest(block_path.read_bytes()),
                    "raw_sha256": builder.digest(raw_path.read_bytes()),
                }
            ]
        },
    )
    assert builder.load_sources(catalog)["D1"]["text"] == text
    target = {"raw": raw_path, "blocks": block_path, "canonical": text_path}[file_kind]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(builder.AnnotationError, match=r"hash mismatch|fingerprint mismatch"):
        builder.load_sources(catalog)


@pytest.mark.parametrize("variant", ["unknown", "translation", "scenario_paraphrase"])
def test_invalid_variant_or_missing_parent_is_rejected(variant: str) -> None:
    with pytest.raises(builder.AnnotationError, match="query variant/parent"):
        builder.compile_samples([query(query_variant_type=variant)], {"D1": source()}, set())


@pytest.mark.parametrize("bad", [None, "object", "polarity"])
def test_negation_pair_preserves_the_same_claim_scope(bad: str | None) -> None:
    parent = {
        "sample_id": "g1",
        "subject": "fuel ice",
        "predicate": "CAUSES",
        "object": "blockage",
        "label": "supported",
        "polarity": "positive",
        "certainty": "conditional",
        "qualifiers": {"conditions": ["Fuel below freezing"]},
        "evidence": [anchor()],
        "rationale": "Explicit case",
        "author": "agent:author",
        "graph_variant_type": "native",
        "claim_group_id": "g1",
    }
    child = {
        **parent,
        "sample_id": "g2",
        "label": "unsupported",
        "polarity": "negative",
        "unsupported_type": "contradicted",
        "paired_claim_of": "g1",
        "graph_variant_type": "negation_stress",
    }
    assert len(builder.compile_samples([parent, child], {"D1": source()}, set())) == 2
    if bad is None:
        child["paired_claim_of"] = "missing"
    elif bad == "object":
        child["object"] = "another phenomenon"
    else:
        child["polarity"] = "positive"
    with pytest.raises(builder.AnnotationError, match="paired negation claim"):
        builder.compile_samples([parent, child], {"D1": source()}, set())
