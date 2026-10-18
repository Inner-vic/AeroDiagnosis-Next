from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from aerodiagnosis.application.knowledge_graph_extractor import KnowledgeGraphExtractor
from aerodiagnosis.ingestion.parsers import PlainTextParser

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/compare_knowledge_pipelines.py"
spec = importlib.util.spec_from_file_location("knowledge_pipeline_comparison_test", MODULE_PATH)
assert spec and spec.loader
comparison = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = comparison
spec.loader.exec_module(comparison)


def test_cut_metrics_validate_exact_offsets_and_full_non_whitespace_coverage() -> None:
    text = "# Engine\n\n" + "Synthetic sentence. " * 70 + "\n\n[PDF_PAGE 2]\nSecond page."
    parsed = PlainTextParser(strategy="structure", chunk_size=100, overlap=10).parse(text.encode())
    result = comparison.cut_metrics(text, parsed.chunks)
    assert result["non_whitespace_coverage"] == 1
    assert result["invalid_locator_indices"] == []
    assert result["fallback_chunks"] > 0
    assert result["page_marker_count"] == 1
    assert result["max_characters"] <= 100


def test_historical_baseline_is_pinned_and_revision_collision_is_visible(tmp_path: Path) -> None:
    module, source_hash = comparison.frozen_baseline(tmp_path)
    assert len(source_hash) == 64
    assert (tmp_path / "lexical_v1_frozen.py").is_file()
    row = comparison.compare_text(
        "example",
        "Compressor fouling causes EGT rise.",
        module.KnowledgeGraphExtractor(),
        comparison.aliases(module),
        node_origins=defaultdict(dict),
        edge_origins=defaultdict(dict),
    )
    old = row["graphs"]["fixed/lexical_v1"]
    new = row["graphs"]["fixed/lexical_v2"]
    assert old["same_content_revision_identity_probe"]["shared_node_ids"] > 0
    assert new["same_content_revision_identity_probe"]["shared_node_ids"] == 0
    assert new["saved_provenance_located_edges"] == new["edge_count"]
    assert old["saved_provenance_located_edges"] == 0


def test_graph_proofs_refer_to_canonical_exact_sentence_spans() -> None:
    text = "  Compressor fouling causes EGT rise.\n"
    parsed = PlainTextParser(strategy="structure").parse(text.encode())
    nodes, edges = KnowledgeGraphExtractor().extract(
        source_ref="doc",
        version_id="v",
        chunks=tuple(chunk.content for chunk in parsed.chunks),
    )
    records = comparison.claim_records(text, parsed.chunks, nodes, edges)
    assert records and all(row["saved_provenance_located"] for row in records)
    for row in records:
        assert all(text[proof["start"] : proof["end"]] == proof["text"] for proof in row["proofs"])


def candidate(**updates: object) -> dict:
    return {
        "candidate_id": "candidate-1",
        "review_status": "machine_candidate",
        "source_document_id": "example",
        "subject": "compressor fouling",
        "predicate": "causes",
        "object": "high EGT",
        "source_locator": {},
        **updates,
    }


def test_candidate_mapping_inverts_causes_and_does_not_claim_gold() -> None:
    row = comparison.candidate_audit(candidate(), "Compressor fouling causes high EGT.")
    assert row["ontology_mapping"]["source_concepts"] == ["symptom:egt_high"]
    assert row["ontology_mapping"]["target_concepts"] == ["cause:fouling"]
    assert row["pair_evaluable"]
    assert not row["is_gold"]
    assert row["evidence_candidates"][0]["supports_relation"] == "unreviewed"


def test_candidates_outside_ontology_or_missing_source_are_not_scored_as_misses() -> None:
    row = comparison.candidate_audit(candidate(predicate="provides"), "Fouling provides a context.")
    assert row["mapping_status"] == "predicate_outside_current_ontology"
    assert not row["pair_evaluable"]
    missing = comparison.candidate_audit(candidate(), None)
    assert missing["localization_status"] == "missing_document"
    assert not missing["pair_evaluable"]


def test_candidate_page_scope_only_returns_original_page_text() -> None:
    text = (
        "[PDF_PAGE 1]\nCompressor fouling causes high EGT.\n"
        "[PDF_PAGE 2]\nCompressor fouling changes high EGT."
    )
    row = comparison.candidate_audit(candidate(source_locator={"page": 2}), text)
    assert row["localization_status"] == "page_marker_scope_located"
    assert row["evidence_candidates"]
    assert all(
        snippet["start"] > text.index("[PDF_PAGE 2]") for snippet in row["evidence_candidates"]
    )


def frozen_fixture(tmp_path: Path, *, refreshed: bool = False) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    payload = b"Compressor fouling causes EGT rise.\n"
    (raw / "example.txt").write_bytes(payload)
    text = tmp_path / "processed.txt"
    text.write_bytes(payload)
    expected_hash = "0" * 64 if refreshed else comparison.digest(payload)
    artifact = {
        "name": "example.txt",
        "sha256": comparison.digest(payload),
        "bytes": len(payload),
        "expected_sha256": expected_hash,
        "expected_bytes": len(payload),
        "url": "https://example.com/source.txt",
    }
    manifest = {
        "dataset_id": "example",
        "version": "1",
        "documents": [
            {
                "document_id": "example",
                "artifacts": [
                    {"name": "example.txt", "sha256": expected_hash, "bytes": len(payload)},
                ],
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    snapshot = {
        "dataset_id": "example-frozen",
        "source_manifest_sha256": comparison.digest(manifest_path.read_bytes()),
        "documents": [
            {
                "document_id": "example",
                "processed_path": "processed.txt",
                "sha256": comparison.digest(payload),
                "cohort": "refreshed" if refreshed else "historical_exact",
                "artifacts": [artifact],
            }
        ],
        "unavailable_document_ids": ["missing"],
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    return manifest_path, snapshot_path


@pytest.mark.parametrize("refreshed", [False, True])
def test_frozen_snapshot_accepts_only_actual_pinned_assets_and_discloses_cohort(
    tmp_path: Path,
    refreshed: bool,
) -> None:
    manifest, snapshot = frozen_fixture(tmp_path, refreshed=refreshed)
    output = tmp_path / "report.json"
    first = comparison.run(tmp_path, output, manifest, snapshot)
    second = comparison.run(tmp_path, output, manifest, snapshot)
    assert first == second
    assert first["status"] == "complete"
    assert first["snapshot_inventory"][0]["artifacts"][0]["historical_hash_match"] is not refreshed
    assert first["unavailable_document_ids"] == ["missing"]
    assert len(first["documents"]) == 1
    (tmp_path / "processed.txt").write_text("altered", encoding="utf-8")
    failed = comparison.run(tmp_path, output, manifest, snapshot)
    assert failed["status"] == "incomplete_frozen_snapshot"
    assert "documents" not in failed


def test_missing_raw_inventory_returns_incomplete_report_without_downloading(
    tmp_path: Path,
) -> None:
    manifest, _snapshot = frozen_fixture(tmp_path)
    (tmp_path / "raw/example.txt").unlink()
    result = comparison.run(tmp_path, tmp_path / "report.json", manifest)
    assert result["status"] == "incomplete_raw_assets"
    assert result["raw_inventory"][0]["status"] == "missing"


def test_compact_summary_keeps_missing_source_and_mapping_dimensions_separate(
    tmp_path: Path,
) -> None:
    manifest, snapshot = frozen_fixture(tmp_path, refreshed=True)
    report = comparison.run(tmp_path, tmp_path / "report.json", manifest, snapshot)
    summary = comparison.summarize_report(report)
    assert summary["total"]["document_count"] == 1
    assert summary["cohorts"]["refreshed"]["document_count"] == 1
    assert summary["total"]["cutting"]["structure"]["non_whitespace_coverage"] == 1
    graph = summary["total"]["graphs"]["structure/lexical_v2"]
    assert graph["saved_provenance_located_edges"] == graph["edge_count"]
    assert graph["synthetic_revision_shared_edge_ids"] == 0
    audit = summary["candidate_audit"]
    assert audit["missing_source_count"] == audit["candidate_count"]
    assert audit["available_source_count"] == 0
    assert audit["mapping_counts_available_sources"] == {}
    assert sum(audit["mapping_counts_all"].values()) == audit["candidate_count"]
    assert not audit["is_gold"]
    assert audit["exclusive_status_counts"] == {"missing_source": audit["candidate_count"]}


@pytest.mark.parametrize("predicate", ["may_indicate", "checks"])
def test_indications_and_inspected_targets_are_not_projected_to_causal_or_check_claims(
    predicate: str,
) -> None:
    row = comparison.candidate_audit(
        candidate(predicate=predicate), "Compressor fouling causes high EGT."
    )
    assert row["mapping_status"] == "predicate_outside_current_ontology"
    assert "ontology_mapping" not in row
    assert not row["pair_evaluable"]
