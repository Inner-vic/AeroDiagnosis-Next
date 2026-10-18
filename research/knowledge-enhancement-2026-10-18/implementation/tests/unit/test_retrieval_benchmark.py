"""Protocol and metric regressions; synthetic fixtures are not evaluation gold."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from aerodiagnosis.retrieval_benchmark import (
    BenchmarkDataset,
    BenchmarkHit,
    BenchmarkQuery,
    MatrixConfig,
    ReviewedSpan,
    paired_bootstrap,
    percentile,
    prepare_dataset,
    score_query,
    sha256,
    summarize_scores,
    validate_run,
    verify_corpus,
)


def query(**updates: object) -> BenchmarkQuery:
    return BenchmarkQuery.model_validate(
        {
            "query_id": "q1",
            "query": "fixture query",
            "qrels": {"a": 2, "b": 1},
            **updates,
        }
    )


def test_chunk_rank_metrics_deduplicate_document_gain_without_removing_rank() -> None:
    hits = [BenchmarkHit(key, str(i)) for i, key in enumerate(["x", "a", "a", "b", "a"])]
    score = score_query(query(), hits, frozenset({"a", "b", "x"}))
    assert score.doc_recall_at_5 == 1
    assert score.mrr_at_5 == 0.5
    expected = (3 / math.log2(3) + 1 / math.log2(5)) / (3 + 1 / math.log2(3))
    assert score.ndcg_at_5 == pytest.approx(expected)
    assert score.reviewed_span_coverage is None
    assert score.no_answer_success is None


def test_missing_positive_source_is_excluded_even_if_another_positive_is_retrieved() -> None:
    score = score_query(query(), [BenchmarkHit("a", "one")], frozenset({"a"}))
    assert score.status == "missing_source"
    assert score.missing_positive_document_ids == ("b",)
    assert score.doc_recall_at_5 is None
    assert score.ndcg_at_5 is None


def test_no_answer_only_for_explicit_human_reviewed_unanswerable_judgments() -> None:
    unknown = score_query(query(), [], frozenset({"a", "b"}))
    assert unknown.doc_recall_at_5 == 0
    assert unknown.no_answer_success is None
    unanswerable = query(
        qrels={},
        answerability="unanswerable",
        review_status="human_reviewed",
        reviewer="unit-test-reviewer",
    )
    no_hits = score_query(unanswerable, [], frozenset({"a", "b"}))
    assert no_hits.no_answer_success == 1
    assert no_hits.doc_recall_at_5 is None
    assert (
        score_query(unanswerable, [BenchmarkHit("a", "one")], frozenset({"a"})).no_answer_success
        == 0
    )
    with pytest.raises(ValidationError, match="answerability labels require"):
        query(answerability="unanswerable")


@pytest.mark.parametrize("grade", [True, 1.5, "2", -1, 4])
def test_relevance_grades_are_strict_integers(grade: object) -> None:
    with pytest.raises(ValidationError):
        query(qrels={"a": grade})


def test_reviewed_span_union_and_child_anchor_are_scored_separately() -> None:
    spans = tuple(
        ReviewedSpan(
            document_id="a",
            start=start,
            end=end,
            text_sha256="0" * 64,
            review_status="human_reviewed",
            reviewer="test",
        )
        for start, end in [(0, 10), (5, 15)]
    )
    reviewed = query(qrels={"a": 2}, evidence_spans=spans)
    hits = [BenchmarkHit("a", "parent", 0, 15, True, 5, 10)] * 2
    score = score_query(reviewed, hits, frozenset({"a"}))
    assert score.reviewed_span_coverage == 1
    assert score.reviewed_anchor_span_coverage == pytest.approx(1 / 3)
    unlocated = score_query(reviewed, [BenchmarkHit("a", "bad", 0, 15)], frozenset({"a"}))
    assert unlocated.reviewed_span_coverage == 0


def test_paired_cluster_bootstrap_uses_only_paired_queries_and_fixed_seed() -> None:
    left = {"a": 0.2, "b": 0.8, "left_only": 1}
    right = {"a": 0.7, "b": 0.8, "right_only": 1}
    kwargs = {"seed": 42, "samples": 300, "groups": {"a": "family1", "b": "family1"}}
    first = paired_bootstrap(left, right, **kwargs)
    assert first == paired_bootstrap(left, right, **kwargs)
    assert first["paired_query_count"] == 2
    assert first["bootstrap_cluster_count"] == 1
    assert first["mean_delta_b_minus_a"] == pytest.approx(0.25)
    assert first["ci95"] == pytest.approx([0.25, 0.25])
    assert (first["wins_b"], first["ties"], first["losses_b"]) == (1, 1, 0)


def test_percentile_and_none_metric_denominators_are_explicit() -> None:
    assert percentile([10, 20, 30], 0.5) == 20
    assert percentile([10, 20, 30], 0.95) == 29
    scored = score_query(query(), [], frozenset({"a", "b"}))
    missing = score_query(query(), [], frozenset({"a"}))
    summary = summarize_scores([scored, missing], [10, 20, 30])
    assert summary["doc_recall_at_5"] == 0
    assert summary["doc_recall_at_5_query_count"] == 1
    assert summary["reviewed_span_coverage_query_count"] == 0
    assert summary["reviewed_span_coverage"] is None
    assert summary["status_counts"] == {"scored": 1, "missing_source": 1}


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    text = b"A real file for synthetic unit-test protocol checks.\n"
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_bytes(text)
    (tmp_path / "a.txt").write_bytes(text)
    snapshot = {
        "dataset_id": "unit-test",
        "documents": [
            {
                "document_id": "a",
                "processed_path": "a.txt",
                "sha256": sha256(text),
                "cohort": "historical_exact",
                "artifacts": [{"name": "a.txt", "sha256": sha256(text), "bytes": len(text)}],
            }
        ],
        "unavailable_document_ids": ["b"],
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "unit-test query",
                "qrels": {"a": 2},
                "review_status": "human_reviewed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return snapshot_path, queries, tmp_path / "annotation.json"


def test_prepare_never_promotes_candidates_or_fabricates_spans_and_splits(tmp_path: Path) -> None:
    snapshot, queries, output = fixture(tmp_path)
    dataset = prepare_dataset(snapshot, queries, output)
    assert dataset.queries[0].review_status == "machine_candidate"
    assert dataset.queries[0].answerability == "unknown"
    assert dataset.queries[0].evidence_spans == ()
    assert dataset.documents[0].split == "unassigned"
    assert not dataset.documents[0].family_reviewed
    assert prepare_dataset(snapshot, queries, output).content_hash == dataset.content_hash
    texts, _ = verify_corpus(dataset, output)
    assert texts["a"].startswith("A real file")
    (tmp_path / "raw/a.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="raw artifact changed"):
        verify_corpus(dataset, output)


def test_reviewed_annotation_must_match_exact_frozen_text(tmp_path: Path) -> None:
    snapshot, queries, output = fixture(tmp_path)
    dataset = prepare_dataset(snapshot, queries, output)
    span = ReviewedSpan(
        document_id="a",
        start=0,
        end=4,
        text_sha256="0" * 64,
        review_status="human_reviewed",
        reviewer="test",
    )
    annotated = dataset.model_copy(
        update={"queries": (query(qrels={"a": 2}, evidence_spans=(span,)),)}
    )
    with pytest.raises(ValueError, match="span/hash mismatch"):
        verify_corpus(annotated, output)


def test_document_family_and_query_judgment_splits_cannot_leak(tmp_path: Path) -> None:
    snapshot, queries, output = fixture(tmp_path)
    dataset = prepare_dataset(snapshot, queries, output)
    document = dataset.documents[0].model_dump()
    first = {**document, "family_id": "same-family", "split": "train"}
    second = {**document, "document_id": "b", "family_id": "same-family", "split": "test"}
    payload = {**dataset.model_dump(), "documents": [first, second], "unavailable_document_ids": []}
    with pytest.raises(ValidationError, match="family crosses"):
        BenchmarkDataset.model_validate(payload)
    payload["documents"] = [first]
    payload["queries"] = [query(qrels={"a": 2}, split="test").model_dump()]
    with pytest.raises(ValidationError, match="query judgments cross"):
        BenchmarkDataset.model_validate(payload)


def test_heldout_policy_must_be_frozen_and_reviewed_before_running(tmp_path: Path) -> None:
    snapshot, queries, output = fixture(tmp_path)
    dataset = prepare_dataset(snapshot, queries, output)
    config = MatrixConfig(purpose="heldout_test", query_split="test")
    with pytest.raises(ValueError, match="policies must be frozen"):
        validate_run(dataset, config)
    doc = dataset.documents[0].model_copy(
        update={"family_reviewed": True, "family_id": "test-family", "split": "test"}
    )
    reviewed = query(qrels={"a": 2}, review_status="human_reviewed", reviewer="test", split="test")
    good = BenchmarkDataset.model_validate(
        {
            **dataset.model_dump(),
            "documents": [doc.model_dump()],
            "queries": [reviewed.model_dump()],
            "test_policy_sha256": config.policy_hash,
        }
    )
    validate_run(good, config)
    changed = config.model_copy(update={"rrf_k": 10})
    with pytest.raises(ValueError, match="reviewed test queries require"):
        validate_run(good, changed)
    with pytest.raises(ValidationError, match="development cannot"):
        MatrixConfig(purpose="development", query_split="test")


def test_production_matrix_preserves_candidate_boundary_and_quality_determinism(
    tmp_path: Path,
) -> None:
    from aerodiagnosis.retrieval_benchmark import run_matrix

    snapshot, queries, output = fixture(tmp_path)
    dataset = prepare_dataset(snapshot, queries, output)
    config = MatrixConfig(
        chunk_strategies=("hierarchical",),
        retrieval_strategies=("sparse_only",),
        context_expansion=(False, True),
        rerankers=("identity",),
        repeats=2,
        bootstrap_samples=100,
    )
    first = run_matrix(dataset, output, config)
    second = run_matrix(dataset, output, config)
    assert first["quality_hash"] == second["quality_hash"]
    assert first["publication_ready"] is False
    assert first["status"] == "partial_candidate_regression"
    assert len(first["policies"]) == 2
    for policy in first["policies"]:
        summary = policy["summary"]
        assert summary["doc_recall_at_5"] == 1
        assert summary["latency_sample_count"] == 2
        assert summary["reviewed_span_coverage"] is None
        assert summary["no_answer_success"] is None
        for result in policy["results"]:
            assert all(hit["source_located"] for hit in result["hits"])


def test_crlf_asset_hash_and_lf_annotation_coordinates_are_both_verified(tmp_path: Path) -> None:
    snapshot_path, queries, output = fixture(tmp_path)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload = b"First line.\r\nSecond line.\r\n"
    (tmp_path / "a.txt").write_bytes(payload)
    (tmp_path / "raw/a.txt").write_bytes(payload)
    doc = snapshot["documents"][0]
    doc["sha256"] = sha256(payload)
    doc["artifacts"][0].update(sha256=sha256(payload), bytes=len(payload))
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    dataset = prepare_dataset(snapshot_path, queries, output)
    texts, _ = verify_corpus(dataset, output)
    assert texts["a"] == "First line.\nSecond line.\n"
    assert dataset.documents[0].sha256 != dataset.documents[0].canonical_sha256
    bad = dataset.documents[0].model_copy(update={"canonical_sha256": "0" * 64})
    altered = dataset.model_copy(update={"documents": (bad,)})
    with pytest.raises(ValueError, match="canonical LF text hash changed"):
        verify_corpus(altered, output)


def test_candidate_run_label_cannot_bypass_reviewed_test_policy(tmp_path: Path) -> None:
    snapshot, queries, output = fixture(tmp_path)
    dataset = prepare_dataset(snapshot, queries, output)
    doc = dataset.documents[0].model_copy(
        update={"family_reviewed": True, "family_id": "test-family", "split": "test"}
    )
    reviewed = query(qrels={"a": 2}, review_status="human_reviewed", reviewer="test", split="test")
    config = MatrixConfig(purpose="candidate_regression", query_split="all")
    payload = {
        **dataset.model_dump(),
        "documents": [doc.model_dump()],
        "queries": [reviewed.model_dump()],
    }
    missing_freeze = BenchmarkDataset.model_validate(payload)
    with pytest.raises(ValueError, match="reviewed test queries require"):
        validate_run(missing_freeze, config)
    frozen = BenchmarkDataset.model_validate({**payload, "test_policy_sha256": config.policy_hash})
    validate_run(frozen, config)
    alternate = MatrixConfig(purpose="candidate_regression", query_split="all", rrf_k=10)
    with pytest.raises(ValueError, match="reviewed test queries require"):
        validate_run(frozen, alternate)


def test_code_hash_receipt_works_in_installed_package_without_repository_src(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aerodiagnosis import retrieval_benchmark as module

    keys = module._source_hashes().keys()
    package_root = tmp_path / "site-packages" / "aerodiagnosis"
    payload = b"installed-package-source-fixture\n"
    for key in keys:
        path = package_root / key.removeprefix("src/aerodiagnosis/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    monkeypatch.setattr(module, "__file__", str(package_root / "retrieval_benchmark.py"))
    assert module._source_hashes() == {key: sha256(payload) for key in keys}
    assert not (tmp_path / "src").exists()
