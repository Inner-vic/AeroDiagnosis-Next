from __future__ import annotations

import pytest

from aerodiagnosis.domain import (
    DiagnosisReport,
    DiagnosisReportStatus,
    EvidenceClaim,
    EvidenceItem,
    FusedEvidenceHit,
    HybridRetrievalResult,
    RetrievalContribution,
    RetrievalRouteSummary,
    ToolExecution,
    ToolExecutionStatus,
)
from aerodiagnosis.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    evaluate,
    evaluate_retrieval,
)


def _report(question: str) -> DiagnosisReport:
    evidence = EvidenceItem(
        evidence_id="a" * 64,
        source_kind="document_chunk",
        source_ref="chunk-1",
        document_id="manual",
        version_id="v1",
        content_hash="b" * 64,
        excerpt="压气机效率下降会造成 EGT 上升。",
        locator={"kind": "chunk", "coordinates": {"chunk": 1}},
        score=0.9,
    )
    return DiagnosisReport(
        run_id="run-1",
        session_id="session-1",
        snapshot_id="c" * 64,
        status=DiagnosisReportStatus.EVIDENCE_READY,
        question=question,
        summary="应优先检查压气机。",
        claims=(
            EvidenceClaim(
                statement="压气机可能发生效率衰退。",
                evidence_ids=(evidence.evidence_id,),
                confidence=0.8,
            ),
        ),
        evidence=(evidence,),
        tool_executions=(
            ToolExecution(
                tool_name="search_manual_chunks",
                query=question,
                status=ToolExecutionStatus.OK,
                evidence_count=1,
            ),
        ),
        attempted_queries=(question,),
        workflow_version="test@1",
    )


def test_frozen_evaluation_scores_grounded_report_and_missing_case() -> None:
    question = "EGT 上升时检查什么?"
    dataset = EvaluationDataset(
        dataset_id="smoke",
        version="1",
        cases=(
            EvaluationCase(
                case_id="found",
                question=question,
                expected_status=DiagnosisReportStatus.EVIDENCE_READY,
                required_claim_terms=("压气机",),
                required_source_kinds=("document_chunk",),
            ),
            EvaluationCase(
                case_id="missing",
                question="另一个问题",
                expected_status=DiagnosisReportStatus.INSUFFICIENT_EVIDENCE,
            ),
        ),
    )

    summary = evaluate(dataset, (_report(question),))

    assert summary.dataset_hash == dataset.content_hash
    assert summary.pass_rate == 0.5
    assert summary.scores[0].passed is True
    assert summary.scores[1].report_found is False


def test_retrieval_evaluation_reports_standard_ranking_metrics() -> None:
    first = _report("EGT 上升时检查什么?").evidence[0]
    second = first.model_copy(
        update={
            "evidence_id": "d" * 64,
            "source_kind": "case",
            "source_ref": "case-1:1",
            "document_id": "case-1",
            "version_id": "1",
        }
    )
    result = HybridRetrievalResult(
        query="compressor EGT",
        algorithm="weighted_rrf@1",
        top_k=5,
        candidate_count=2,
        hits=(
            FusedEvidenceHit(
                rank=1,
                evidence=first,
                fused_score=0.9,
                selection_reason="route_coverage",
                contributions=(
                    RetrievalContribution(
                        route="manual",
                        route_rank=1,
                        raw_score=0.9,
                        normalized_score=0.9,
                        reciprocal_rank_score=1.0,
                        weight=1.0,
                    ),
                ),
            ),
            FusedEvidenceHit(
                rank=2,
                evidence=second,
                fused_score=0.8,
                selection_reason="route_coverage",
                contributions=(
                    RetrievalContribution(
                        route="case",
                        route_rank=1,
                        raw_score=0.8,
                        normalized_score=0.8,
                        reciprocal_rank_score=1.0,
                        weight=0.94,
                    ),
                ),
            ),
        ),
        routes=(
            RetrievalRouteSummary(
                route="manual", candidate_count=1, included_count=1, top_raw_score=0.9
            ),
            RetrievalRouteSummary(
                route="graph", candidate_count=0, included_count=0
            ),
            RetrievalRouteSummary(
                route="case", candidate_count=1, included_count=1, top_raw_score=0.8
            ),
        ),
    )

    metrics = evaluate_retrieval(result, {second.evidence_id, "e" * 64})

    assert metrics.precision_at_k == 0.2
    assert metrics.recall_at_k == 0.5
    assert metrics.reciprocal_rank == 0.5
    assert 0 < metrics.ndcg_at_k < 1
    assert metrics.source_coverage == pytest.approx(2 / 3)
