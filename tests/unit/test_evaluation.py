from __future__ import annotations

from aerodiagnosis.domain import (
    DiagnosisReport,
    DiagnosisReportStatus,
    EvidenceClaim,
    EvidenceItem,
    ToolExecution,
    ToolExecutionStatus,
)
from aerodiagnosis.evaluation import EvaluationCase, EvaluationDataset, evaluate


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
