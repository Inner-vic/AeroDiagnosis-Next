from __future__ import annotations

import json
from pathlib import Path

from aerodiagnosis.diagnostic_evaluation import (
    DiagnosisEvalCase,
    evaluate_diagnosis,
    summarize_diagnostic_evaluation,
)
from aerodiagnosis.domain import (
    DiagnosisReport,
    DiagnosisReportStatus,
    EvidenceClaim,
    EvidenceItem,
    ToolExecution,
    ToolExecutionStatus,
)


def _report() -> DiagnosisReport:
    evidence = EvidenceItem(
        evidence_id="a" * 64,
        source_kind="document_chunk",
        source_ref="chunk-1",
        document_id="document",
        version_id="b" * 64,
        content_hash="c" * 64,
        excerpt="Compressor stall raises EGT. Inspect the compressor and clean blades.",
        locator={"kind": "character_range", "coordinates": {"start": 0, "end": 60}},
        score=0.9,
    )
    return DiagnosisReport(
        run_id="run-1",
        session_id="session-1",
        snapshot_id="d" * 64,
        status=DiagnosisReportStatus.EVIDENCE_READY,
        question="Why did EGT increase?",
        summary="Compressor stall is the likely cause; inspect the compressor.",
        claims=(
            EvidenceClaim(
                statement="Compressor stall can cause EGT rise.",
                evidence_ids=(evidence.evidence_id,),
                confidence=0.9,
            ),
        ),
        evidence=(evidence,),
        tool_executions=(
            ToolExecution(
                tool_name="search_manual_chunks",
                query="EGT rise",
                status=ToolExecutionStatus.OK,
                evidence_count=1,
            ),
        ),
        attempted_queries=("EGT rise",),
        workflow_version="workflow@1",
    )


def test_diagnostic_evaluation_scores_cause_and_solution_hits() -> None:
    case = DiagnosisEvalCase(
        case_id="case-1",
        question="Why did EGT increase?",
        difficulty="medium",
        expected_cause_keywords=("compressor stall",),
        expected_solution_keywords=("inspect the compressor",),
        min_causal_hops=2,
    )

    result = evaluate_diagnosis(_report(), case)

    assert result.cause_hit is True
    assert result.solution_hit is True
    assert result.cause_hit_rank == 1
    assert result.causal_complete is True
    assert result.has_evidence is True


def test_diagnostic_evaluation_summarizes_metrics() -> None:
    case = DiagnosisEvalCase(
        case_id="case-1",
        question="Why did EGT increase?",
        difficulty="medium",
        expected_cause_keywords=("compressor stall",),
        expected_solution_keywords=("inspect the compressor",),
    )

    summary = summarize_diagnostic_evaluation([(_report(), case)])

    assert summary["total_cases"] == 1
    assert summary["cause_accuracy"] == 1.0
    assert summary["solution_accuracy"] == 1.0
    assert summary["causal_completeness"] == 1.0
    assert summary["mrr"] == 1.0


def test_frozen_diagnostic_cases_are_valid() -> None:
    path = Path(__file__).resolve().parents[2] / "evaluation" / "diagnosis_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))

    assert len(cases) == 4
    assert all(case["case_id"].startswith("AERO-DIAG-") for case in cases)
    assert all(case["expected_cause_keywords"] for case in cases)
    assert all(case["expected_solution_keywords"] for case in cases)
