from __future__ import annotations

import pytest
from pydantic import ValidationError

from aerodiagnosis.domain import (
    DiagnosisReport,
    DiagnosisReportStatus,
    EvidenceClaim,
    EvidenceItem,
    ParameterObservation,
    ToolExecution,
    ToolExecutionStatus,
)


def _evidence() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="a" * 64,
        source_kind="document_chunk",
        source_ref="chunk-1",
        document_id="document",
        version_id="b" * 64,
        content_hash="c" * 64,
        excerpt="Compressor stall may cause an exhaust gas temperature rise.",
        locator={"kind": "character_range", "coordinates": {"start": 0, "end": 60}},
        score=0.8,
    )


def _trace() -> tuple[ToolExecution, ...]:
    return (
        ToolExecution(
            tool_name="search_manual_chunks",
            query="EGT rise",
            status=ToolExecutionStatus.OK,
            evidence_count=1,
        ),
    )


def test_evidence_ready_report_accepts_verifier_approved_bound_claims() -> None:
    evidence = _evidence()
    report = DiagnosisReport(
        run_id="run-1",
        session_id="session-1",
        snapshot_id="d" * 64,
        status=DiagnosisReportStatus.EVIDENCE_READY,
        question="What can cause EGT rise?",
        summary="Compressor stall is a supported candidate.",
        claims=(
            EvidenceClaim(
                statement="Compressor stall can cause EGT rise.",
                evidence_ids=(evidence.evidence_id,),
                confidence=0.8,
            ),
        ),
        evidence=(evidence,),
        tool_executions=_trace(),
        attempted_queries=("EGT rise",),
        workflow_version="workflow@1",
    )

    assert report.claims[0].verification_method == "llm_verifier@1"


def test_report_rejects_claim_referencing_evidence_outside_snapshot() -> None:
    with pytest.raises(ValidationError, match="outside the run snapshot"):
        DiagnosisReport(
            run_id="run-1",
            session_id="session-1",
            snapshot_id="d" * 64,
            status=DiagnosisReportStatus.EVIDENCE_READY,
            question="What can cause EGT rise?",
            summary="Candidate",
            claims=(
                EvidenceClaim(
                    statement="Unsupported candidate",
                    evidence_ids=("f" * 64,),
                    confidence=0.8,
                ),
            ),
            evidence=(_evidence(),),
            tool_executions=_trace(),
            attempted_queries=("EGT rise",),
            workflow_version="workflow@1",
        )


def test_insufficient_report_must_refuse_without_claims() -> None:
    report = DiagnosisReport(
        run_id="run-1",
        session_id="session-1",
        snapshot_id="d" * 64,
        status=DiagnosisReportStatus.INSUFFICIENT_EVIDENCE,
        question="What can cause EGT rise?",
        tool_executions=_trace(),
        attempted_queries=("EGT rise", "EGT rise fault cause"),
        refusal_reason="No evidence met the threshold.",
        workflow_version="workflow@1",
    )

    assert report.claims == ()
    payload = report.model_dump(mode="python")
    payload["refusal_reason"] = None
    with pytest.raises(ValidationError, match="must refuse"):
        DiagnosisReport.model_validate(payload)


def test_parameter_observation_requires_an_ordered_reference_range() -> None:
    with pytest.raises(ValidationError, match="expected_max"):
        ParameterObservation(name="EGT", value=700, expected_min=800, expected_max=700)


def test_tool_trace_error_shape_is_fail_closed() -> None:
    with pytest.raises(ValidationError, match="require an error"):
        ToolExecution(
            tool_name="search_manual_chunks",
            query="EGT",
            status=ToolExecutionStatus.ERROR,
            evidence_count=0,
        )
