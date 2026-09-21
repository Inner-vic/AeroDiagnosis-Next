from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aerodiagnosis.domain import (
    DiagnosisReport,
    DiagnosisReportStatus,
    EvidenceClaim,
    EvidenceItem,
    ToolExecution,
    ToolExecutionStatus,
)
from aerodiagnosis.generation_quality import (
    evaluate_generation_quality,
    judge_generation_quality,
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


class FakeLanguageModel:
    @property
    def identity(self) -> str:
        return "fake-judge"

    def complete_json(self, task: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        assert task == "judge_diagnosis_quality"
        assert payload["deterministic_metrics"]["answer_grounding"] == 1.0
        return {
            "faithfulness": 0.95,
            "context_precision": 1.0,
            "context_recall": 1.0,
            "issues": [],
            "human_review_required": False,
        }


def test_generation_quality_scores_grounding_and_context() -> None:
    report = _report()

    metrics = evaluate_generation_quality(
        report,
        relevant_evidence_ids={report.evidence[0].evidence_id},
    )

    assert metrics.answer_grounding == 1.0
    assert metrics.faithfulness > 0.9
    assert metrics.context_precision == 1.0
    assert metrics.context_recall == 1.0
    assert metrics.human_review_required is False


def test_llm_judge_receives_deterministic_metrics_and_validates_verdict() -> None:
    verdict = judge_generation_quality(
        _report(),
        FakeLanguageModel(),
        relevant_evidence_ids={_report().evidence[0].evidence_id},
    )

    assert verdict.faithfulness == 0.95
    assert verdict.human_review_required is False
