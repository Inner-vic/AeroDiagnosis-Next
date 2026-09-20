from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aerodiagnosis.domain import DiagnosisReport, DiagnosisReportStatus


@dataclass(frozen=True, slots=True)
class DiagnosisEvalCase:
    case_id: str
    question: str
    difficulty: str
    expected_cause_keywords: tuple[str, ...] = ()
    expected_solution_keywords: tuple[str, ...] = ()
    min_causal_hops: int = 1
    category: str = ""


@dataclass(frozen=True, slots=True)
class DiagnosisEvalResult:
    case_id: str
    difficulty: str
    cause_hit: bool
    solution_hit: bool
    cause_hit_rank: int
    causal_complete: bool
    has_evidence: bool
    has_prevention: bool
    response_time_ms: float


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    folded = text.casefold()
    return any(keyword.casefold() in folded for keyword in keywords)


def evaluate_diagnosis(report: DiagnosisReport, case: DiagnosisEvalCase) -> DiagnosisEvalResult:
    evidence_text = " ".join(item.excerpt for item in report.evidence)
    claims_text = " ".join(claim.statement for claim in report.claims)
    generated_text = " ".join((report.summary or "", evidence_text, claims_text))

    cause_hit = _contains_any(generated_text, case.expected_cause_keywords)
    solution_hit = _contains_any(generated_text, case.expected_solution_keywords)
    has_prevention = _contains_any(generated_text, ("prevent", "预防", "prevention"))

    cause_hit_rank = 999
    if cause_hit:
        for rank, item in enumerate(report.evidence, start=1):
            if _contains_any(item.excerpt, case.expected_cause_keywords):
                cause_hit_rank = rank
                break
        else:
            cause_hit_rank = max(len(report.evidence) + 1, 1)

    causal_complete = (
        report.status is DiagnosisReportStatus.EVIDENCE_READY
        and cause_hit
        and solution_hit
        and bool(report.claims)
    )

    return DiagnosisEvalResult(
        case_id=case.case_id,
        difficulty=case.difficulty,
        cause_hit=cause_hit,
        solution_hit=solution_hit,
        cause_hit_rank=cause_hit_rank,
        causal_complete=causal_complete,
        has_evidence=bool(report.evidence),
        has_prevention=has_prevention,
        response_time_ms=0.0,
    )


def summarize_diagnostic_evaluation(
    results: list[tuple[DiagnosisReport, DiagnosisEvalCase]],
) -> dict[str, Any]:
    evaluated = [evaluate_diagnosis(report, case) for report, case in results]
    total = len(evaluated)
    if total == 0:
        return {
            "total_cases": 0,
            "cause_accuracy": 0.0,
            "solution_accuracy": 0.0,
            "causal_completeness": 0.0,
            "mrr": 0.0,
        }

    reciprocal_ranks = [
        1.0 / result.cause_hit_rank
        for result in evaluated
        if result.cause_hit and result.cause_hit_rank < 999
    ]
    mrr = sum(reciprocal_ranks) / total if reciprocal_ranks else 0.0

    return {
        "total_cases": total,
        "cause_accuracy": sum(result.cause_hit for result in evaluated) / total,
        "solution_accuracy": sum(result.solution_hit for result in evaluated) / total,
        "causal_completeness": sum(result.causal_complete for result in evaluated) / total,
        "mrr": mrr,
    }
