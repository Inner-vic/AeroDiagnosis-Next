"""Deterministic scoring for frozen AeroDiagnosis evaluation datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from aerodiagnosis.domain import DiagnosisReport, DiagnosisReportStatus


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=3)
    expected_status: DiagnosisReportStatus
    required_claim_terms: tuple[str, ...] = ()
    required_source_kinds: tuple[str, ...] = ()


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    cases: tuple[EvaluationCase, ...] = Field(min_length=1)

    @property
    def content_hash(self) -> str:
        payload = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode()).hexdigest()


class CaseScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    report_found: bool
    status_match: bool
    claim_term_recall: float = Field(ge=0, le=1)
    source_kind_recall: float = Field(ge=0, le=1)
    claims_grounded: bool
    passed: bool


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    dataset_version: str
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(ge=1)
    pass_rate: float = Field(ge=0, le=1)
    mean_claim_term_recall: float = Field(ge=0, le=1)
    mean_source_kind_recall: float = Field(ge=0, le=1)
    scores: tuple[CaseScore, ...]


def _recall(required: tuple[str, ...], observed: set[str] | str) -> float:
    if not required:
        return 1.0
    if isinstance(observed, str):
        normalized = observed.casefold()
        hits = sum(term.casefold() in normalized for term in required)
    else:
        normalized_set = {value.casefold() for value in observed}
        hits = sum(term.casefold() in normalized_set for term in required)
    return hits / len(required)


def evaluate(
    dataset: EvaluationDataset,
    reports: Sequence[DiagnosisReport],
) -> EvaluationSummary:
    by_question = {report.question: report for report in reports}
    scores = []
    for case in dataset.cases:
        report = by_question.get(case.question)
        if report is None:
            scores.append(
                CaseScore(
                    case_id=case.case_id,
                    report_found=False,
                    status_match=False,
                    claim_term_recall=0,
                    source_kind_recall=0,
                    claims_grounded=False,
                    passed=False,
                )
            )
            continue
        evidence_ids = {item.evidence_id for item in report.evidence}
        grounded = all(
            evidence_id in evidence_ids
            for claim in report.claims
            for evidence_id in claim.evidence_ids
        )
        claim_text = " ".join((report.summary or "", *(claim.statement for claim in report.claims)))
        term_recall = _recall(case.required_claim_terms, claim_text)
        source_recall = _recall(
            case.required_source_kinds,
            {item.source_kind for item in report.evidence},
        )
        status_match = report.status is case.expected_status
        scores.append(
            CaseScore(
                case_id=case.case_id,
                report_found=True,
                status_match=status_match,
                claim_term_recall=term_recall,
                source_kind_recall=source_recall,
                claims_grounded=grounded,
                passed=status_match and grounded and term_recall == 1 and source_recall == 1,
            )
        )
    count = len(scores)
    return EvaluationSummary(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        dataset_hash=dataset.content_hash,
        case_count=count,
        pass_rate=sum(score.passed for score in scores) / count,
        mean_claim_term_recall=sum(score.claim_term_recall for score in scores) / count,
        mean_source_kind_recall=sum(score.source_kind_recall for score in scores) / count,
        scores=tuple(scores),
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aerodiagnosis-evaluate",
        description="Score saved diagnosis reports against a versioned frozen dataset.",
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("reports", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        dataset = EvaluationDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
        report_payload = json.loads(args.reports.read_text(encoding="utf-8"))
        reports = tuple(DiagnosisReport.model_validate(item) for item in report_payload)
        summary = evaluate(dataset, reports)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(summary.model_dump_json())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
