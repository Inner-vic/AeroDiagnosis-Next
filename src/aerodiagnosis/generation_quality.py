from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aerodiagnosis.domain import DiagnosisReport
from aerodiagnosis.ports import LanguageModel


class GenerationQualityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_grounding: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    human_review_required: bool


class LLMJudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    faithfulness: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    issues: tuple[str, ...] = ()
    human_review_required: bool


def _terms(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    for sequence in re.findall(r"[\u3400-\u9fff]+", normalized):
        tokens.add(sequence)
        tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def _overlap(claim: str, evidence_text: str) -> float:
    claim_terms = _terms(claim)
    if not claim_terms:
        return 0.0
    evidence_terms = _terms(evidence_text)
    return len(claim_terms & evidence_terms) / len(claim_terms)


def evaluate_generation_quality(
    report: DiagnosisReport,
    relevant_evidence_ids: set[str] | frozenset[str] = frozenset(),
) -> GenerationQualityMetrics:
    """Compute deterministic RAGAS-style proxies before optional LLM judging."""

    evidence_by_id = {item.evidence_id: item for item in report.evidence}
    if not report.claims:
        return GenerationQualityMetrics(
            faithfulness=0.0,
            answer_grounding=0.0,
            context_precision=0.0,
            context_recall=0.0,
            human_review_required=True,
        )

    grounded_claims = 0
    support_scores: list[float] = []
    for claim in report.claims:
        if all(identifier in evidence_by_id for identifier in claim.evidence_ids):
            grounded_claims += 1
        evidence_text = " ".join(
            evidence_by_id[identifier].excerpt
            for identifier in claim.evidence_ids
            if identifier in evidence_by_id
        )
        support_scores.append(_overlap(claim.statement, evidence_text))

    answer_grounding = grounded_claims / len(report.claims)
    faithfulness = (
        answer_grounding * 0.9 + (sum(support_scores) / len(support_scores)) * 0.1
    )
    retrieved_ids = [item.evidence_id for item in report.evidence]
    relevant = set(relevant_evidence_ids)
    context_precision = (
        sum(identifier in relevant for identifier in retrieved_ids) / len(retrieved_ids)
        if retrieved_ids
        else 0.0
    )
    context_recall = (
        sum(identifier in retrieved_ids for identifier in relevant) / len(relevant)
        if relevant
        else 0.0
    )
    human_review_required = (
        faithfulness < 0.8 or answer_grounding < 1.0 or context_recall < 1.0
    )
    return GenerationQualityMetrics(
        faithfulness=round(min(1.0, faithfulness), 6),
        answer_grounding=answer_grounding,
        context_precision=context_precision,
        context_recall=context_recall,
        human_review_required=human_review_required,
    )


def judge_generation_quality(
    report: DiagnosisReport,
    language_model: LanguageModel,
    relevant_evidence_ids: set[str] | frozenset[str] = frozenset(),
) -> LLMJudgeVerdict:
    """Ask a constrained LLM judge to audit the deterministic RAGAS-style proxies."""

    payload: Mapping[str, Any] = {
        "question": report.question,
        "summary": report.summary,
        "claims": [
            claim.model_dump(mode="json")
            for claim in report.claims
        ],
        "evidence": [
            item.model_dump(mode="json")
            for item in report.evidence
        ],
        "relevant_evidence_ids": sorted(relevant_evidence_ids),
        "deterministic_metrics": evaluate_generation_quality(
            report, relevant_evidence_ids
        ).model_dump(mode="json"),
    }
    raw = language_model.complete_json("judge_diagnosis_quality", payload)
    return LLMJudgeVerdict.model_validate(raw)
