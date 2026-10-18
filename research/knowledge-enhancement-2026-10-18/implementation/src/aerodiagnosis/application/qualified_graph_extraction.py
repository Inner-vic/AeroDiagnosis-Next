"""G2 structured candidates and G3 necessary exact-source/qualifier checks.

Passing these checks proves source locations and minimum qualifier constraints,
not relation entailment, causal direction, correct negation scope or completeness.
No page-level/document-level applicability is inherited. No API is called here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from aerodiagnosis.domain.qualified_claim import (
    CandidateBatch,
    ExtractionValidation,
    QualifiedCandidate,
    QualifiedClaim,
    RejectedCandidate,
    SourceUnit,
    SpanEvidence,
)

_NEGATION = re.compile(r"\b(?:not|no|never|cannot|without)\b|不|无|未|并非|不能", re.IGNORECASE)
_HYPOTHESIS = re.compile(
    r"\b(?:may|might|could|possible|possibly|probable|probably|likely|suspect\w*|believ\w*|seemed)\b|可能|疑似|未确定",
    re.IGNORECASE,
)
_CONDITIONAL = re.compile(
    r"\b(?:if|when|provided|unless|can)\b|若|如果|当|只要|条件", re.IGNORECASE
)
_REPORTED = re.compile(r"\b(?:report\w*|stated|observed|recorded)\b|报告|记录|观察", re.IGNORECASE)


def extraction_schema() -> dict[str, Any]:
    return CandidateBatch.model_json_schema()


def extraction_instruction() -> str:
    return (
        "Extract all source-supported assertions within the declared predicate schema from this "
        "single source unit. Preserve original entity surface forms, causal direction, entity "
        "roles, negation, uncertainty and conditions. Return JSON matching the supplied schema. "
        "evidence_quote and every qualifier quote must be exact substrings including raw line "
        "breaks. Do not translate, normalize, repair or invent quoted source text. "
        "Use role='unknown' and empty qualifier arrays/null for unstated conditions. Known roles "
        "require role_quote; each qualifier value must occur literally in its quote. "
        "Do not copy metadata/page headings/model/phase/revision into a claim unless that value "
        "is explicitly evidenced inside that claim's cited quote. "
        "Finding cannot strengthen a possible, suspected or merely reported association. "
        "A source does not support every same-sentence entity pair. If unclear, omit the claim. "
        "Empty claims is valid. Candidate extraction is not a human/expert gold annotation."
    )


def parse_candidates(
    payload: str | Mapping[str, Any],
) -> tuple[tuple[QualifiedCandidate, ...], tuple[RejectedCandidate, ...]]:
    """G2 schema-only; preserve every invalid candidate with rejection reasons."""
    try:
        value = json.loads(payload) if isinstance(payload, str) else dict(payload)
    except (ValueError, TypeError):
        return (), (
            RejectedCandidate(
                candidate_index=-1,
                candidate={"raw_payload": str(payload)},
                reasons=("invalid_json",),
            ),
        )
    if (
        not isinstance(value, dict)
        or set(value) != {"claims"}
        or not isinstance(value["claims"], list)
    ):
        return (), (
            RejectedCandidate(
                candidate_index=-1,
                candidate={"raw_payload": value},
                reasons=("invalid_batch_schema",),
            ),
        )
    accepted, rejected = [], []
    for index, row in enumerate(value["claims"]):
        try:
            accepted.append(QualifiedCandidate.model_validate(row))
        except ValidationError as exc:
            rejected.append(
                RejectedCandidate(
                    candidate_index=index,
                    candidate=row if isinstance(row, dict) else {"raw_candidate": row},
                    reasons=tuple(
                        "schema_invalid:" + ".".join(str(part) for part in error["loc"])
                        for error in exc.errors(include_input=False)
                    ),
                )
            )
    return tuple(accepted), tuple(rejected)


def _span(unit: SourceUnit, text: str, start: int) -> SpanEvidence:
    return SpanEvidence(
        document_id=unit.document_id,
        canonical_sha256=unit.canonical_sha256,
        canonical_start=unit.canonical_start + start,
        canonical_end=unit.canonical_start + start + len(text),
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _unique(text: str, needle: str, *, entity: bool = False) -> int | None:
    pattern = re.escape(needle)
    if entity and needle.isascii():
        if needle[0].isalnum():
            pattern = r"(?<!\w)" + pattern
        if needle[-1].isalnum():
            pattern += r"(?!\w)"
    matches = list(re.finditer(pattern, text))
    return matches[0].start() if len(matches) == 1 else None


def validate_candidates(
    unit: SourceUnit,
    candidates: tuple[QualifiedCandidate, ...],
) -> ExtractionValidation:
    accepted, rejected = [], []
    seen = set()
    for index, candidate in enumerate(candidates):
        reasons = []
        evidence_start = _unique(unit.text, candidate.evidence_quote)
        if evidence_start is None:
            rejected.append(
                RejectedCandidate(
                    candidate_index=index,
                    candidate=candidate.model_dump(mode="json"),
                    reasons=("evidence_quote_not_unique_exact_source_substring",),
                )
            )
            continue
        quote = candidate.evidence_quote
        qualifier_evidence: dict[str, tuple[SpanEvidence, ...]] = {}
        entity_spans = []
        for name, entity in (("subject", candidate.subject), ("object", candidate.object)):
            start = _unique(quote, entity.surface, entity=True)
            if start is None:
                reasons.append(name + "_surface_not_unique_in_cited_evidence")
            else:
                entity_spans.append(_span(unit, entity.surface, evidence_start + start))
            if entity.role != "unknown":
                role_start = _unique(quote, entity.role_quote or "")
                if not entity.role_quote or role_start is None:
                    reasons.append(name + "_known_role_missing_exact_evidence")
                else:
                    qualifier_evidence[name + "_role"] = (
                        _span(unit, entity.role_quote, evidence_start + role_start),
                    )
        qualifiers = candidate.qualifiers
        for name in ("engine_models", "flight_phases", "conditions", "revision"):
            field = getattr(qualifiers, name)
            values = (
                (field,)
                if name == "revision" and field is not None
                else (() if field is None else field)
            )
            spans = []
            for value in values:
                start = _unique(quote, value.quote)
                if start is None or _unique(value.quote, value.value) is None:
                    reasons.append(name + "_value_or_quote_not_unique_in_cited_evidence")
                else:
                    spans.append(_span(unit, value.quote, evidence_start + start))
            qualifier_evidence[name] = tuple(spans)
        for name in ("polarity_quote", "certainty_quote"):
            qualifier_quote = getattr(qualifiers, name)
            if qualifier_quote is not None:
                start = _unique(quote, qualifier_quote)
                if start is None:
                    reasons.append(name + "_not_unique_in_cited_evidence")
                else:
                    qualifier_evidence[name] = (
                        _span(unit, qualifier_quote, evidence_start + start),
                    )
        if candidate.polarity == "negative" and (
            not qualifiers.polarity_quote or not _NEGATION.search(qualifiers.polarity_quote)
        ):
            reasons.append("negative_polarity_missing_explicit_negation_evidence")
        certainty_quote = qualifiers.certainty_quote or ""
        if candidate.certainty == "finding" and _HYPOTHESIS.search(quote):
            reasons.append("finding_strengthens_source_hypothesis")
        if candidate.certainty == "hypothesis" and not _HYPOTHESIS.search(certainty_quote):
            reasons.append("hypothesis_missing_uncertainty_evidence")
        if candidate.certainty == "conditional" and not (
            qualifiers.conditions or _CONDITIONAL.search(certainty_quote)
        ):
            reasons.append("conditional_missing_condition_evidence")
        if candidate.certainty == "reported" and not (
            _REPORTED.search(certainty_quote)
            or unit.source_kind in {"service_difficulty_records", "accident_report"}
        ):
            reasons.append("reported_missing_reporting_context")
        payload = candidate.model_dump(mode="json")
        identity = hashlib.sha256(
            json.dumps(
                {
                    "unit_id": unit.unit_id,
                    "source_sha256": unit.canonical_sha256,
                    "source_start": unit.canonical_start + evidence_start,
                    "candidate": payload,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        if identity in seen:
            reasons.append("duplicate_candidate")
        seen.add(identity)
        if reasons:
            rejected.append(
                RejectedCandidate(
                    candidate_index=index,
                    candidate=payload,
                    reasons=tuple(reasons),
                )
            )
            continue
        accepted.append(
            QualifiedClaim(
                claim_id=identity,
                unit_id=unit.unit_id,
                subject=candidate.subject,
                predicate=candidate.predicate,
                object=candidate.object,
                polarity=candidate.polarity,
                certainty=candidate.certainty,
                qualifiers=qualifiers,
                evidence=_span(unit, quote, evidence_start),
                subject_evidence=entity_spans[0],
                object_evidence=entity_spans[1],
                qualifier_evidence=qualifier_evidence,
                source_locator=unit.source_locator,
            )
        )
    return ExtractionValidation(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        input_count=len(candidates),
    )
