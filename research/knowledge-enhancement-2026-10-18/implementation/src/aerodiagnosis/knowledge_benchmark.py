"""Offline, provenance-aware knowledge benchmark contracts and validation CLI.

Validation checks documentation completeness, not whether a human review occurred.
No labels are upgraded automatically and no diagnostic accuracy is calculated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aerodiagnosis.domain.document_metadata import DocumentApplicability


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceVersion(FrozenContract):
    document_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceSpan(SourceVersion):
    evidence_id: str | None = Field(default=None, min_length=1)
    locator: dict[str, str | int] = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        if self.end <= self.start or self.end - self.start != len(self.text):
            raise ValueError("span offsets must match the exact Unicode text length")
        return self


class ReviewRecord(FrozenContract):
    status: Literal["machine_candidate", "silver", "gold"] = "machine_candidate"
    annotator_ids: tuple[str, ...] = ()
    outcome: Literal["agreed", "adjudicated"] | None = None
    adjudicator_id: str | None = None
    expert_reviewer_id: str | None = None
    reviewed_at: datetime | None = None
    frozen_at: datetime | None = None

    @field_validator("annotator_ids")
    @classmethod
    def normalize_annotators(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(value.strip() for value in values)

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        identifiers = self.annotator_ids + tuple(
            value for value in (self.adjudicator_id, self.expert_reviewer_id) if value is not None
        )
        if any(not value.strip() for value in identifiers):
            raise ValueError("reviewer identifiers must not be blank")
        if len(set(self.annotator_ids)) != len(self.annotator_ids):
            raise ValueError("annotators must be distinct")
        if self.status in {"silver", "gold"} and (
            len(self.annotator_ids) < 2 or self.reviewed_at is None
        ):
            raise ValueError("silver/gold require two annotators and review time")
        if self.status in {"silver", "gold"} and (
            self.outcome is None or (self.outcome == "adjudicated" and self.adjudicator_id is None)
        ):
            raise ValueError("reviewed labels require agreement or documented adjudication")
        if self.status == "gold" and (self.expert_reviewer_id is None or self.frozen_at is None):
            raise ValueError("gold requires an expert reviewer and frozen timestamp")
        for value in (self.reviewed_at, self.frozen_at):
            if value is not None and value.utcoffset() is None:
                raise ValueError("review timestamps must include a timezone")
        if (
            self.reviewed_at is not None
            and self.frozen_at is not None
            and self.frozen_at < self.reviewed_at
        ):
            raise ValueError("freeze time must not precede review")
        return self


class RelationJudgment(FrozenContract):
    subject: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    object: str = Field(min_length=1)
    polarity: Literal["positive", "negative"] = "positive"
    certainty: Literal["asserted", "possible", "unknown"] = "unknown"
    applicability: DocumentApplicability = Field(default_factory=DocumentApplicability)
    source_span_index: int = Field(ge=0)


class KnowledgeBenchmarkItem(FrozenContract):
    annotation_id: str = Field(min_length=1)
    task: Literal["retrieval", "extraction", "parsing"]
    review: ReviewRecord = Field(default_factory=ReviewRecord)
    partition: Literal["development", "test"] = "development"
    query: str | None = None
    applicability: DocumentApplicability = Field(default_factory=DocumentApplicability)
    answerability: Literal["answerable", "unanswerable"] = "answerable"
    judgment_scope: Literal["document", "evidence"] = "document"
    judgment_complete: bool = False
    relevant_ids: tuple[str, ...] = ()
    inapplicable_ids: tuple[str, ...] = ()
    source_spans: tuple[SourceSpan, ...] = ()
    relations: tuple[RelationJudgment, ...] = ()
    reference: str | None = None

    @model_validator(mode="after")
    def consistent_item(self) -> Self:
        if self.task == "retrieval":
            if self.query is None or not self.query.strip():
                raise ValueError("retrieval needs a query")
            if self.answerability == "answerable" and not self.relevant_ids:
                raise ValueError("answerable retrieval needs relevant IDs")
            if self.answerability == "unanswerable" and self.relevant_ids:
                raise ValueError("unanswerable retrieval must not have relevant IDs")
        elif not self.source_spans:
            raise ValueError("parsing/extraction need exact source spans")
        if self.task == "parsing" and (self.reference is None or not self.reference.strip()):
            raise ValueError("parsing needs a reference transcription or structure")
        for values in (self.relevant_ids, self.inapplicable_ids):
            if len(set(values)) != len(values) or any(not value.strip() for value in values):
                raise ValueError("judgment IDs must be distinct and nonblank")
        if set(self.relevant_ids) & set(self.inapplicable_ids):
            raise ValueError("relevant and inapplicable IDs must be disjoint")
        if self.relations and self.task != "extraction":
            raise ValueError("relation judgments only belong to extraction")
        if any(relation.source_span_index >= len(self.source_spans) for relation in self.relations):
            raise ValueError("relation must point to an existing source span")
        if (
            self.review.status != "machine_candidate"
            and self.relevant_ids
            and not self.source_spans
        ):
            raise ValueError("reviewed relevant judgments need exact supporting source spans")
        if self.task == "retrieval" and self.review.status != "machine_candidate":
            supported_ids = {
                span.document_id if self.judgment_scope == "document" else span.evidence_id
                for span in self.source_spans
            }
            if not set(self.relevant_ids) <= supported_ids:
                raise ValueError("every reviewed relevant ID needs its supporting source span")
        return self


class KnowledgeBenchmark(FrozenContract):
    schema_version: Literal["1"] = "1"
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    judgment_protocol: str = Field(min_length=1)
    snapshot: tuple[SourceVersion, ...] = Field(min_length=1)
    items: tuple[KnowledgeBenchmarkItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        annotation_ids = [item.annotation_id for item in self.items]
        if len(set(annotation_ids)) != len(annotation_ids):
            raise ValueError("annotation IDs must be unique")
        keys = [(source.document_id, source.version_id) for source in self.snapshot]
        if len(set(keys)) != len(keys):
            raise ValueError("source versions must be unique")
        sources = {
            (source.document_id, source.version_id): source.source_hash for source in self.snapshot
        }
        for item in self.items:
            for span in item.source_spans:
                if sources.get((span.document_id, span.version_id)) != span.source_hash:
                    raise ValueError("source span does not match the frozen source snapshot")
            if item.judgment_scope == "document" and item.task == "retrieval":
                documents = {source.document_id for source in self.snapshot}
                if not set(item.relevant_ids + item.inapplicable_ids) <= documents:
                    raise ValueError("document judgments must reference snapshot documents")
        return self

    @property
    def content_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validation_summary(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "content_hash": self.content_hash,
            "snapshot_versions": len(self.snapshot),
            "item_count": len(self.items),
            "review_counts": dict(Counter(item.review.status for item in self.items)),
            "task_counts": dict(Counter(item.task for item in self.items)),
            "scope_counts": dict(Counter(item.judgment_scope for item in self.items)),
            "incomplete_judgments": sum(not item.judgment_complete for item in self.items),
            "unanswerable_queries": sum(
                item.task == "retrieval" and item.answerability == "unanswerable"
                for item in self.items
            ),
            "limitations": [
                "Contract validation does not verify that human review actually occurred.",
                "Incomplete judgments support recall over labeled relevant items only.",
                "Document and evidence judgments must be scored separately.",
                "Candidate labels do not establish diagnostic accuracy.",
            ],
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a frozen knowledge benchmark offline")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        benchmark = KnowledgeBenchmark.model_validate_json(
            args.path.read_text(encoding="utf-8-sig")
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(benchmark.validation_summary(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
