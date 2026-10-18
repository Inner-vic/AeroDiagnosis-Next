"""Question-bound atomic evaluation requirements and exact source annotations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuestionSlot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    slot_id: str = Field(min_length=1)
    requirement: str = Field(min_length=1)


class AtomicFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fact_key: str = Field(min_length=1)
    slot_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    text: str = Field(min_length=1)
    certainty: Literal["finding", "possible", "undetermined"]
    scope: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    allowed_alternative_texts: tuple[str, ...] = ()


class AtomicAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    question_slots: tuple[QuestionSlot, ...] = Field(min_length=1)
    required_facts: tuple[AtomicFact, ...] = Field(min_length=1)
    supplementary_facts: tuple[AtomicFact, ...] = ()

    @model_validator(mode="after")
    def question_contract(self) -> Self:
        slots = [s.slot_id for s in self.question_slots]
        facts = [f.fact_key for f in self.required_facts + self.supplementary_facts]
        if len(slots) != len(set(slots)) or len(facts) != len(set(facts)):
            raise ValueError("Duplicate slot or fact identity")
        if set(slots) != {f.slot_id for f in self.required_facts}:
            raise ValueError("Every required fact must address exactly one explicit question slot")
        if any(f.slot_id != "supplementary" for f in self.supplementary_facts):
            raise ValueError("Supplementary facts cannot participate in the required score")
        if any(
            f.source_end - f.source_start != len(f.source_quote)
            for f in self.required_facts + self.supplementary_facts
        ):
            raise ValueError("Source quote extent mismatch")
        return self

    def verify_source(self, text: str) -> None:
        import hashlib

        sha = hashlib.sha256(text.encode()).hexdigest()
        for fact in self.required_facts + self.supplementary_facts:
            if (
                sha != fact.source_sha256
                or text[fact.source_start : fact.source_end] != fact.source_quote
            ):
                raise ValueError("Source hash or exact quote mismatch")
