"""Explicit applicability constraints; no inference of a missing engine model."""

from __future__ import annotations

import unicodedata
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ApplicabilityStatus = Literal["not_checked", "matched", "generic", "unknown", "conflict"]


def normalize_scope(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


class KnowledgeQueryContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("document_id")
    @classmethod
    def preserve_document_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("document_id must not be blank")
        return value.strip()

    engine_model: str | None = Field(default=None, min_length=1, max_length=120)
    flight_phase: str | None = Field(default=None, min_length=1, max_length=120)
    source_revision: str | None = Field(default=None, min_length=1, max_length=120)
    require_known: bool = False

    @field_validator("engine_model", "flight_phase", "source_revision")
    @classmethod
    def normalize_constraints(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = normalize_scope(value)
        if not cleaned:
            raise ValueError("scope constraints must not be blank")
        return cleaned

    @property
    def has_constraints(self) -> bool:
        return any(
            value is not None
            for value in (self.engine_model, self.flight_phase, self.source_revision)
        )

    @model_validator(mode="after")
    def require_constraints_for_strict_query(self) -> Self:
        if self.require_known and not self.has_constraints:
            raise ValueError("require_known needs an engine model, flight phase or source revision")
        return self
