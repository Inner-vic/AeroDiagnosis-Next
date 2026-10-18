"""Versioned source context; missing applicability is never implicitly generic."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DocumentApplicability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    engine_models: tuple[str, ...] = ()
    flight_phases: tuple[str, ...] = ()
    scope_status: Literal["unknown", "known", "generic"] = "unknown"

    @field_validator("engine_models", "flight_phases")
    @classmethod
    def normalize_identifiers(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(
            " ".join(unicodedata.normalize("NFKC", value).split()).casefold() for value in values
        )
        if any(not value for value in cleaned):
            raise ValueError("applicability identifiers must not be blank")
        return tuple(sorted(set(cleaned)))

    @model_validator(mode="after")
    def consistent_scope(self) -> Self:
        limited = bool(self.engine_models or self.flight_phases)
        if self.scope_status == "known" and not limited:
            raise ValueError("known scope requires at least one explicit limitation")
        if self.scope_status != "known" and limited:
            raise ValueError("unknown and generic scope must not carry limitations")
        return self


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    source_url: str | None = None
    source_type: str = "unknown"
    source_revision: str | None = None
    language: str | None = None
    license_or_basis: str | None = None
    applicability: DocumentApplicability = Field(default_factory=DocumentApplicability)

    @field_validator("source_url", "source_type", "source_revision", "language", "license_or_basis")
    @classmethod
    def clean_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("metadata strings must not be blank")
        return cleaned

    @field_validator("source_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is not None:
            parts = urlsplit(value)
            if parts.scheme not in {"http", "https"} or not parts.netloc:
                raise ValueError("source_url must be an absolute HTTP(S) URL")
        return value

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def bind_version_id(self, content_version_id: str) -> str:
        """Preserve legacy identity for the default metadata, version every explicit change."""
        if self == DocumentMetadata():
            return content_version_id
        return hashlib.sha256(f"{content_version_id}\0{self.canonical_json()}".encode()).hexdigest()
