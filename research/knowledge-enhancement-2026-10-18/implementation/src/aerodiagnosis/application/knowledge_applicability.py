"""Shared applicability policy for document, graph and case evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from aerodiagnosis.domain.document_metadata import DocumentMetadata
from aerodiagnosis.domain.knowledge_query import (
    ApplicabilityStatus,
    KnowledgeQueryContext,
    normalize_scope,
)

POLICY_ID = "knowledge-applicability@2"


def assess_applicability(
    metadata: Mapping[str, Any] | None,
    context: KnowledgeQueryContext,
) -> ApplicabilityStatus:
    if not context.has_constraints:
        return "not_checked"
    try:
        record = DocumentMetadata.model_validate(dict(metadata or {}))
    except ValidationError:
        return "unknown"
    revision_missing = False
    if context.source_revision is not None:
        if record.source_revision is None:
            revision_missing = True
        elif normalize_scope(record.source_revision) != context.source_revision:
            return "conflict"
    scope = record.applicability
    if scope.scope_status == "unknown":
        return "unknown"
    if scope.scope_status == "generic":
        return "unknown" if revision_missing else "generic"
    incomplete = revision_missing
    for query_value, allowed in (
        (context.engine_model, scope.engine_models),
        (context.flight_phase, scope.flight_phases),
    ):
        if query_value is None:
            # A source restriction cannot be confirmed without its query condition.
            if allowed:
                incomplete = True
            continue
        if not allowed:
            incomplete = True
        elif query_value not in {normalize_scope(value) for value in allowed}:
            return "conflict"
    return "unknown" if incomplete else "matched"


def permits(status: ApplicabilityStatus, context: KnowledgeQueryContext) -> bool:
    if status == "conflict":
        return False
    return not context.require_known or status == "matched"


def assess_assertion_applicability(
    properties: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    context: KnowledgeQueryContext,
) -> ApplicabilityStatus:
    """Source authority is retained; explicit assertion restrictions can narrow it."""
    status = assess_applicability(metadata, context)
    if status == "conflict":
        return status
    qualifiers = properties.get("qualifiers")
    if not isinstance(qualifiers, Mapping):
        return status
    missing = False
    for name, value in (
        ("engine_models", context.engine_model),
        ("flight_phases", context.flight_phase),
    ):
        declared = qualifiers.get(name, [])
        allowed = {
            normalize_scope(item["value"])
            for item in declared
            if isinstance(item, Mapping) and isinstance(item.get("value"), str)
        }
        if not allowed:
            continue
        if value is None:
            missing = True
        elif value not in allowed:
            return "conflict"
    revision = qualifiers.get("revision")
    if isinstance(revision, Mapping) and isinstance(revision.get("value"), str):
        if context.source_revision is None:
            missing = True
        elif normalize_scope(revision["value"]) != context.source_revision:
            return "conflict"
    if qualifiers.get("conditions"):
        missing = True  # Additional environmental conditions have not been supplied/verified.
    return "unknown" if missing else status
