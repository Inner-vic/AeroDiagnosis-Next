from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from aerodiagnosis.domain.diagnosis import EvidenceItem
from aerodiagnosis.domain.evidence_bundle import EvidenceBundle


@pytest.fixture
def pilot(monkeypatch: pytest.MonkeyPatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "api_pilot_test", scripts / "run_thesis_api_pilot.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def item(pilot, identity: str, text: str, start: int, doc: str = "d", version: str = "v"):
    return EvidenceItem(
        evidence_id=identity * 64,
        source_kind="document_chunk",
        source_ref=identity,
        document_id=doc,
        version_id=version,
        content_hash=pilot.builder.digest(text.encode()),
        excerpt=text,
        locator={"start": start, "end": start + len(text)},
        score=0,
    )


def bundle(*items):
    return EvidenceBundle(bundle_id="b" * 64, query="What does the source support?", evidence=items)


def claim(quote: str, ids: list[str]):
    return {
        "claims": [
            {
                "claim_id": "c1",
                "text": "a claim",
                "evidence_ids": ids,
                "support": "supported",
                "support_quote": quote,
            }
        ]
    }


def test_cross_chunk_quote_is_checked_only_against_continuous_snapshot(pilot) -> None:
    left = item(pilot, "1", "deposits reduce ", 0)
    right = item(pilot, "2", "reduce efficiency", 9)
    checked = pilot.validate_claim_quotes(
        bundle(left, right), claim("deposits reduce efficiency", ["1" * 64, "2" * 64])
    )
    assert len(checked["accepted"]) == 1
    report = pilot.assemble_traceable_report(bundle(left, right), checked)
    assert len(report["source_statements"]) == 1 and len(report["citations"]) == 2


@pytest.mark.parametrize("change", ["document", "version", "gap", "inconsistent_overlap"])
def test_no_join_across_sources_revisions_gaps_or_changed_text(pilot, change: str) -> None:
    left = item(pilot, "1", "deposits reduce ", 0)
    right = item(pilot, "2", "reduce efficiency", 9)
    if change == "document":
        right = right.model_copy(update={"document_id": "other"})
    if change == "version":
        right = right.model_copy(update={"version_id": "other"})
    if change == "gap":
        right = item(pilot, "2", "reduce efficiency", 20)
    if change == "inconsistent_overlap":
        right = item(pilot, "2", "REDUCE efficiency", 9)
    checked = pilot.validate_claim_quotes(
        bundle(left, right), claim("deposits reduce efficiency", ["1" * 64, "2" * 64])
    )
    assert not checked["accepted"] and checked["rejected"]


def test_verifier_cannot_drop_claims_or_change_them_silently(pilot) -> None:
    source = item(pilot, "1", "deposits reduce efficiency", 0)
    proposed = [{"claim_id": "c1", "text": "a claim"}, {"claim_id": "c2", "text": "another claim"}]
    checked = pilot.validate_claim_quotes(
        bundle(source), claim("deposits reduce efficiency", ["1" * 64]), proposed
    )
    assert checked["rejected"][0]["reasons"] == ["verifier_omitted_claim"]
    proposed[0]["text"] = "original statement"
    checked = pilot.validate_claim_quotes(
        bundle(source), claim("deposits reduce efficiency", ["1" * 64]), proposed
    )
    assert not checked["accepted"]
    assert "verifier_changed_claim_text" in checked["rejected"][0]["reasons"]
    report = pilot.assemble_traceable_report(bundle(source), checked)
    assert not report["source_statements"]
    assert all(r["status"] == "unverified" and "text" not in r for r in report["unresolved_claims"])


def test_bundle_rejects_duplicate_or_conflicting_support(pilot) -> None:
    source = item(pilot, "1", "deposits reduce efficiency", 0)
    with pytest.raises(ValidationError, match="unique"):
        bundle(source, source)
    with pytest.raises(ValidationError, match="Incompatible"):
        bundle(source.model_copy(update={"applicability_status": "conflict"}))
