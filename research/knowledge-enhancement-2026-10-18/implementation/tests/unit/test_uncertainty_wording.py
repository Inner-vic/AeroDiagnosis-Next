import hashlib

import pytest

from aerodiagnosis.application.qualified_graph_extraction import validate_candidates
from aerodiagnosis.domain.qualified_claim import QualifiedCandidate, SourceUnit


@pytest.mark.parametrize("marker", ["likely", "probably", "probable", "possibly"])
def test_uncertain_cause_cannot_be_promoted_to_finding(marker: str) -> None:
    text = f"Deposits are a {marker} cause of vibration."
    unit = SourceUnit(
        unit_id="u",
        document_id="d",
        canonical_sha256=hashlib.sha256(text.encode()).hexdigest(),
        canonical_start=0,
        text=text,
    )
    base = dict(
        subject={"surface": "Deposits", "entity_type": "Cause"},
        predicate="CAUSES",
        object={"surface": "vibration", "entity_type": "Symptom"},
        polarity="positive",
        evidence_quote=text,
    )
    strengthened = QualifiedCandidate.model_validate({**base, "certainty": "finding"})
    result = validate_candidates(unit, (strengthened,))
    assert not result.accepted
    assert "finding_strengthens_source_hypothesis" in result.rejected[0].reasons
    qualified = QualifiedCandidate.model_validate(
        {**base, "certainty": "hypothesis", "qualifiers": {"certainty_quote": marker}}
    )
    assert len(validate_candidates(unit, (qualified,)).accepted) == 1
