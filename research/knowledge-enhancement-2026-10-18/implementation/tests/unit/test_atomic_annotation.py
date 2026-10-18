import hashlib

import pytest
from pydantic import ValidationError

from aerodiagnosis.domain.atomic_annotation import AtomicAnnotation


def annotation():
    source = "The initiating cause could not be determined."
    fact = {
        "fact_key": "f1",
        "slot_id": "s1",
        "subject": "起始原因",
        "predicate": "未能确定",
        "object": "调查范围内的具体原因",
        "text": "起始原因未能确定。",
        "certainty": "undetermined",
        "scope": "本次调查",
        "source_quote": source,
        "source_start": 0,
        "source_end": len(source),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    return {
        "question_slots": [{"slot_id": "s1", "requirement": "起始原因是否确定"}],
        "required_facts": [fact],
        "supplementary_facts": [],
    }, source


def test_required_scope_cannot_add_unasked_slot_or_duplicate_fact():
    value, _source = annotation()
    value["required_facts"][0]["slot_id"] = "unasked"
    with pytest.raises(ValidationError, match="explicit question slot"):
        AtomicAnnotation.model_validate(value)
    value, _source = annotation()
    value["required_facts"].append(value["required_facts"][0])
    with pytest.raises(ValidationError, match="Duplicate"):
        AtomicAnnotation.model_validate(value)


def test_source_and_supplementary_identity_are_checked():
    value, source = annotation()
    model = AtomicAnnotation.model_validate(value)
    model.verify_source(source)
    with pytest.raises(ValueError, match="Source hash"):
        model.verify_source("The initiating cause was found.")
    value["supplementary_facts"] = [{**value["required_facts"][0], "fact_key": "extra"}]
    with pytest.raises(ValidationError, match="Supplementary"):
        AtomicAnnotation.model_validate(value)
