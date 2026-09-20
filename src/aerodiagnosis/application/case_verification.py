from __future__ import annotations

from datetime import UTC, datetime

from aerodiagnosis.ports import CaseRecord, CaseStore

_ACCURACY = {
    "correct": 100,
    "partial": 50,
    "wrong": 0,
}


class RecordCaseVerification:
    def __init__(self, case_store: CaseStore) -> None:
        self._case_store = case_store

    def execute(
        self,
        *,
        case_id: str,
        outcome: str,
        actual_cause: str = "",
        actual_fault_ids: tuple[str, ...] = (),
        notes: str = "",
    ) -> CaseRecord:
        if outcome not in _ACCURACY:
            raise ValueError("outcome must be one of correct, partial or wrong")
        current = self._case_store.get_case(case_id)
        if current is None:
            raise KeyError(f"unknown case: {case_id}")
        verification = {
            "outcome": outcome,
            "accuracy": _ACCURACY[outcome],
            "actual_cause": actual_cause,
            "actual_fault_ids": list(actual_fault_ids),
            "notes": notes,
            "verified_at": datetime.now(UTC).isoformat(),
        }
        attributes = dict(current.attributes)
        attributes["verification"] = verification
        updated = CaseRecord(
            case_id=current.case_id,
            version=current.version + 1,
            summary=current.summary,
            attributes=attributes,
        )
        self._case_store.upsert(updated)
        return updated
