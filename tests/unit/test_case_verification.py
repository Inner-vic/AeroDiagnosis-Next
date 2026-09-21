from __future__ import annotations

from pathlib import Path

import pytest

from aerodiagnosis.adapters.api.routers import verify_case
from aerodiagnosis.adapters.api.schemas import CaseVerificationRequest
from aerodiagnosis.adapters.persistence import SQLiteCaseStore
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.ports import CaseRecord


def _settings(tmp_path: Path) -> RuntimeSettings:
    return RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "runtime.db",
    )


def test_verified_case_records_accuracy_and_bumps_version(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    application = bootstrap(settings)
    SQLiteCaseStore(settings.database_path).upsert(
        CaseRecord("case-1", 1, "Compressor stall caused EGT rise.")
    )

    updated = application.record_case_verification.execute(
        case_id="case-1",
        outcome="correct",
        actual_cause="Compressor stall",
        actual_fault_ids=("stall",),
        notes="Confirmed by borescope.",
    )

    assert updated.version == 2
    verification = updated.attributes["verification"]
    assert verification["outcome"] == "correct"
    assert verification["accuracy"] == 100
    assert verification["actual_cause"] == "Compressor stall"


def test_case_verification_rejects_unknown_case(tmp_path: Path) -> None:
    application = bootstrap(_settings(tmp_path))

    with pytest.raises(KeyError, match="unknown case"):
        application.record_case_verification.execute(
            case_id="missing",
            outcome="correct",
        )


def test_case_verification_route_maps_to_application_use_case(tmp_path: Path) -> None:
    application = bootstrap(_settings(tmp_path))
    SQLiteCaseStore(_settings(tmp_path).database_path).upsert(
        CaseRecord("case-1", 1, "Compressor stall caused EGT rise.")
    )

    class FakeApplication:
        record_case_verification = application.record_case_verification

    response = verify_case(
        "case-1",
        CaseVerificationRequest(
            outcome="correct",
            actual_cause="Compressor stall",
            actual_fault_ids=("stall",),
            notes="Confirmed.",
        ),
        FakeApplication(),  # type: ignore[arg-type]
    )

    assert response.case_id == "case-1"
    assert response.version == 2
    assert response.attributes["verification"]["outcome"] == "correct"
