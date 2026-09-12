from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from aerodiagnosis.adapters.api.routers import _provider_model, run_diagnosis
from aerodiagnosis.adapters.api.schemas import ProviderConfiguration, RunDiagnosisRequest
from aerodiagnosis.adapters.llm import OpenAICompatibleLanguageModel
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings


def test_api_uses_per_request_provider_without_persisting_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "runtime.db",
    )
    application = bootstrap(settings)
    application.ingest_document.execute(
        display_name="manual.txt",
        content=b"Compressor efficiency loss can increase EGT.",
    )
    session_id = application.conversation_sessions.create()

    def complete_json(
        _self: OpenAICompatibleLanguageModel,
        task: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if task == "plan_evidence":
            return {
                "search_queries": ["compressor efficiency EGT"],
                "tools": ["search_manual_chunks"],
                "rationale": "Search the active manual.",
            }
        if task == "generate_diagnosis":
            evidence_id = payload["evidence"][0]["evidence_id"]
            return {
                "summary": "Inspect compressor performance.",
                "claims": [
                    {
                        "statement": "Compressor efficiency loss may increase EGT.",
                        "evidence_ids": [evidence_id],
                        "confidence": 0.8,
                    }
                ],
            }
        if task == "verify_diagnosis":
            return {
                "sufficient": True,
                "supported_claim_indexes": [0],
                "issues": [],
                "corrected_queries": [],
                "corrected_tools": [],
            }
        raise AssertionError(task)

    monkeypatch.setattr(OpenAICompatibleLanguageModel, "complete_json", complete_json)
    request = RunDiagnosisRequest(
        session_id=session_id,
        question="Why did EGT increase during this compressor event?",
        provider=ProviderConfiguration(
            base_url="https://provider.example/v1",
            api_key="credential-that-must-never-be-persisted",
            model="diagnostic-model",
        ),
    )

    report = run_diagnosis(request, application)

    assert report.status == "evidence_ready"
    assert b"credential-that-must-never-be-persisted" not in settings.database_path.read_bytes()


def test_api_can_use_private_local_default_provider(tmp_path: Path) -> None:
    settings = RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "runtime.db",
        default_llm_base_url="https://local-provider.example/v1",
        default_llm_model="demo-model",
        default_llm_api_key=SecretStr("private-local-key"),
    )
    application = bootstrap(settings)

    model = _provider_model(None, application)

    assert model.identity.startswith("openai_compatible:demo-model:")
    assert "private-local-key" not in repr(model.__dict__)
