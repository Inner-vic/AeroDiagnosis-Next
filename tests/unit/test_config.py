from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aerodiagnosis.config import RuntimeSettings


def test_settings_default_to_project_local_embedded_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "AERODIAGNOSIS_RUNTIME_DIR",
        "AERODIAGNOSIS_DATABASE_PATH",
        "AERODIAGNOSIS_VECTOR_BACKEND",
        "AERODIAGNOSIS_GRAPH_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = RuntimeSettings.from_env(tmp_path)

    assert settings.runtime_dir == tmp_path / ".runtime"
    assert settings.database_path == tmp_path / ".runtime" / "data" / "aerodiagnosis.db"
    assert settings.vector_backend == "sqlite"
    assert settings.graph_backend == "sqlite"


def test_settings_reject_non_loopback_binding(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="non-loopback"):
        RuntimeSettings(
            runtime_dir=tmp_path,
            database_path=tmp_path / "database.db",
            api_host="0.0.0.0",
        )


def test_settings_default_to_local_first_external_store_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AERODIAGNOSIS_EXTERNAL_STORE_MODE", raising=False)
    settings = RuntimeSettings.from_env(tmp_path)
    assert settings.external_store_mode == "local-first"


def test_settings_read_external_primary_store_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AERODIAGNOSIS_EXTERNAL_STORE_MODE", "external-primary")
    settings = RuntimeSettings.from_env(tmp_path)
    assert settings.external_store_mode == "external-primary"


def test_settings_reject_unknown_external_store_mode(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="external_store_mode"):
        RuntimeSettings(
            runtime_dir=tmp_path,
            database_path=tmp_path / "runtime.db",
            external_store_mode="not-a-mode",
        )


def test_prepare_creates_runtime_and_database_parent(tmp_path: Path) -> None:
    settings = RuntimeSettings(
        runtime_dir=tmp_path / "runtime",
        database_path=tmp_path / "database" / "runtime.db",
    )

    settings.prepare()

    assert settings.runtime_dir.is_dir()
    assert settings.database_path.parent.is_dir()


def test_git_ignored_local_profile_configures_default_provider(tmp_path: Path) -> None:
    (tmp_path / ".env.local").write_text(
        "AERODIAGNOSIS_LLM_BASE_URL=https://provider.example/v1\n"
        "AERODIAGNOSIS_LLM_MODEL=demo-model\n"
        "AERODIAGNOSIS_LLM_API_KEY=local-secret\n",
        encoding="utf-8",
    )

    settings = RuntimeSettings.from_env(tmp_path)

    assert settings.default_provider_available is True
    assert settings.default_llm_model == "demo-model"
    assert settings.default_llm_api_key is not None
    assert settings.default_llm_api_key.get_secret_value() == "local-secret"
    assert "local-secret" not in repr(settings)


def test_local_profile_can_reference_separate_key_file(tmp_path: Path) -> None:
    key_file = tmp_path / "provider.key"
    key_file.write_text("file-secret", encoding="utf-8")
    (tmp_path / ".env.local").write_text(
        "AERODIAGNOSIS_LLM_BASE_URL=https://provider.example/v1\n"
        "AERODIAGNOSIS_LLM_MODEL=demo-model\n"
        f"AERODIAGNOSIS_LLM_API_KEY_FILE={key_file}\n",
        encoding="utf-8",
    )

    settings = RuntimeSettings.from_env(tmp_path)

    assert settings.default_provider_available is True
    assert settings.default_llm_api_key is not None
    assert settings.default_llm_api_key.get_secret_value() == "file-secret"
