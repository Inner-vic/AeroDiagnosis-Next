from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from aerodiagnosis.adapters.persistence import create_embedding_provider, create_vector_store
from aerodiagnosis.config import RuntimeSettings


def _settings(tmp_path: Path, **updates: object) -> RuntimeSettings:
    return RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "runtime.db",
        **updates,
    )


def test_openai_compatible_embedding_provider_requires_complete_configuration(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="base_url, model and api_key"):
        _settings(tmp_path, embedding_backend="openai_compatible")


def test_chroma_default_embeddings_require_external_primary_chroma(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="vector_backend=chroma_http"):
        _settings(tmp_path, embedding_backend="chroma_default")

    with pytest.raises(ValidationError, match="external-primary"):
        _settings(
            tmp_path,
            vector_backend="chroma_http",
            embedding_backend="chroma_default",
        )


def test_chroma_default_external_primary_uses_semantic_server_identity(
    tmp_path: Path,
) -> None:
    settings = _settings(
        tmp_path,
        vector_backend="chroma_http",
        external_store_mode="external-primary",
        embedding_backend="chroma_default",
    )

    store = create_vector_store(settings)

    assert store.backend_name == "chroma_http_chroma_server_default"
    assert store.embedding_identity == "chroma_server_default"


def test_openai_compatible_embedding_provider_is_constructed_from_settings(
    tmp_path: Path,
) -> None:
    settings = _settings(
        tmp_path,
        embedding_backend="openai_compatible",
        embedding_base_url="https://embeddings.example/v1",
        embedding_model="embedding-model",
        embedding_api_key=SecretStr("embedding-secret"),
        embedding_dimensions=384,
    )

    provider = create_embedding_provider(settings)

    assert provider.name == "openai_compatible:embedding-model:384"
    assert provider.dimensions == 384
