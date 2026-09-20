from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import SecretStr

from aerodiagnosis.adapters.api import create_app
from aerodiagnosis.adapters.persistence import (
    UnsupportedBackendError,
    create_external_sync,
    create_graph_store,
    create_vector_store,
)
from aerodiagnosis.config import RuntimeSettings


def _settings(tmp_path: Path, **updates: str) -> RuntimeSettings:
    baseline = RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "runtime.db",
    )
    return baseline.model_copy(update=updates)


def test_default_factories_select_embedded_backends(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    assert create_vector_store(settings).backend_name == "sqlite_hashing"
    assert create_graph_store(settings).backend_name == "sqlite_graph"


def test_optional_backends_are_local_first_replicas(tmp_path: Path) -> None:
    vector = create_vector_store(_settings(tmp_path, vector_backend="chroma_http"))
    assert vector.backend_name == "chroma_http_cdc"
    with pytest.raises(UnsupportedBackendError, match="NEO4J_PASSWORD"):
        create_graph_store(_settings(tmp_path, graph_backend="neo4j"))
    graph = create_graph_store(
        _settings(
            tmp_path,
            graph_backend="neo4j",
            neo4j_password=SecretStr("test-password"),
        )
    )
    assert graph.backend_name == "neo4j_cdc"


def test_external_primary_selects_direct_external_stores(tmp_path: Path) -> None:
    vector = create_vector_store(
        _settings(
            tmp_path,
            vector_backend="chroma_http",
            external_store_mode="external-primary",
        )
    )
    graph = create_graph_store(
        _settings(
            tmp_path,
            graph_backend="neo4j",
            external_store_mode="external-primary",
            neo4j_password=SecretStr("test-password"),
        )
    )

    assert vector.backend_name == "chroma_http_hashing"
    assert graph.backend_name == "neo4j"


def test_external_primary_disables_cdc_sync(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedBackendError, match="external-primary"):
        create_external_sync(
            _settings(
                tmp_path,
                vector_backend="chroma_http",
                graph_backend="neo4j",
                external_store_mode="external-primary",
                neo4j_password=SecretStr("test-password"),
            )
        )


def test_api_surface_is_honest_about_current_application_status(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    paths = set(app.openapi()["paths"])

    assert {
        "/api/health",
        "/api/system",
        "/api/documents",
        "/api/evidence/search",
        "/api/diagnoses",
        "/api/diagnoses/{run_id}/resume",
        "/api/sessions",
        "/api/sessions/{session_id}/messages",
        "/api/sessions/{session_id}",
    } <= paths
    assert "/docs" in {getattr(route, "path", None) for route in app.routes}
    assert app.description == "Evidence-grounded diagnostic application runtime."


def test_api_lifespan_initializes_embedded_stores(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            state = app.state.application.get_runtime_status.execute()
            assert state.vector_backend == "sqlite_hashing"
            assert state.graph_backend == "sqlite_graph"

    asyncio.run(exercise())
