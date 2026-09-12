from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aerodiagnosis.adapters.api import create_app
from aerodiagnosis.adapters.persistence import (
    UnsupportedBackendError,
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


def test_optional_backends_fail_explicitly_when_adapter_is_absent(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedBackendError, match="chroma_http"):
        create_vector_store(_settings(tmp_path, vector_backend="chroma_http"))
    with pytest.raises(UnsupportedBackendError, match="neo4j"):
        create_graph_store(_settings(tmp_path, graph_backend="neo4j"))


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
