"""Protect the migration bridge from returning to external-only storage."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SETTINGS = PROJECT_ROOT / "code" / "python" / "config" / "settings.py"
LEGACY_VECTOR = PROJECT_ROOT / "code" / "python" / "services" / "vector_store.py"
LEGACY_GRAPH = PROJECT_ROOT / "code" / "python" / "services" / "knowledge_graph.py"
COMPOSE = PROJECT_ROOT / "code" / "docker-compose.yml"


def _class_defaults(path: Path, class_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    target = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    defaults: dict[str, object] = {}
    for statement in target.body:
        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
            continue
        if isinstance(statement.value, ast.Constant):
            defaults[statement.target.id] = statement.value.value
    return defaults


def test_legacy_local_defaults_are_embedded_and_loopback_only() -> None:
    defaults = _class_defaults(LEGACY_SETTINGS, "Settings")

    assert defaults["vector_store_backend"] == "embedded"
    assert defaults["graph_store_backend"] == "sqlite"
    assert defaults["api_host"] == "127.0.0.1"
    assert defaults["neo4j_password"] == ""


def test_legacy_adapters_have_explicit_local_and_external_paths() -> None:
    vector_source = LEGACY_VECTOR.read_text(encoding="utf-8")
    graph_source = LEGACY_GRAPH.read_text(encoding="utf-8")

    assert "chromadb.PersistentClient" in vector_source
    assert "chromadb.HttpClient" in vector_source
    assert 'self._backend == "sqlite"' in graph_source
    assert 'self._backend != "neo4j"' in graph_source
    assert "Cypher administration is unavailable" in graph_source


def test_compose_explicitly_selects_external_adapters() -> None:
    compose = COMPOSE.read_text(encoding="utf-8")

    assert "VECTOR_STORE_BACKEND: chroma_http" in compose
    assert "GRAPH_STORE_BACKEND: neo4j" in compose
