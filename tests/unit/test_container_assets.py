from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_v3_container_runs_locked_package_as_non_root() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.13-slim-bookworm" in dockerfile
    assert "uv sync --frozen --no-dev --no-editable" in dockerfile
    assert "USER aerodiagnosis" in dockerfile
    assert "aerodiagnosis.adapters.api:app" in dockerfile
    assert "--host\", \"0.0.0.0" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "COPY . ." not in dockerfile


def test_v3_compose_is_local_only_and_persistent() -> None:
    compose = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert '"127.0.0.1:${AERODIAGNOSIS_PORT:-8080}:8080"' in compose
    assert "aerodiagnosis_runtime:/app/runtime" in compose
    assert "name: aerodiagnosis-runtime" in compose
    assert "AERODIAGNOSIS_VECTOR_BACKEND: sqlite" in compose
    assert "AERODIAGNOSIS_GRAPH_BACKEND: sqlite" in compose
    assert "read_only: true" in compose
    assert "no-new-privileges:true" in compose
    assert "neo4j:" not in compose
    assert "chromadb:" not in compose

    full = (PROJECT_ROOT / "compose.full.yaml").read_text(encoding="utf-8")
    assert "chromadb/chroma:1.5.9" in full
    assert "neo4j:5.26.30-community" in full
    assert 'command: ["aerodiagnosis-sync", "--forever", "--interval", "2"]' in full
    assert "AERODIAGNOSIS_VECTOR_BACKEND: chroma_http" in full
    assert "AERODIAGNOSIS_GRAPH_BACKEND: neo4j" in full


def test_ci_builds_and_smoke_tests_v3_container() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "container:" in workflow
    assert "docker/build-push-action" in workflow
    assert "docker compose config --quiet" in workflow
    assert "http://127.0.0.1:18080/api/health" in workflow
    assert "compose.full.yaml" in workflow
    assert 'status["vector"]["backend"] == "chroma_http_cdc"' in workflow
    assert "ghcr.io/inner-vic/aerodiagnosis-next" in workflow
