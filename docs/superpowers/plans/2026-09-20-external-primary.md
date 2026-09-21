# External-Primary Production Store Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `external-primary` store mode where ChromaDB and Neo4j are direct primary stores for vector and graph data, while SQLite remains only the metadata, session, case, and checkpoint store.

**Architecture:** Introduce `AERODIAGNOSIS_EXTERNAL_STORE_MODE`. When it is `external-primary`, persistence factories return `ChromaHttpVectorStore` and `Neo4jGraphStore` directly instead of `ReplicatedVectorStore` and `ReplicatedGraphStore`. A production Compose overlay starts app, ChromaDB, and Neo4j without the CDC sync worker.

**Tech Stack:** Python 3.13, Pydantic, FastAPI, Docker Compose, pytest.

## Global Constraints

- Keep `local-first` as the default mode so embedded SQLite and the existing CDC full stack remain available.
- Do not remove the outbox/replication code or its tests.
- Keep API secrets out of responses and tests.
- Keep the application image non-root and read-only.
- Preserve all current endpoints and database schema version 7.

---

## Task 1: Add External Store Mode Configuration

**Files:**
- Modify: `src/aerodiagnosis/config.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `RuntimeSettings.external_store_mode: str`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_config.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_config.py -k external_store -v`
Expected: FAIL because `external_store_mode` does not exist.

- [ ] **Step 3: Add the field and environment parsing**

In `src/aerodiagnosis/config.py`, add after `graph_backend`:

```python
external_store_mode: str = Field(
    default="local-first", pattern=r"^(local-first|external-primary)$"
)
```

In `RuntimeSettings.from_env`, add before `operator_token`:

```python
external_store_mode=value("AERODIAGNOSIS_EXTERNAL_STORE_MODE", "local-first"),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_config.py -k external_store -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aerodiagnosis/config.py tests/unit/test_config.py
git commit -m "feat: add external store mode setting"
```

---

## Task 2: Select Direct Primary Stores

**Files:**
- Modify: `src/aerodiagnosis/adapters/persistence/__init__.py`
- Test: `tests/unit/test_persistence_factory.py`

**Interfaces:**
- Consumes: `RuntimeSettings.external_store_mode`
- Produces: direct `ChromaHttpVectorStore` and `Neo4jGraphStore` when mode is `external-primary`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_persistence_factory.py`:

```python
from aerodiagnosis.adapters.persistence import create_external_sync


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_persistence_factory.py -k external_primary -v`
Expected: FAIL for direct stores and sync behavior.

- [ ] **Step 3: Add helper constructors and mode branches**

Add after the imports in `src/aerodiagnosis/adapters/persistence/__init__.py`:

```python
def _chroma_vector_store(settings: RuntimeSettings) -> ChromaHttpVectorStore:
    return ChromaHttpVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        ssl=settings.chroma_ssl,
        collection_name=settings.chroma_collection,
    )


def _neo4j_graph_store(settings: RuntimeSettings) -> Neo4jGraphStore:
    if settings.neo4j_password is None:
        raise UnsupportedBackendError(
            "AERODIAGNOSIS_NEO4J_PASSWORD is required when graph backend is neo4j"
        )
    return Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )
```

In `create_vector_store`, insert after the `sqlite` branch:

```python
if settings.external_store_mode == "external-primary":
    return _chroma_vector_store(settings)
```

Replace the existing `replica = ChromaHttpVectorStore(...)` with:

```python
replica = _chroma_vector_store(settings)
```

In `create_graph_store`, replace the password check and `replica = Neo4jGraphStore(...)` with:

```python
if settings.external_store_mode == "external-primary":
    return _neo4j_graph_store(settings)
if settings.neo4j_password is None:
    raise UnsupportedBackendError(
        "AERODIAGNOSIS_NEO4J_PASSWORD is required when graph backend is neo4j"
    )
...
replica = _neo4j_graph_store(settings)
```

At the start of `create_external_sync`, add:

```python
if settings.external_store_mode == "external-primary":
    raise UnsupportedBackendError(
        "external sync is disabled in external-primary mode"
    )
```

Use the helper constructors for `create_external_sync` too.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_persistence_factory.py tests/unit/test_external_store_sync.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aerodiagnosis/adapters/persistence/__init__.py tests/unit/test_persistence_factory.py
git commit -m "feat: use direct external stores in external-primary mode"
```

---

## Task 3: Report Direct External Status

**Files:**
- Modify: `src/aerodiagnosis/adapters/api/routers.py`
- Test: `tests/unit/test_api_external_primary_status.py`

**Interfaces:**
- Produces: `_external_sync_status(settings)` returning a status dictionary.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_api_external_primary_status.py`:

```python
from pathlib import Path

from aerodiagnosis.adapters.api.routers import _external_sync_status
from aerodiagnosis.config import RuntimeSettings


def test_external_primary_status_does_not_report_cdc(tmp_path: Path) -> None:
    settings = RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "runtime.db",
        vector_backend="chroma_http",
        graph_backend="neo4j",
        external_store_mode="external-primary",
    )

    assert _external_sync_status(settings) == {
        "mode": "direct_external_primary",
        "events": {"applied": 0, "pending": 0, "dead": 0},
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_api_external_primary_status.py -v`
Expected: FAIL with `_external_sync_status` not defined.

- [ ] **Step 3: Add the helper and use it**

Add near the top of `src/aerodiagnosis/adapters/api/routers.py`, after `ApplicationDependency`:

```python
def _external_sync_status(settings: RuntimeSettings) -> dict[str, Any]:
    if settings.external_store_mode == "external-primary":
        return {
            "mode": "direct_external_primary",
            "events": {"applied": 0, "pending": 0, "dead": 0},
        }
    outbox = ExternalStoreOutbox(settings.database_path).counts()
    return {"mode": "transactional_outbox", "events": outbox}
```

In `system()`, replace the `outbox = ...` line and the `"external_sync"` block with:

```python
external_sync = _external_sync_status(application.settings)
...
"external_sync": external_sync,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_api_external_primary_status.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aerodiagnosis/adapters/api/routers.py tests/unit/test_api_external_primary_status.py
git commit -m "feat: report direct external primary status"
```

---

## Task 4: Add Production Compose Overlay and Deploy

**Files:**
- Create: `compose.production.yaml`
- Modify: `.env`
- Modify: `docs/DEPLOYMENT.md`
- Modify: `docs/architecture/EXTERNAL-STORE-SYNC.md`
- Test: `tests/unit/test_container_assets.py`

**Interfaces:**
- Produces: `docker compose -f compose.yaml -f compose.production.yaml up --build -d`

- [ ] **Step 1: Write the failing container asset test**

Add to `tests/unit/test_container_assets.py`:

```python
def test_production_compose_uses_direct_external_primary() -> None:
    compose = (PROJECT_ROOT / "compose.production.yaml").read_text(encoding="utf-8")

    assert "AERODIAGNOSIS_EXTERNAL_STORE_MODE: external-primary" in compose
    assert "AERODIAGNOSIS_VECTOR_BACKEND: chroma_http" in compose
    assert "AERODIAGNOSIS_GRAPH_BACKEND: neo4j" in compose
    assert "chromadb/chroma:1.5.9" in compose
    assert "neo4j:5.26.30-community" in compose
    assert "aerodiagnosis-sync" not in compose
```

Also change the Dockerfile assertion from:

```python
assert "FROM python:3.13-slim-bookworm" in dockerfile
```

to:

```python
assert "FROM ${PYTHON_BASE_IMAGE}" in dockerfile
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_container_assets.py -v`
Expected: FAIL before the production Compose file exists.

- [ ] **Step 3: Create production Compose overlay**

Create `compose.production.yaml` with app, ChromaDB, and Neo4j services only. The app must use:

```yaml
AERODIAGNOSIS_EXTERNAL_STORE_MODE: external-primary
AERODIAGNOSIS_VECTOR_BACKEND: chroma_http
AERODIAGNOSIS_GRAPH_BACKEND: neo4j
```

No `sync` service and no `aerodiagnosis-sync` command.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_container_assets.py tests/unit/test_persistence_factory.py tests/unit/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Rebuild and switch the running deployment**

Run:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/e/AeroDiagnosis/AeroDiagnosis-Next && docker compose -f compose.yaml -f compose.production.yaml up --build -d"
```

Verify:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/api/system
```

Expected:

```json
{
  "vector": {"backend": "chroma_http_hashing"},
  "graph": {"backend": "neo4j"},
  "external_sync": {"mode": "direct_external_primary"}
}
```

- [ ] **Step 6: Commit**

```bash
git add compose.production.yaml .env docs/DEPLOYMENT.md docs/architecture/EXTERNAL-STORE-SYNC.md tests/unit/test_container_assets.py
git commit -m "feat: add external-primary production deployment"
```

---

## Self-Review

1. **Spec coverage:** P0 is covered by configuration, factories, status reporting, and deployment.
2. **Placeholder scan:** No TBD or generic error handling remains.
3. **Type consistency:** `external_store_mode` is used consistently in config, factories, and API status.
