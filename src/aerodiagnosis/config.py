"""Runtime configuration with safe, local-first defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


def _local_environment(root: Path) -> dict[str, str]:
    """Read a git-ignored local profile without executing shell syntax."""

    path = root / ".env.local"
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#") or "=" not in cleaned:
            continue
        key, value = cleaned.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class RuntimeSettings(BaseModel):
    """Configuration shared by inbound and persistence adapters.

    The Windows scripts set ``AERODIAGNOSIS_RUNTIME_DIR`` explicitly. Direct
    Python invocations fall back to a project-local ``.runtime`` directory so
    cloning the repository to D: also keeps data on D:.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_dir: Path
    database_path: Path
    vector_backend: str = Field(default="sqlite", pattern=r"^(sqlite|chroma_http)$")
    graph_backend: str = Field(default="sqlite", pattern=r"^(sqlite|neo4j)$")
    chroma_host: str = "127.0.0.1"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_ssl: bool = False
    chroma_collection: str = "aerodiagnosis_chunks"
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr | None = None
    neo4j_database: str = "neo4j"
    sync_batch_size: int = Field(default=50, ge=1, le=500)
    sync_max_attempts: int = Field(default=12, ge=1, le=100)
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8080, ge=1, le=65535)
    operator_token: str | None = None
    default_llm_base_url: str | None = None
    default_llm_model: str | None = None
    default_llm_api_key: SecretStr | None = None
    seed_demo_content: bool = False

    @field_validator("api_host")
    @classmethod
    def require_loopback_by_default(cls, value: str) -> str:
        if value not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("non-loopback API binding requires a future hardened deployment mode")
        return value

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> Self:
        root = (project_root or Path.cwd()).resolve()
        local = _local_environment(root)

        def value(name: str, default: str = "") -> str:
            return os.environ.get(name, local.get(name, default))

        runtime_dir = Path(value("AERODIAGNOSIS_RUNTIME_DIR", str(root / ".runtime"))).expanduser()
        database_path = Path(
            value(
                "AERODIAGNOSIS_DATABASE_PATH",
                str(runtime_dir / "data" / "aerodiagnosis.db"),
            )
        ).expanduser()
        token = value("AERODIAGNOSIS_OPERATOR_TOKEN") or None
        llm_key = value("AERODIAGNOSIS_LLM_API_KEY") or None
        neo4j_password = value("AERODIAGNOSIS_NEO4J_PASSWORD") or None
        key_file = value("AERODIAGNOSIS_LLM_API_KEY_FILE") or None
        if llm_key is None and key_file is not None:
            key_path = Path(key_file).expanduser().resolve(strict=True)
            if not key_path.is_file() or key_path.stat().st_size > 4096:
                raise ValueError("LLM API key file must be a small regular file")
            llm_key = key_path.read_text(encoding="utf-8-sig").strip()
        return cls(
            runtime_dir=runtime_dir,
            database_path=database_path,
            vector_backend=value("AERODIAGNOSIS_VECTOR_BACKEND", "sqlite"),
            graph_backend=value("AERODIAGNOSIS_GRAPH_BACKEND", "sqlite"),
            chroma_host=value("AERODIAGNOSIS_CHROMA_HOST", "127.0.0.1"),
            chroma_port=int(value("AERODIAGNOSIS_CHROMA_PORT", "8000")),
            chroma_ssl=value("AERODIAGNOSIS_CHROMA_SSL", "false").lower()
            in {"1", "true", "yes", "on"},
            chroma_collection=value(
                "AERODIAGNOSIS_CHROMA_COLLECTION", "aerodiagnosis_chunks"
            ),
            neo4j_uri=value("AERODIAGNOSIS_NEO4J_URI", "bolt://127.0.0.1:7687"),
            neo4j_user=value("AERODIAGNOSIS_NEO4J_USER", "neo4j"),
            neo4j_password=SecretStr(neo4j_password) if neo4j_password else None,
            neo4j_database=value("AERODIAGNOSIS_NEO4J_DATABASE", "neo4j"),
            sync_batch_size=int(value("AERODIAGNOSIS_SYNC_BATCH_SIZE", "50")),
            sync_max_attempts=int(value("AERODIAGNOSIS_SYNC_MAX_ATTEMPTS", "12")),
            api_host=value("AERODIAGNOSIS_API_HOST", "127.0.0.1"),
            api_port=int(value("AERODIAGNOSIS_API_PORT", "8080")),
            operator_token=token,
            default_llm_base_url=value("AERODIAGNOSIS_LLM_BASE_URL") or None,
            default_llm_model=value("AERODIAGNOSIS_LLM_MODEL") or None,
            default_llm_api_key=SecretStr(llm_key) if llm_key else None,
            seed_demo_content=value("AERODIAGNOSIS_SEED_DEMO_CONTENT", "true").lower()
            in {"1", "true", "yes", "on"},
        )

    @property
    def default_provider_available(self) -> bool:
        return all(
            (
                self.default_llm_base_url,
                self.default_llm_model,
                self.default_llm_api_key,
            )
        )

    def prepare(self) -> None:
        """Create only application-owned local directories."""

        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
