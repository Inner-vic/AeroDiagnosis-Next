"""Runtime configuration with safe, local-first defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


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
    embedding_backend: str = Field(
        default="hashing", pattern=r"^(hashing|openai_compatible|chroma_default|local_e5)$"
    )
    embedding_input_policy: str = Field(default="source", pattern=r"^(source|chinese_display)$")
    embedding_model_path: Path | None = None
    manual_reranker: str = Field(default="lexical", pattern=r"^(identity|lexical|local_e5)$")
    embedding_dimensions: int = Field(default=256, ge=32, le=8192)
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_api_key: SecretStr | None = None
    external_store_mode: str = Field(
        default="local-first", pattern=r"^(local-first|external-primary)$"
    )
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
    rate_limit_enabled: bool = True
    rate_limit_diagnosis_per_minute: int = Field(default=30, ge=1, le=600)
    default_llm_base_url: str | None = None
    default_llm_model: str | None = None
    default_llm_api_key: SecretStr | None = None
    seed_demo_content: bool = False
    knowledge_parser_backend: str = Field(default="native", pattern=r"^(native|docling)$")
    knowledge_extractor_backend: str = Field(default="lexical", pattern=r"^(lexical|langextract)$")
    knowledge_graph_chinese_display: bool = False
    knowledge_tokenizer_path: Path | None = None
    knowledge_budget_runtime: Path = Path(".runtime/thesis-research-provider-v1")
    knowledge_api_cap_cny: float = Field(default=5.0, gt=0, le=100, allow_inf_nan=False)
    knowledge_extraction_stage: str = Field(
        default="corpus_graph", pattern=r"^(pilot|corpus_graph)$"
    )
    knowledge_profile: str = Field(default="research", pattern=r"^(research|production)$")
    knowledge_chunking: str = Field(
        default="hierarchical", pattern=r"^(fixed|structure|hierarchical)$"
    )
    knowledge_parent_budget: int = Field(default=3200, ge=100, le=32000)
    knowledge_child_budget: int = Field(default=600, ge=100, le=8000)
    manual_retrieval_strategy: str = Field(
        default="hybrid_rrf", pattern=r"^(vector_only|sparse_only|hybrid_rrf)$"
    )
    manual_candidate_k: int = Field(default=50, ge=1, le=500)
    manual_rrf_k: int = Field(default=60, ge=1, le=1000)
    manual_vector_weight: float = Field(default=1.0, ge=0, le=5, allow_inf_nan=False)
    manual_sparse_weight: float = Field(default=1.0, ge=0, le=5, allow_inf_nan=False)
    manual_graph_weight: float = Field(default=0.0, ge=0, le=5, allow_inf_nan=False)
    manual_graph_mode: str = Field(default="entity", pattern=r"^(entity|path)$")
    manual_max_per_document: int = Field(default=3, ge=0, le=50)
    manual_single_source_unlimited: bool = True
    manual_expand_context: bool = True
    text_hybrid_only: bool = False

    @field_validator("api_host")
    @classmethod
    def require_loopback_by_default(cls, value: str) -> str:
        if value not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("non-loopback API binding requires a future hardened deployment mode")
        return value

    @model_validator(mode="after")
    def validate_embedding_configuration(self) -> Self:
        if self.embedding_input_policy == "chinese_display" and (
            self.vector_backend != "sqlite" or self.embedding_backend != "local_e5"
        ):
            raise ValueError("chinese_display requires SQLite and the pinned local_e5 backend")
        if self.manual_vector_weight + self.manual_sparse_weight <= 0:
            raise ValueError("At least one text retrieval weight must be positive")
        if self.manual_graph_weight > 0 and self.manual_retrieval_strategy != "hybrid_rrf":
            raise ValueError("Graph candidates require hybrid_rrf")
        if self.knowledge_parser_backend == "docling" and self.knowledge_tokenizer_path is None:
            raise ValueError("Docling requires a local knowledge tokenizer path")
        if (
            self.knowledge_extractor_backend == "langextract"
            or self.knowledge_graph_chinese_display
        ) and not (
            self.default_llm_api_key
            and (self.default_llm_base_url or "").rstrip("/")
            in {"https://api.deepseek.com", "https://api.deepseek.com/v1"}
            and self.default_llm_model in {"deepseek-flash", "deepseek-v4-pro"}
        ):
            raise ValueError("Budgeted LangExtract requires a configured supported DeepSeek model")
        if self.knowledge_child_budget > self.knowledge_parent_budget:
            raise ValueError("knowledge child budget must not exceed parent budget")
        if self.knowledge_profile == "production":
            if self.seed_demo_content:
                raise ValueError("production knowledge profile forbids demo seeding")
            if self.manual_retrieval_strategy != "sparse_only" and self.embedding_backend not in {
                "openai_compatible",
                "local_e5",
            }:
                raise ValueError(
                    "production vector retrieval requires an explicit trained embedding service"
                )
        if self.embedding_backend == "local_e5" and (
            self.embedding_model_path is None
            or self.embedding_dimensions != 384
            or self.vector_backend != "sqlite"
        ):
            raise ValueError(
                "local_e5 requires local model path, 384 dimensions and SQLite vectors"
            )
        if self.manual_reranker == "local_e5" and self.embedding_backend != "local_e5":
            raise ValueError("local_e5 reranking requires the local_e5 embedding backend")
        if self.embedding_backend == "openai_compatible":
            missing = [
                name
                for name, value in (
                    ("base_url", self.embedding_base_url),
                    ("model", self.embedding_model),
                    ("api_key", self.embedding_api_key),
                )
                if value is None
            ]
            if missing:
                raise ValueError("openai_compatible embeddings require base_url, model and api_key")
        if self.embedding_backend == "chroma_default":
            if self.vector_backend != "chroma_http":
                raise ValueError("chroma_default embeddings require vector_backend=chroma_http")
            if self.external_store_mode != "external-primary":
                raise ValueError("chroma_default embeddings require external-primary store mode")
        return self

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> Self:
        root = (project_root or Path.cwd()).resolve()
        local = _local_environment(root)

        def value(name: str, default: str = "") -> str:
            return os.environ.get(name, local.get(name, default))

        alias = value("AERODIAGNOSIS_MANUAL_RETRIEVAL_STRATEGY")
        canonical = value("AERODIAGNOSIS_MANUAL_RETRIEVAL")
        if alias and canonical and alias != canonical:
            raise ValueError("Conflicting MANUAL_RETRIEVAL and MANUAL_RETRIEVAL_STRATEGY")
        retrieval_strategy = canonical or alias or "hybrid_rrf"

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
            key_text = key_path.read_text(encoding="utf-8-sig").strip()
            if "=" in key_text or "\n" in key_text:
                import re

                matches = set(re.findall(r"sk-[A-Za-z0-9_-]{20,}", key_text))
                if len(matches) != 1:
                    raise ValueError("Credential file must identify exactly one provider key")
                llm_key = matches.pop()
            else:
                llm_key = key_text
        return cls(
            runtime_dir=runtime_dir,
            database_path=database_path,
            vector_backend=value("AERODIAGNOSIS_VECTOR_BACKEND", "sqlite"),
            graph_backend=value("AERODIAGNOSIS_GRAPH_BACKEND", "sqlite"),
            embedding_model_path=(
                Path(value("AERODIAGNOSIS_EMBEDDING_MODEL_PATH"))
                if value("AERODIAGNOSIS_EMBEDDING_MODEL_PATH")
                else None
            ),
            manual_reranker=value("AERODIAGNOSIS_MANUAL_RERANKER", "lexical"),
            embedding_backend=value("AERODIAGNOSIS_EMBEDDING_BACKEND", "hashing"),
            embedding_dimensions=int(value("AERODIAGNOSIS_EMBEDDING_DIMENSIONS", "256")),
            embedding_base_url=value("AERODIAGNOSIS_EMBEDDING_BASE_URL") or None,
            embedding_model=value("AERODIAGNOSIS_EMBEDDING_MODEL") or None,
            embedding_api_key=(
                SecretStr(value("AERODIAGNOSIS_EMBEDDING_API_KEY"))
                if value("AERODIAGNOSIS_EMBEDDING_API_KEY")
                else None
            ),
            external_store_mode=value("AERODIAGNOSIS_EXTERNAL_STORE_MODE", "local-first"),
            chroma_host=value("AERODIAGNOSIS_CHROMA_HOST", "127.0.0.1"),
            chroma_port=int(value("AERODIAGNOSIS_CHROMA_PORT", "8000")),
            chroma_ssl=value("AERODIAGNOSIS_CHROMA_SSL", "false").lower()
            in {"1", "true", "yes", "on"},
            chroma_collection=value("AERODIAGNOSIS_CHROMA_COLLECTION", "aerodiagnosis_chunks"),
            neo4j_uri=value("AERODIAGNOSIS_NEO4J_URI", "bolt://127.0.0.1:7687"),
            neo4j_user=value("AERODIAGNOSIS_NEO4J_USER", "neo4j"),
            neo4j_password=SecretStr(neo4j_password) if neo4j_password else None,
            neo4j_database=value("AERODIAGNOSIS_NEO4J_DATABASE", "neo4j"),
            sync_batch_size=int(value("AERODIAGNOSIS_SYNC_BATCH_SIZE", "50")),
            sync_max_attempts=int(value("AERODIAGNOSIS_SYNC_MAX_ATTEMPTS", "12")),
            api_host=value("AERODIAGNOSIS_API_HOST", "127.0.0.1"),
            api_port=int(value("AERODIAGNOSIS_API_PORT", "8080")),
            operator_token=token,
            rate_limit_enabled=value("AERODIAGNOSIS_RATE_LIMIT_ENABLED", "true").lower()
            in {"1", "true", "yes", "on"},
            rate_limit_diagnosis_per_minute=int(
                value("AERODIAGNOSIS_RATE_LIMIT_DIAGNOSIS_PER_MINUTE", "30")
            ),
            default_llm_base_url=value("AERODIAGNOSIS_LLM_BASE_URL") or None,
            default_llm_model=value("AERODIAGNOSIS_LLM_MODEL") or None,
            default_llm_api_key=SecretStr(llm_key) if llm_key else None,
            knowledge_parser_backend=value("AERODIAGNOSIS_KNOWLEDGE_PARSER", "native"),
            knowledge_extractor_backend=value("AERODIAGNOSIS_KNOWLEDGE_EXTRACTOR", "lexical"),
            knowledge_graph_chinese_display=value(
                "AERODIAGNOSIS_KNOWLEDGE_GRAPH_CHINESE_DISPLAY", "false"
            ).lower()
            in {"1", "true", "yes", "on"},
            knowledge_tokenizer_path=Path(value("AERODIAGNOSIS_KNOWLEDGE_TOKENIZER_PATH"))
            if value("AERODIAGNOSIS_KNOWLEDGE_TOKENIZER_PATH")
            else None,
            knowledge_budget_runtime=Path(
                value(
                    "AERODIAGNOSIS_KNOWLEDGE_BUDGET_RUNTIME",
                    str(root / ".runtime" / "thesis-research-provider-v1"),
                )
            ),
            knowledge_api_cap_cny=float(value("AERODIAGNOSIS_KNOWLEDGE_API_CAP_CNY", "5")),
            knowledge_extraction_stage=value(
                "AERODIAGNOSIS_KNOWLEDGE_EXTRACTION_STAGE", "corpus_graph"
            ),
            knowledge_profile=value("AERODIAGNOSIS_KNOWLEDGE_PROFILE", "research"),
            knowledge_chunking=value("AERODIAGNOSIS_KNOWLEDGE_CHUNKING", "hierarchical"),
            knowledge_parent_budget=int(value("AERODIAGNOSIS_KNOWLEDGE_PARENT_BUDGET", "3200")),
            knowledge_child_budget=int(value("AERODIAGNOSIS_KNOWLEDGE_CHILD_BUDGET", "600")),
            manual_retrieval_strategy=retrieval_strategy,
            text_hybrid_only=value("AERODIAGNOSIS_TEXT_HYBRID_ONLY", "false").lower()
            in {"1", "true", "yes", "on"},
            manual_candidate_k=int(value("AERODIAGNOSIS_MANUAL_CANDIDATE_K", "50")),
            manual_rrf_k=int(value("AERODIAGNOSIS_MANUAL_RRF_K", "60")),
            manual_vector_weight=float(value("AERODIAGNOSIS_MANUAL_VECTOR_WEIGHT", "1")),
            manual_sparse_weight=float(value("AERODIAGNOSIS_MANUAL_SPARSE_WEIGHT", "1")),
            manual_graph_weight=float(value("AERODIAGNOSIS_MANUAL_GRAPH_WEIGHT", "0")),
            manual_graph_mode=value("AERODIAGNOSIS_MANUAL_GRAPH_MODE", "entity"),
            embedding_input_policy=value("AERODIAGNOSIS_EMBEDDING_INPUT_POLICY", "source"),
            manual_max_per_document=int(value("AERODIAGNOSIS_MANUAL_MAX_PER_DOCUMENT", "3")),
            manual_single_source_unlimited=value(
                "AERODIAGNOSIS_MANUAL_SINGLE_SOURCE_UNLIMITED", "true"
            ).lower()
            in {"1", "true", "yes", "on"},
            manual_expand_context=value("AERODIAGNOSIS_MANUAL_EXPAND_CONTEXT", "true").lower()
            in {"1", "true", "yes", "on"},
            seed_demo_content=value("AERODIAGNOSIS_SEED_DEMO_CONTENT", "false").lower()
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
