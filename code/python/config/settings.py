"""
应用配置 — 通过环境变量或 .env 文件加载
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Persistence adapters. Embedded stores are the local default; Compose
    # overrides these to chroma_http/neo4j.
    vector_store_backend: str = "embedded"  # embedded | chroma_http
    graph_store_backend: str = "sqlite"  # sqlite | neo4j
    chroma_persist_dir: str = "./data/chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    sqlite_graph_path: str = "./data/knowledge_graph.db"

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8080

    # Document Store
    upload_dir: str = "./uploads"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
