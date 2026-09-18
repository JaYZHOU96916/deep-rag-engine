from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables only."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Deep-RAG API"
    environment: Literal["development", "test", "production"] = "development"
    database_url: str = "postgresql+psycopg://deep_rag:deep-rag-local-development-only@postgres:5432/deep_rag"
    redis_url: str = "redis://redis:6379/0"
    uploads_dir: Path = Path("/app/data/uploads")
    max_upload_bytes: int = 100 * 1024 * 1024
    embedding_dimension: int = 384
    embedding_provider: Literal["local_hash", "openai_compatible"] = "local_hash"
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    dense_candidate_limit: int = 24
    sparse_candidate_limit: int = 24
    rrf_k: int = 60
    reranker_provider: Literal["lexical", "bge_http"] = "lexical"
    reranker_endpoint: str | None = None
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_min_score: float = 0.05
    semantic_cache_threshold: float = 0.95
    semantic_cache_ttl_seconds: int = 86_400
    sse_replay_ttl_seconds: int = 3_600
    llm_provider: Literal["local", "openai", "deepseek", "claude"] = "local"
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.1
    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    claude_api_key: SecretStr | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
