"""Application settings.

Loads from (in order of priority):
  1. Process environment variables (e.g. on Render)
  2. ``.env`` file in the project root (local development)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- Application ----
    app_name: str = "Docs RAG API"
    app_env: str = "development"
    debug: bool = False

    # ---- Database ----
    database_url: str = "postgresql+psycopg://docs_rag:docs_rag@localhost:5434/docs_rag"

    # ---- JWT ----
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ---- CORS ----
    cors_origins: str = "*"

    # ---- Gemini ----
    # Same key powers both chat completion and embeddings - one rate-limit pool,
    # one place to rotate. Get a free one at https://aistudio.google.com/apikey.
    gemini_api_key: str = "change-me"
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    # ---- RAG ----
    # Chunk sizes are measured in *tokens* (approximated by Gemini's tokenizer).
    # 500 / 80 are sane defaults for portfolio-sized documents; tune later if
    # retrieval quality drops.
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 80
    retrieval_top_k: int = 5

    # ---- Validators ----

    @field_validator("database_url", mode="before")
    @classmethod
    def _coerce_postgres_driver(cls, value: str) -> str:
        """Normalise bare ``postgres://`` or ``postgresql://`` URLs (as Render
        and Heroku hand them out) to the explicit ``postgresql+psycopg://``
        driver string that SQLAlchemy with psycopg 3 requires.
        """
        if not isinstance(value, str):
            return value
        if value.startswith("postgres://"):
            value = "postgresql://" + value[len("postgres://") :]
        if value.startswith("postgresql://"):
            value = "postgresql+psycopg://" + value[len("postgresql://") :]
        return value

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
