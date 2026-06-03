"""
Application configuration using pydantic-settings.
All settings are loaded from environment variables with sensible defaults.
"""

import logging
from pydantic_settings import BaseSettings
from functools import lru_cache

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Ollama (Local LLM) ---
    ollama_host: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout: int = 120  # seconds

    # --- Redis ---
    redis_url: str = "redis://redis:6379/0"
    cache_ttl: int = 3600  # 1 hour

    # --- PostgreSQL ---
    database_url: str = "postgresql+asyncpg://agent:agent@postgres:5432/research_agent"

    # --- Agent Settings ---
    max_iterations: int = 2
    max_sub_queries: int = 4
    search_results_per_query: int = 5
    scrape_top_n: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50
    rerank_top_k: int = 10

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # --- Logging ---
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    logger.info(
        "Settings loaded: ollama=%s model=%s redis=%s",
        settings.ollama_host,
        settings.ollama_model,
        settings.redis_url,
    )
    return settings
