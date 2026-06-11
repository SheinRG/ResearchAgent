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

    # --- Groq (Cloud LLM) ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_timeout: int = 60  # seconds

    # --- Serper (Search API) ---
    serper_api_key: str = ""

    # --- Redis ---
    redis_url: str = "redis://redis:6379/0"
    cache_ttl: int = 3600  # 1 hour in seconds

    # --- PostgreSQL ---
    database_url: str = "postgresql+asyncpg://agent:agent@postgres:5432/research_agent"

    # --- Auth ---
    auth_secret: str = "change-me-in-production-use-a-random-string"
    google_client_id: str = ""
    google_client_secret: str = ""
    auth_token_expiry_hours: int = 72  # 3 days

    # --- Agent Settings ---
    max_iterations: int = 1
    max_sub_queries: int = 4
    search_results_per_query: int = 5
    scrape_top_n: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50
    rerank_top_k: int = 10

    # --- Rate Limiting ---
    rate_limit_per_hour: int = 30  # research queries per user per hour

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
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    logger.info(
        "Settings loaded: groq_model=%s redis=%s",
        settings.groq_model,
        settings.redis_url,
    )
    return settings
