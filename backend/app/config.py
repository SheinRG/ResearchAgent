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
    # Fast model for structured/auxiliary tasks (planning, reflection).
    groq_model: str = "llama-3.1-8b-instant"
    # Stronger model for the final synthesized answer. Set equal to groq_model
    # to trade answer quality for lower cost/latency.
    groq_synth_model: str = "llama-3.3-70b-versatile"
    groq_timeout: int = 60  # seconds
    # Headroom for a thorough, well-developed answer (multi-paragraph + a table
    # or list) without inviting rambling. The synthesizer scales depth to the
    # question and can use fewer.
    synth_max_tokens: int = 2000  # max answer length for the synthesizer
    # Transient-failure retries (429/5xx/timeouts) before giving up on a call.
    groq_max_retries: int = 2
    groq_retry_base_delay: float = 0.8  # seconds, doubled each retry

    # --- Serper (Search API) ---
    serper_api_key: str = ""

    # --- Tavily (search + read API) ---
    # When use_tavily_search is true AND a key is set, the researcher uses
    # Tavily's single search+content call instead of Serper search + Trafilatura
    # scraping. This is faster (one network round-trip per sub-query instead of
    # search-then-scrape) and far more reliable on JS-heavy sites the scraper
    # returns empty for. Images still come from Serper. Leave the key blank (or
    # set use_tavily_search=false) to fall back to the Serper+scrape path.
    tavily_api_key: str = ""
    use_tavily_search: bool = True
    tavily_search_depth: str = "basic"   # "basic" (1 credit, fast) | "advanced" (2 credits, deeper)
    tavily_timeout: int = 20             # seconds
    # When False, use Tavily's per-result relevance excerpt (~1k chars, returned
    # immediately) instead of fetching+extracting full pages. This is much faster
    # (~1-2s vs ~7s cold) and is plenty of grounding for concise cited answers.
    # Set True to pull full page text for deeper, long-form synthesis.
    tavily_include_raw_content: bool = False

    # --- Redis ---
    redis_url: str = "redis://redis:6379/0"
    cache_ttl: int = 3600  # 1 hour in seconds

    # --- PostgreSQL ---
    database_url: str = "postgresql+asyncpg://agent:agent@postgres:5432/research_agent"

    # --- Auth ---
    auth_secret: str = "change-me-in-production-use-a-random-string"
    google_client_id: str = ""
    google_client_secret: str = ""
    auth_token_expiry_hours: int = 1   # short-lived; refresh tokens handle silent re-issue
    refresh_token_expiry_days: int = 30

    # --- Agent Settings ---
    max_iterations: int = 1
    max_sub_queries: int = 4
    search_results_per_query: int = 5
    # Scrape fewer pages per sub-query: the reranker already trims to top_k, so
    # reading 2 (not 3) pages per sub-query cuts the slowest pipeline step with
    # negligible answer-quality impact.
    scrape_top_n: int = 2
    chunk_size: int = 500
    chunk_overlap: int = 50
    rerank_top_k: int = 12
    # Max distinct sources surfaced to the model + UI for citation.
    max_cited_sources: int = 8

    # --- Re-ranker ---
    # TinyBERT-L-2 is the speed-first model: a 2-layer cross-encoder that reranks
    # several times faster on CPU than MiniLM-L-12. The synthesizer still sees the
    # top-k chunks, so the small ranking-quality trade is worth the latency cut.
    # Use "ms-marco-MiniLM-L-12-v2" if you want max ranking quality over speed.
    reranker_model: str = "ms-marco-TinyBERT-L-2-v2"

    # --- Scraper ---
    # Cap per-page fetch at 8s so a single slow/hanging page can't dominate the
    # batch and stall the whole research run (was 15s).
    scrape_timeout: int = 8
    scrape_max_concurrent: int = 8

    # --- Rate Limiting ---
    rate_limit_per_hour: int = 30  # research queries per user per hour

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # --- Logging ---
    log_level: str = "INFO"

    # --- Observability (optional) ---
    # Set sentry_dsn to enable error tracking; blank = disabled (no-op).
    sentry_dsn: str = ""
    environment: str = "development"        # tags Sentry events, e.g. "production"
    sentry_traces_sample_rate: float = 0.0  # 0 = capture errors only, no perf tracing

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
