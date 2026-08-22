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
    # Migrated off Llama on 2026-08-22: Groq decommissioned both
    # llama-3.1-8b-instant and llama-3.3-70b-versatile, and every call started
    # returning 404 model_not_found. The pipeline degraded exactly as designed
    # -- triage fell back to research, synthesis returned the no-sources
    # message -- so the demo answered nothing while looking like it was working.
    #
    # Fast model for structured/auxiliary tasks (triage, follow-ups). It must
    # hold JSON mode: triage's whole contract is a structured response, and of
    # the models Groq now serves, gpt-oss-20b is the one that passes Groq's JSON
    # validation. gpt-oss-120b and qwen3.6-27b both fail it, and qwen
    # additionally streams <think> reasoning that would land in the answer.
    groq_model: str = "openai/gpt-oss-20b"
    # Stronger model for the final synthesized answer. Set equal to groq_model
    # to trade answer quality for lower cost/latency.
    groq_synth_model: str = "openai/gpt-oss-120b"
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

    # --- Search retries (Serper + Tavily) ---
    # Both providers used to return [] on any failure, which the pipeline cannot
    # tell apart from "no results exist" — a blip produced a confident answer
    # with no sources instead of a visible error.
    #
    # Deliberately shorter-tempered than the Groq retries above. Every attempt
    # that reaches a search provider is billed by it, and the sub-queries run in
    # parallel, so a provider outage multiplies both the credit burn and the
    # latency by the number of sub-queries at once. The deadline is the real
    # guard: at tavily_timeout=20s, three attempts would otherwise be a 60s wall
    # on the slowest step in the product.
    search_max_retries: int = 2
    search_retry_base_delay: float = 0.5   # seconds, doubled each retry
    search_retry_deadline: float = 25.0    # seconds; no further attempt starts after this

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

    # --- Answer cache (exact match) ---
    # Short-circuits an identical repeat question before it reaches the graph,
    # skipping the triage call, search, rerank AND the expensive synthesis call.
    # The existing search/scrape caches only avoid the network hops; both LLM
    # calls still ran on a repeat query before this.
    answer_cache_enabled: bool = True
    # TTL for answers triage flagged as time-sensitive (news, prices, "latest").
    # Short, because these go wrong quickly.
    answer_cache_ttl: int = 900  # 15 minutes
    # TTL for evergreen answers ("what is quantum computing"). Triage now labels
    # each answer as it is produced, so the flag rides along with the cached
    # entry and a definition no longer expires as fast as a stock price.
    answer_cache_evergreen_ttl: int = 21600  # 6 hours

    # --- Semantic answer cache (near-duplicate questions) ---
    # Sits behind the exact-match cache: on an exact miss, embed the question and
    # look for a previously answered one with the same meaning.
    #
    # OFF by default, unlike the other optional integrations. This one can serve
    # a user an answer to a question they did not ask, so it should be switched
    # on deliberately after the threshold has been checked against real pairs
    # (see evals/). It additionally requires an embedding key.
    semantic_cache_enabled: bool = False
    # Cosine similarity required to reuse an answer. 0.92 is a starting point,
    # not a tuned value — verify with the threshold sweep before trusting it.
    semantic_similarity_threshold: float = 0.92
    # Questions must also share this fraction of their meaningful words, as a
    # blunt backstop against two unrelated questions that happen to score well.
    #
    # 0.3 rather than something stricter because short questions have very few
    # content words once stopwords are dropped: "what changed in Python 3.12"
    # and "what is new in Python 3.12" reduce to {changed, python} vs
    # {new, python} — a genuine paraphrase scoring only 0.33. Unrelated
    # questions score ~0.0, so this still separates them decisively. The
    # dangerous near-misses are caught by the number and polarity rails, not
    # by this one.
    semantic_min_token_overlap: float = 0.3
    # Vectors held per bucket. Every lookup scores the whole bucket in-process,
    # so this caps both memory and lookup cost (~40ms at 500).
    semantic_max_index_entries: int = 300
    # A semantic hit on a time-sensitive answer must be much fresher than an
    # exact hit: the question is not even the same one, so staleness compounds.
    semantic_time_sensitive_max_age: int = 120  # 2 minutes

    # --- Embeddings (only used by the semantic cache) ---
    # Groq serves no embedding models, so this is a separate provider. Blank key
    # = semantic cache stays inert regardless of the flag above.
    embedding_provider: str = "jina"      # "jina" | "openai"
    embedding_api_key: str = ""
    embedding_model: str = "jina-embeddings-v3"
    embedding_timeout: int = 10           # seconds; the hot path cannot wait longer
    embedding_dimensions: int = 384       # smaller = less Redis, still plenty here

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

    # New accounts per IP per rolling 24h. 0 disables the limit.
    #
    # Not a budget-bypass guard — the ceilings below are global, so minting
    # accounts cannot get past them by construction. This exists for denial of
    # service: one script registering in a loop burns the shared daily allowance
    # and every legitimate visitor lands in degraded mode.
    #
    # Rolling day rather than the hourly window used for queries, because signup
    # abuse is a script left running rather than a burst — an hourly cap of the
    # same size would hand out 24x the accounts. Generous enough for an office,
    # campus or carrier NAT where several real people share one address, and
    # like every per-IP counter here it is best-effort: X-Forwarded-For is
    # client-supplied. It raises the cost of the attack; it does not close it.
    registrations_per_ip_per_day: int = 5

    # --- Anonymous demo mode ---
    # Logged-out visitors can run real research without registering, because a
    # product pitched on "watch it show its work" should not make you sign up
    # before you can watch it work.
    #
    # The per-visitor number is a *conversion* limit, not a cost limit — the
    # ceilings below are what protect the budget. It is small on purpose: nobody
    # runs 30 research queries on a stranger's demo, so a 30-query wall is one
    # nobody ever reaches, and a signup gate nobody reaches converts nobody.
    # Three is roughly where a visitor is still interested when they hit it.
    anonymous_demo_enabled: bool = True
    anon_free_queries: int = 3          # per visitor (cookie), rolling 24h
    # Higher, because this is keyed by IP and offices, campuses and mobile
    # carriers share one. It exists to blunt cookie-clearing, not to be precise:
    # X-Forwarded-For is spoofable, so the global ceilings below — which are not
    # bypassable by construction — remain the real guard.
    anon_free_queries_per_ip: int = 10  # per IP, rolling 24h

    # --- Spend ceiling (see services/budget.py) ---
    # Sized against Tavily's 1,000 credits/month free tier, which is the binding
    # constraint: Groq's free tier throttles and recovers within the hour, but a
    # spent search credit is gone until the month rolls over.
    #
    #   1,000/month  -  ~240 for the weekly eval run  -  ~160 headroom
    #   = ~600/month of serving budget.
    #
    # One research turn bills up to max_sub_queries × depth = 4 credits, so 600
    # is roughly 150 live runs a month. That sounds small, and it is — the
    # answer cache is what makes it work. Logged-out traffic is highly
    # correlated (everyone tries the same few questions), and a cache hit costs
    # zero credits and zero tokens. The daily allowance is really a
    # cache-warming budget; the cache does the serving.
    budget_guard_enabled: bool = True
    monthly_search_credits: int = 600
    # 6× the 20/day monthly average, on purpose: a launch day should be allowed
    # to burst. The monthly ceiling means you cannot burst every day — about
    # five heavy days and it stops, which is the intended trade.
    daily_search_credits: int = 120
    # The demo's sub-pool of the daily allowance (~10 live runs), so a spike of
    # logged-out visitors throttles itself before it can starve signed-in users.
    # Raise this for a launch day; the monthly ceiling still backstops it.
    anon_daily_search_credits: int = 40
    # Backstop against a model swap making tokens expensive. Not the binding
    # constraint at current prices — ~$0.003/run means the credit ceilings bite
    # first — but it is what catches groq_synth_model pointing somewhere pricey.
    daily_cost_budget_usd: float = 0.25

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

    # --- Langfuse tracing (optional) ---
    # Per-stage latency and token cost for the research pipeline. Enabled only
    # when BOTH keys are set; blank = disabled (no-op), same as Sentry above.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"  # self-hosted: your own URL
    # Fraction of traces recorded. 1.0 is fine at this traffic level; lower it
    # if the free tier's event quota becomes the constraint.
    langfuse_sample_rate: float = 1.0
    # False sends only timings, token counts and counts-of-things — no prompts,
    # no answers, no source text. Traces stay useful for latency and cost work
    # while carrying none of the user's content.
    langfuse_capture_content: bool = True
    # Prompts carry the full text of every scraped source. Truncate before
    # shipping: nobody reads 40KB in a trace viewer, and it is billed volume.
    langfuse_max_content_chars: int = 2000

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
