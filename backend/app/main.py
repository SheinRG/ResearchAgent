"""FastAPI application factory."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import get_settings
from app.models.database import init_db, close_db, get_engine
from app.services.llm import get_llm_client
from app.services.cache import close_redis, get_redis
from app.services.scraper import close_scraper
from app.services.tavily import close_tavily
from app.routers import auth, research, sessions, upload, notes, files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _init_sentry(settings) -> None:
    """
    Enable Sentry error tracking when SENTRY_DSN is set. No-op otherwise, so a
    deployment without a DSN behaves exactly as before. Sentry auto-instruments
    FastAPI/Starlette, capturing unhandled exceptions across all routes and the
    research agent pipeline.
    """
    if not settings.sentry_dsn:
        return
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.environment,
            traces_sample_rate=settings.sentry_traces_sample_rate,
            send_default_pii=False,
        )
        logger.info("Sentry error tracking enabled (env=%s)", settings.environment)
    except Exception as e:
        logger.warning("Sentry init failed, continuing without it: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Research Agent backend...")

    startup_settings = get_settings()
    if startup_settings.auth_secret.startswith("change-me"):
        logger.warning(
            "=" * 70 + "\n"
            "  SECURITY WARNING: AUTH_SECRET is still the insecure default.\n"
            "  Set a strong random AUTH_SECRET before exposing this to users:\n"
            "    python -c \"import secrets; print(secrets.token_hex(32))\"\n"
            + "=" * 70
        )

    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Database init failed (will retry on first request): %s", e)

    try:
        llm = get_llm_client()
        healthy = await llm.health_check()
        if healthy:
            logger.info("LLM client is healthy (model: %s)", llm.model)
        else:
            logger.warning("LLM health check failed — check API key")
    except Exception as e:
        logger.warning("LLM health check failed: %s", e)

    yield

    await close_scraper()
    await close_tavily()
    await close_redis()
    await close_db()
    logger.info("Backend shutdown complete")


app = FastAPI(
    title="AI Research Agent",
    description="Autonomous research agent with cited answers — powered by Groq + Serper",
    version="2.0.0",
    lifespan=lifespan,
)

settings = get_settings()
_init_sentry(settings)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(research.router)
app.include_router(sessions.router)
app.include_router(upload.router)
app.include_router(notes.router)
app.include_router(files.router)


async def _check_db() -> bool:
    """Run a trivial query to confirm Postgres is reachable."""
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning("Health check: database unreachable: %s", e)
        return False


async def _check_redis() -> bool:
    """Ping Redis. A missing/unavailable Redis is degraded, not fatal."""
    try:
        client = await get_redis()
        if client is None:
            return False
        await client.ping()
        return True
    except Exception as e:
        logger.warning("Health check: redis unreachable: %s", e)
        return False


@app.get("/api/health")
async def health_check():
    """
    Liveness/readiness probe. Checks the LLM, Postgres, and Redis concurrently.

    Postgres is critical (auth + sessions depend on it), so the endpoint returns
    503 when the DB is down — telling the platform's health check the instance is
    not ready. The LLM and Redis are reported but treated as degraded-not-fatal.
    """
    llm = get_llm_client()
    llm_ok, db_ok, redis_ok = await asyncio.gather(
        llm.health_check(),
        _check_db(),
        _check_redis(),
    )

    body = {
        "status": "healthy" if db_ok else "unhealthy",
        "llm": "connected" if llm_ok else "disconnected",
        "model": llm.model,
        "database": "connected" if db_ok else "disconnected",
        "redis": "connected" if redis_ok else "unavailable",
    }
    return JSONResponse(content=body, status_code=200 if db_ok else 503)
