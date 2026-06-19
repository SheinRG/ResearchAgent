"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models.database import init_db, close_db
from app.services.llm import get_llm_client
from app.services.cache import close_redis
from app.services.scraper import close_scraper
from app.routers import auth, research, sessions, upload, notes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


@app.get("/api/health")
async def health_check():
    llm = get_llm_client()
    llm_ok = await llm.health_check()
    return {
        "status": "healthy",
        "llm": "connected" if llm_ok else "disconnected",
        "model": llm.model,
    }
