"""
SQLAlchemy async models for PostgreSQL persistence.
Stores users, research sessions, and individual queries with their results.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import Column, String, Text, Float, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class User(Base):
    """A registered user of the platform."""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, default="")
    password_hash = Column(String, nullable=True)  # Null for Google-only users
    google_id = Column(String, nullable=True, unique=True, index=True)
    picture = Column(String, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    sessions = relationship("ResearchSession", back_populates="user", cascade="all, delete-orphan")


class ResearchSession(Base):
    """A research session that can contain multiple queries."""
    __tablename__ = "research_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    user = relationship("User", back_populates="sessions")
    queries = relationship("ResearchQuery", back_populates="session", cascade="all, delete-orphan")


class ResearchQuery(Base):
    """An individual research query within a session."""
    __tablename__ = "research_queries"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("research_sessions.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    query = Column(Text, nullable=False)
    sub_queries = Column(JSON, default=list)
    answer = Column(Text, default="")
    sources = Column(JSON, default=list)
    citations = Column(JSON, default=list)
    confidence = Column(Float, default=0.0)
    iterations = Column(Integer, default=1)
    follow_up_suggestions = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    session = relationship("ResearchSession", back_populates="queries")


# --- Database Engine & Session Factory ---

_engine = None
_session_factory = None


def _normalize_db_url(url: str) -> str:
    """
    Ensure the URL uses the async ``asyncpg`` driver.

    Managed hosts (Render, Railway, Neon, Heroku-style) hand out
    ``postgres://`` or ``postgresql://`` URLs, but SQLAlchemy's async engine
    needs the driver named explicitly as ``postgresql+asyncpg://``. Without
    this, the app crashes on boot with "dialect requires async driver".
    """
    if url.startswith("postgresql+"):
        return url  # already has an explicit driver (e.g. +asyncpg)
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://"):]
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://"):]
    return url


def get_engine():
    """Get or create the async database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            _normalize_db_url(settings.database_url),
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory():
    """Get or create the async session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields an async database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all database tables."""
    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)
        raise


async def close_db() -> None:
    """Close the database engine."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database engine closed")
