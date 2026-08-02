"""
SQLAlchemy async models for PostgreSQL persistence.
Stores users, research sessions, and individual queries with their results.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import Column, String, Text, Float, Integer, DateTime, ForeignKey, JSON, LargeBinary
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
    preferred_name = Column(String, default="")  # What the agent calls the user
    password_hash = Column(String, nullable=True)  # Null for Google-only users
    google_id = Column(String, nullable=True, unique=True, index=True)
    picture = Column(String, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    sessions = relationship("ResearchSession", back_populates="user", cascade="all, delete-orphan")
    notes = relationship("Note", back_populates="user", cascade="all, delete-orphan")


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
    documents = Column(JSON, default=list)
    # Answer-quality and cost signals, recorded per turn so they can be
    # aggregated later (which questions cost the most, which fabricate).
    invalid_citations = Column(Integer, default=0)   # [n] markers with no such source
    total_tokens = Column(Integer, default=0)        # across every LLM call in the turn
    cost_usd = Column(Float, default=0.0)            # estimated; see services.usage
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    session = relationship("ResearchSession", back_populates="queries")


class Note(Base):
    """A user-created note stored in the database."""
    __tablename__ = "notes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    user = relationship("User", back_populates="notes")


class RefreshToken(Base):
    """Server-side record of a refresh token (stored as a SHA-256 hash, never
    plaintext). DB-backed rather than Redis so sessions survive cache restarts —
    losing these rows silently logs every user out."""
    __tablename__ = "refresh_tokens"

    token_hash = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class UploadedFile(Base):
    """Raw bytes of an uploaded file, kept so the document viewer can re-render
    it on restored sessions (the browser's in-memory File is gone after reload)."""
    __tablename__ = "uploaded_files"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    filename = Column(String, default="")
    mime = Column(String, default="")
    size = Column(Integer, default=0)
    content = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


# --- Database Engine & Session Factory ---

_engine = None
_session_factory = None


# libpq connection parameters that asyncpg does not accept as kwargs. SQLAlchemy
# forwards unknown query-string params straight to the driver, so leaving these
# in the URL crashes on boot with:
#   TypeError: connect() got an unexpected keyword argument 'sslmode'
# Supabase and Neon both include them in the URLs they hand out.
_LIBPQ_ONLY_PARAMS = ("sslmode", "channel_binding", "sslrootcert", "sslcert", "sslkey")


def _normalize_db_url(url: str) -> tuple[str, dict]:
    """
    Adapt a managed-host Postgres URL to SQLAlchemy + asyncpg.

    Returns ``(url, connect_args)``.

    Two fixes are applied:

    1. **Driver.** Managed hosts (Supabase, Neon, Render, Heroku-style) hand out
       ``postgres://`` or ``postgresql://`` URLs, but SQLAlchemy's async engine
       needs the driver named explicitly as ``postgresql+asyncpg://``. Without
       this, the app crashes on boot with "dialect requires async driver".

    2. **SSL.** Those hosts also append libpq-style params like
       ``?sslmode=require``, which asyncpg rejects (see ``_LIBPQ_ONLY_PARAMS``).
       They are stripped from the URL and re-expressed as ``ssl=True`` in
       connect_args, which is asyncpg's equivalent.

    Note for Supabase: use the **session** pooler (port 5432), not the
    transaction pooler (6543). Transaction mode is incompatible with asyncpg's
    prepared-statement cache. The direct ``db.<ref>.supabase.co`` host is
    IPv6-only and unreachable from IPv4-only platforms such as Render.
    """
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://"):]

    if not url.startswith("postgresql+asyncpg://"):
        return url, {}  # e.g. a non-postgres or already-custom driver URL

    parts = urlsplit(url)
    kept, sslmode = [], None
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key == "sslmode":
            sslmode = value.lower()
        elif key not in _LIBPQ_ONLY_PARAMS:
            kept.append((key, value))
        # other libpq-only params (channel_binding, sslcert, ...) are dropped

    # asyncpg accepts the libpq sslmode *strings* directly, so pass the mode
    # through rather than translating it. Do NOT map require -> ssl=True: in
    # libpq "require" means encrypt-without-verifying, but asyncpg's ssl=True
    # means verify-full. Supabase's pooler serves a chain that fails default
    # verification, so ssl=True raises SSLCertVerificationError on connect.
    connect_args: dict = {"ssl": sslmode} if sslmode else {}

    # With no sslmode, asyncpg defaults to "prefer": TLS when the server offers
    # it (verified TLS 1.3 against the Supabase pooler), plaintext against the
    # local docker-compose Postgres, which serves no TLS. Both work untouched.
    return urlunsplit(parts._replace(query=urlencode(kept))), connect_args


def get_engine():
    """Get or create the async database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        url, connect_args = _normalize_db_url(settings.database_url)
        _engine = create_async_engine(
            url,
            echo=False,
            pool_size=5,
            max_overflow=10,
            # Managed Postgres (Supabase pooler, Neon autosuspend) drops idle
            # connections server-side; without this the app serves one 500 per
            # stale connection after a quiet period.
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args=connect_args,
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
    """Create all database tables and apply lightweight column migrations."""
    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # create_all only creates missing tables, not new columns on existing
            # ones. Until Alembic is introduced, add new columns idempotently so
            # already-deployed databases pick them up without a manual migration.
            from sqlalchemy import text
            await conn.execute(
                text("ALTER TABLE users ADD COLUMN IF NOT EXISTS preferred_name VARCHAR DEFAULT ''")
            )
            await conn.execute(
                text("ALTER TABLE research_queries ADD COLUMN IF NOT EXISTS documents JSON")
            )
            await conn.execute(
                text("ALTER TABLE research_queries ADD COLUMN IF NOT EXISTS invalid_citations INTEGER DEFAULT 0")
            )
            await conn.execute(
                text("ALTER TABLE research_queries ADD COLUMN IF NOT EXISTS total_tokens INTEGER DEFAULT 0")
            )
            await conn.execute(
                text("ALTER TABLE research_queries ADD COLUMN IF NOT EXISTS cost_usd DOUBLE PRECISION DEFAULT 0")
            )
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
