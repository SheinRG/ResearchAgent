"""
Pydantic models for request/response validation and data transfer.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import uuid
import re


# --- Auth Schemas ---

class RegisterRequest(BaseModel):
    """User registration request."""
    email: str = Field(..., description="User email")
    password: str = Field(..., min_length=8, description="User password (min 8 chars, 1 uppercase, 1 number)")
    name: str = Field(default="", description="Display name")

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Enforce password policy: min 8 chars, at least 1 uppercase, 1 number."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one number")
        return v


class LoginRequest(BaseModel):
    """User login request."""
    email: str = Field(..., description="User email")
    password: str = Field(..., description="User password")


class GoogleAuthRequest(BaseModel):
    """Google OAuth login request."""
    credential: str = Field(..., description="Google ID token")


class AuthResponse(BaseModel):
    """Authentication response with JWT token."""
    token: str
    user: dict


class ProfileUpdateRequest(BaseModel):
    """Update a user's personalization settings."""
    preferred_name: str = Field(
        default="",
        max_length=50,
        description="What the agent should call the user (blank to clear)",
    )

    @field_validator("preferred_name")
    @classmethod
    def clean_preferred_name(cls, v: str) -> str:
        """Trim and strip control characters so it's safe to drop into prompts."""
        v = (v or "").strip()
        return re.sub(r"[\x00-\x1f\x7f]", "", v)[:50]


# --- Research Schemas ---

class DocumentInput(BaseModel):
    """A single uploaded document supplied alongside a research query."""
    name: str = Field(default="", max_length=255, description="File name (used as source title)")
    text: str = Field(default="", max_length=16000, description="Extracted plain text of the document")

    @field_validator("name", "text")
    @classmethod
    def clean_control_chars(cls, v: str) -> str:
        """Strip control characters (except newline/tab) so content is safe to drop into prompts."""
        v = (v or "").strip()
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)


class HistoryTurn(BaseModel):
    """A prior question/answer exchange, supplied for follow-up context."""
    query: str = Field(..., max_length=2000, description="The question asked in that turn")
    answer: str = Field(default="", description="The answer given in that turn")


class ResearchRequest(BaseModel):
    """Incoming research query from the frontend."""
    query: str = Field(..., min_length=1, max_length=2000, description="The research question")
    max_iterations: int = Field(default=1, ge=1, le=5, description="Max reflection loops")
    session_id: Optional[str] = Field(default=None, description="Existing session ID for follow-ups")
    history: list[HistoryTurn] = Field(
        default_factory=list,
        max_length=20,
        description="Prior conversation turns for follow-up context",
    )
    documents: list[DocumentInput] = Field(
        default_factory=list,
        max_length=5,
        description="Uploaded document texts to answer from",
    )


class SearchResult(BaseModel):
    """A single search result from web search."""
    url: str
    title: str
    domain: str
    favicon: str
    snippet: str


class ImageResult(BaseModel):
    """A single image result from image search (powers the Images tab)."""
    url: str = ""          # Full-resolution image URL
    thumbnail: str = ""    # Thumbnail URL (falls back to the full image)
    title: str = ""        # Image/page title
    source: str = ""       # URL of the page the image was found on
    domain: str = ""       # Domain of the source page


class RankedChunk(BaseModel):
    """A text chunk with its relevance score after re-ranking."""
    text: str
    score: float
    source_url: str
    source_title: str
    source_domain: str


class Citation(BaseModel):
    """A citation linking a claim to a source."""
    index: int
    source_url: str
    source_title: str
    source_domain: str
    claim: str = ""


class ReflectionResult(BaseModel):
    """Output of the reflector node."""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    gaps: list[str] = Field(default_factory=list)
    should_continue: bool = False
    refined_queries: list[str] = Field(default_factory=list)
    reasoning: str = ""


class PhaseEvent(BaseModel):
    """SSE phase update event."""
    phase: str
    message: str


class SubQueriesEvent(BaseModel):
    """SSE sub-queries event."""
    queries: list[str]


class SourcesEvent(BaseModel):
    """SSE sources event."""
    sources: list[SearchResult]


class ImagesEvent(BaseModel):
    """SSE images event powering the frontend Images tab."""
    images: list[ImageResult]


class TokenEvent(BaseModel):
    """SSE token event for streaming answer."""
    token: str


class FollowUpEvent(BaseModel):
    """SSE follow-up suggestions event."""
    suggestions: list[str]


class DoneEvent(BaseModel):
    """SSE done event with session metadata."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    total_sources: int = 0
    iterations: int = 1
    confidence: float = 0.0


class SessionResponse(BaseModel):
    """Response for session retrieval (legacy single-turn shape, kept for back-compat)."""
    id: str
    query: str
    answer: Optional[str] = None
    sources: list[SearchResult] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    iterations: int = 1
    follow_up_suggestions: list[str] = Field(default_factory=list)
    created_at: str


class SessionListItem(BaseModel):
    """Compact session info for list view (one entry per session/thread)."""
    id: str
    query: str          # title = first query in the session (kept for back-compat)
    title: str          # same as query; explicit field for the new contract
    confidence: float = 0.0
    turn_count: int = 1
    created_at: str     # ISO timestamp of the first turn
    updated_at: str     # ISO timestamp of the most recent turn


# ---------------------------------------------------------------------------
# Multi-turn thread response shapes
# ---------------------------------------------------------------------------

class SessionTurn(BaseModel):
    """A single turn (question + answer) within a stored research thread."""
    query: str
    answer: str = ""
    sources: list[SearchResult] = Field(default_factory=list)
    citations: list = Field(default_factory=list)   # raw; may be dict or Citation
    confidence: float = 0.0
    iterations: int = 1
    follow_up_suggestions: list[str] = Field(default_factory=list)
    created_at: str


class SessionThreadResponse(BaseModel):
    """Full thread: session metadata plus all ordered turns."""
    id: str
    title: str          # first query in the session
    created_at: str     # ISO timestamp of the first turn
    turns: list[SessionTurn] = Field(default_factory=list)


# --- Notes Schemas ---

class NoteCreate(BaseModel):
    """Create a new note."""
    text: str = Field(..., min_length=1, max_length=10000)


class NoteUpdate(BaseModel):
    """Update an existing note."""
    text: str = Field(..., min_length=1, max_length=10000)


class NoteResponse(BaseModel):
    """A note returned from the API."""
    id: str
    text: str
    created_at: str
    updated_at: str
