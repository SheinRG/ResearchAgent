"""
Pydantic models for request/response validation and data transfer.
"""

from pydantic import BaseModel, Field
from typing import Optional
import uuid


class ResearchRequest(BaseModel):
    """Incoming research query from the frontend."""
    query: str = Field(..., min_length=1, max_length=2000, description="The research question")
    max_iterations: int = Field(default=2, ge=1, le=5, description="Max reflection loops")
    session_id: Optional[str] = Field(default=None, description="Existing session ID for follow-ups")


class SearchResult(BaseModel):
    """A single search result from DuckDuckGo."""
    url: str
    title: str
    domain: str
    favicon: str
    snippet: str


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
    """Response for session retrieval."""
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
    """Compact session info for list view."""
    id: str
    query: str
    confidence: float = 0.0
    created_at: str
