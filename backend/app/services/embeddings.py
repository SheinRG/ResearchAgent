"""
Text embeddings for the semantic answer cache.

Groq serves no embedding models, so this is the one place the app talks to a
second inference provider. Two are supported out of the box — Jina and OpenAI,
both plain HTTP — and adding another is a single function.

Design rules, all following from the fact that this sits on the request path:

- **Never raises.** Every failure returns None, and None means "no semantic
  lookup this time". A dead embedding provider must degrade the cache, not the
  product.
- **Never blocks for long.** A short timeout, no retries. The whole point of a
  cache hit is being fast; a slow lookup is worse than a miss, because a miss
  was going to run the pipeline anyway.
- **Vectors come back normalized.** Storing unit vectors means cosine
  similarity is a plain dot product later, which keeps the hot path to one
  multiply-add pass with no square roots.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

_client: Optional[httpx.AsyncClient] = None

# Endpoints per provider. Both speak the same "input -> data[].embedding" shape.
_ENDPOINTS = {
    "jina": "https://api.jina.ai/v1/embeddings",
    "openai": "https://api.openai.com/v1/embeddings",
}


def is_configured() -> bool:
    """Whether embeddings can be produced at all."""
    settings = get_settings()
    return bool(settings.embedding_api_key) and settings.embedding_provider in _ENDPOINTS


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = httpx.AsyncClient(timeout=httpx.Timeout(settings.embedding_timeout))
    return _client


async def close_embeddings() -> None:
    """Close the HTTP client on shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def normalize(vector: list[float]) -> list[float]:
    """
    Scale a vector to unit length so cosine similarity becomes a dot product.

    A zero vector is returned unchanged; :func:`cosine` treats it as similar to
    nothing, which is the correct behaviour for a degenerate embedding.
    """
    magnitude = math.sqrt(sum(v * v for v in vector))
    if magnitude == 0:
        return list(vector)
    return [v / magnitude for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity of two ALREADY-NORMALIZED vectors, as a plain dot product.

    Uses math.sumprod (stdlib, 3.12+), which is C-implemented and fast enough
    that the whole index can be scored in tens of milliseconds without pulling
    numpy into a 512MB container. Mismatched lengths score 0 rather than
    raising — a stored vector from a different embedding model is not similar
    to anything, and must not take the request down.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    try:
        return math.sumprod(a, b)
    except AttributeError:  # pragma: no cover — Python < 3.12
        return sum(x * y for x, y in zip(a, b))


def _build_payload(text: str, settings) -> dict:
    payload: dict = {"model": settings.embedding_model, "input": [text]}
    if settings.embedding_provider == "jina":
        # Jina charges by dimension and supports Matryoshka truncation, so a
        # smaller vector is both cheaper to store and cheaper to send.
        payload["dimensions"] = settings.embedding_dimensions
        payload["task"] = "retrieval.query"
    elif settings.embedding_provider == "openai":
        payload["dimensions"] = settings.embedding_dimensions
    return payload


async def embed(text: str) -> Optional[list[float]]:
    """
    Embed one piece of text, returning a unit vector.

    Returns None when embeddings are unconfigured or anything at all goes
    wrong. Callers treat None as "skip the semantic lookup".
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return None

    settings = get_settings()
    if not is_configured():
        return None

    url = _ENDPOINTS[settings.embedding_provider]
    try:
        response = await _get_client().post(
            url,
            headers={
                "Authorization": f"Bearer {settings.embedding_api_key}",
                "Content-Type": "application/json",
            },
            json=_build_payload(cleaned, settings),
        )
        response.raise_for_status()
        data = response.json()
        vector = data["data"][0]["embedding"]
        if not isinstance(vector, list) or not vector:
            logger.warning("Embedding provider returned an empty vector")
            return None
        return normalize([float(v) for v in vector])
    except Exception as e:
        # Includes timeouts, auth failures, quota exhaustion and shape changes.
        # All of them mean the same thing here: no semantic lookup this time.
        logger.warning("Embedding failed (semantic cache skipped): %s", e)
        return None
