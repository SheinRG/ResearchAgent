"""
Async Groq client for cloud LLM inference.
Supports regular generation, streaming, and structured JSON output,
with exponential-backoff retries on transient failures.

Every call reports its token usage: each method takes a ``stage`` label (which
pipeline step is spending the tokens) and an optional ``on_usage`` callback.
Usage is always logged, and is always handed to ``services.usage``, which sums
it for the request in flight. The per-call ``on_usage`` callback is the extra
seam a tracing backend hooks into without this module having to know about it.
"""

import json
import time
import asyncio
import logging
from typing import AsyncGenerator, Optional

import httpx
from groq import AsyncGroq

from app.config import get_settings

# TokenUsage/UsageCallback live in services.usage — the accounting module owns
# the accounting type. Re-exported here so callers can keep importing them from
# the client that produces them.
from app.services.usage import TokenUsage, UsageCallback, record_usage

__all__ = ["GroqClient", "TokenUsage", "UsageCallback", "get_llm_client"]

logger = logging.getLogger(__name__)

# HTTP statuses worth retrying — transient server/throttling errors only.
# 4xx like 400/401/404 are permanent and must NOT be retried.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    """Whether a Groq/httpx exception represents a transient, retryable failure."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)):
        return True
    # Groq SDK errors expose .status_code; some wrap an httpx response instead.
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    return status in _RETRYABLE_STATUS


class GroqClient:
    """Async wrapper around the Groq Python SDK."""

    def __init__(self):
        settings = get_settings()
        self.model = settings.groq_model
        self.timeout = settings.groq_timeout
        self.max_retries = settings.groq_max_retries
        self.retry_base_delay = settings.groq_retry_base_delay
        self.client = AsyncGroq(
            api_key=settings.groq_api_key,
            timeout=httpx.Timeout(self.timeout),
        )
        # Short-lived health-check cache so frequent /api/health polls from a
        # load balancer don't hammer Groq's models endpoint.
        self._health_cache: Optional[bool] = None
        self._health_cache_at: float = 0.0

    def _record_usage(
        self,
        stage: str,
        model: str,
        raw_usage: object,
        on_usage: Optional[UsageCallback],
    ) -> Optional[TokenUsage]:
        """
        Log a call's token usage, add it to the request total, and hand it to
        the per-call observer, if any.

        ``raw_usage`` is the SDK's usage object (or None when the API omitted
        it). Never raises: token accounting must not be able to fail a request.
        """
        if raw_usage is None:
            logger.debug("LLM usage unavailable for stage=%s model=%s", stage, model)
            return None

        usage = TokenUsage(
            stage=stage or "unknown",
            model=model,
            prompt_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(raw_usage, "total_tokens", 0) or 0),
        )
        logger.info(
            "LLM usage: stage=%s model=%s prompt=%d completion=%d total=%d",
            usage.stage, usage.model,
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
        )

        # Ambient accounting first: it is a no-op outside a request scope and
        # cannot raise, so it runs regardless of what the caller passed.
        record_usage(usage)

        if on_usage is not None:
            try:
                on_usage(usage)
            except Exception as e:
                logger.warning("on_usage callback failed (ignored): %s", e)

        return usage

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        format_json: bool = False,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        stage: str = "",
        on_usage: Optional[UsageCallback] = None,
    ) -> str:
        """
        Generate a complete response from Groq.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            temperature: Sampling temperature.
            format_json: If True, request JSON formatted output.
            model: Override the default model for this call.
            max_tokens: Maximum tokens to generate.
            stage: Pipeline stage spending these tokens (e.g. "triage").
            on_usage: Optional observer called with this call's TokenUsage.

        Returns:
            The full generated text.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if format_json:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                self._record_usage(
                    stage, kwargs["model"], getattr(response, "usage", None), on_usage
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt < self.max_retries and _is_retryable(e):
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "Groq generate attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt + 1, self.max_retries + 1, e, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error("Groq request failed: %s", e)
                raise

    async def generate_stream(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        stage: str = "",
        on_usage: Optional[UsageCallback] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Groq one at a time.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            temperature: Sampling temperature.
            model: Override the default model for this call.
            max_tokens: Maximum tokens to generate.
            stage: Pipeline stage spending these tokens (e.g. "synthesis").
            on_usage: Optional observer called once, after the final chunk.

        Yields:
            Individual tokens as they're generated.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resolved_model = model or self.model

        # Retry only while establishing the stream. Once tokens have started
        # flowing we can't safely restart without duplicating output.
        stream = None
        for attempt in range(self.max_retries + 1):
            try:
                stream = await self.client.chat.completions.create(
                    model=resolved_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    # Ask for a trailing usage chunk. Without this a streamed
                    # call reports no token counts at all — and the synthesizer,
                    # the most expensive call in the pipeline, is streamed.
                    stream_options={"include_usage": True},
                )
                break
            except Exception as e:
                if attempt < self.max_retries and _is_retryable(e):
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "Groq stream setup attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt + 1, self.max_retries + 1, e, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error("Groq stream failed: %s", e)
                raise

        raw_usage = None
        try:
            async for chunk in stream:
                # The usage chunk arrives last and carries an EMPTY choices list,
                # so choices must be checked before indexing it.
                if getattr(chunk, "usage", None) is not None:
                    raw_usage = chunk.usage
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content
                if token:
                    yield token
        except Exception as e:
            logger.error("Groq stream interrupted mid-flight: %s", e)
            raise
        finally:
            # Report whatever usage arrived, even if the stream broke partway —
            # a failed expensive call still cost tokens worth knowing about.
            self._record_usage(stage, resolved_model, raw_usage, on_usage)

    async def generate_structured(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        stage: str = "",
        on_usage: Optional[UsageCallback] = None,
    ) -> dict:
        """
        Generate structured JSON output from Groq.

        Args:
            prompt: The user prompt (should request JSON output).
            system: Optional system prompt.
            temperature: Lower temperature for more deterministic JSON.
            stage: Pipeline stage spending these tokens (e.g. "triage").
            on_usage: Optional observer called with this call's TokenUsage.

        Returns:
            Parsed JSON dictionary.
        """
        raw = await self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            format_json=True,
            stage=stage,
            on_usage=on_usage,
        )

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from Groq, attempting extraction")
            # Try to extract JSON from the response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(raw[start:end])
                except json.JSONDecodeError:
                    pass
            logger.error("Could not parse structured output: %s", raw[:200])
            return {}

    async def health_check(self, ttl: int = 30) -> bool:
        """
        Check if Groq API is reachable and responding.

        Caches the result for ``ttl`` seconds so repeated health polls don't
        burn Groq rate limit on the models endpoint.
        """
        now = time.monotonic()
        if self._health_cache is not None and (now - self._health_cache_at) < ttl:
            return self._health_cache

        try:
            models = await self.client.models.list()
            available = any(m.id == self.model for m in models.data)
            if not available:
                model_ids = [m.id for m in models.data]
                logger.warning(
                    "Model '%s' not found. Available: %s",
                    self.model,
                    model_ids,
                )
            self._health_cache = True
        except Exception as e:
            logger.error("Groq health check failed: %s", e)
            self._health_cache = False

        self._health_cache_at = now
        return self._health_cache


# Singleton instance
_client: Optional[GroqClient] = None


def get_llm_client() -> GroqClient:
    """Get the singleton Groq client."""
    global _client
    if _client is None:
        _client = GroqClient()
    return _client
