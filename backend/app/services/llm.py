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
from app.services import tracing
# The retry predicate is shared with the search clients — see services.retry.
# This module keeps its own retry *loop* because each attempt has to update the
# surrounding Langfuse observation, which a generic helper has no business
# knowing about.
from app.services.retry import RETRYABLE_STATUS as _RETRYABLE_STATUS, is_retryable as _is_retryable

# TokenUsage/UsageCallback live in services.usage — the accounting module owns
# the accounting type. Re-exported here so callers can keep importing them from
# the client that produces them.
from app.services.usage import TokenUsage, UsageCallback, record_usage

__all__ = ["GroqClient", "TokenUsage", "UsageCallback", "get_llm_client"]

logger = logging.getLogger(__name__)


def stream_request_kwargs(
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> dict:
    """
    Build the arguments for a streaming completion.

    Extracted so a test can check every key against the installed SDK's
    signature. That guard exists because of a real outage: ``stream_options``
    was passed as a top-level keyword, which groq 0.25.0's client does not
    accept, and since a TypeError is not retryable every streamed synthesis
    failed instantly. Mocked tests could not see it — the mock accepted any
    keyword the real client would reject.

    ``extra_body`` is the SDK's pass-through to the HTTP payload, so the option
    reaches Groq's OpenAI-compatible API without the client needing to know
    about it.

    Worth knowing: Groq sends a usage-bearing final chunk even when this is not
    requested. We ask anyway — the explicit request is the documented contract,
    and relying on an undocumented convenience is how the counter silently
    goes to zero later.
    """
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "extra_body": {"stream_options": {"include_usage": True}},
    }


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

        # One observation spans the whole retry loop: the caller waited for all
        # of it, so that is the latency worth recording. Retries show up in the
        # metadata rather than as separate generations.
        with tracing.generation(
            stage or "llm",
            model=kwargs["model"],
            input=tracing.preview(prompt),
            metadata={"temperature": temperature, "json_mode": format_json},
        ) as observation:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self.client.chat.completions.create(**kwargs)
                    usage = self._record_usage(
                        stage, kwargs["model"], getattr(response, "usage", None), on_usage
                    )
                    content = response.choices[0].message.content or ""
                    observation.update(
                        output=tracing.preview(content),
                        usage_details=tracing.usage_details(usage),
                        cost_details=tracing.cost_details(usage),
                        metadata={"attempts": attempt + 1},
                    )
                    return content
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

        # The observation spans stream setup AND consumption, so its duration is
        # time-to-last-token — the number that actually describes this call.
        with tracing.generation(
            stage or "llm",
            model=resolved_model,
            input=tracing.preview(prompt),
            metadata={"temperature": temperature, "streamed": True},
        ) as observation:
            # Retry only while establishing the stream. Once tokens have started
            # flowing we can't safely restart without duplicating output.
            stream = None
            for attempt in range(self.max_retries + 1):
                try:
                    stream = await self.client.chat.completions.create(
                        **stream_request_kwargs(
                            resolved_model, messages, temperature, max_tokens
                        )
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
            collected: list[str] = []
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
                        collected.append(token)
                        yield token
            except Exception as e:
                logger.error("Groq stream interrupted mid-flight: %s", e)
                raise
            finally:
                # Report whatever usage arrived, even if the stream broke partway —
                # a failed expensive call still cost tokens worth knowing about.
                usage = self._record_usage(stage, resolved_model, raw_usage, on_usage)
                observation.update(
                    output=tracing.preview("".join(collected)),
                    usage_details=tracing.usage_details(usage),
                    cost_details=tracing.cost_details(usage),
                )

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
        Can this client actually serve a request right now?

        Two conditions, not one: Groq must answer, **and** the model this
        client is configured to call must be one Groq serves. Reporting only
        reachability is what made the 2026-08-22 outage (`72f8650`) invisible
        — Groq had decommissioned both Llama models, every layer degraded
        politely, the demo answered nothing, and this method kept returning
        True because the API itself was up. It logged the missing model on the
        line above and then discarded that fact.

        A missing model is an error, not a warning: nothing the app does will
        work until it is changed. It does not take the service down, though —
        ``/api/health`` gates its 503 on Postgres alone, so this surfaces as
        ``llm: disconnected`` on an instance that still serves cached reads.

        Caches for ``ttl`` seconds so repeated polls don't burn Groq rate
        limit on the models endpoint.
        """
        now = time.monotonic()
        if self._health_cache is not None and (now - self._health_cache_at) < ttl:
            return self._health_cache

        try:
            models = await self.client.models.list()
            available = any(m.id == self.model for m in models.data)
            if not available:
                logger.error(
                    "Model '%s' is not served by Groq — every call will 404. "
                    "Available: %s",
                    self.model,
                    [m.id for m in models.data],
                )
            self._health_cache = available
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
