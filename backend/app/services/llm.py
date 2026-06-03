"""
Async Ollama client for local LLM inference.
Supports regular generation, streaming, and structured JSON output.
"""

import json
import logging
from typing import AsyncGenerator, Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async wrapper around the Ollama HTTP API."""

    def __init__(self):
        settings = get_settings()
        self.base_url = settings.ollama_host
        self.model = settings.ollama_model
        self.timeout = settings.ollama_timeout

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        format_json: bool = False,
    ) -> str:
        """
        Generate a complete response from Ollama.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            temperature: Sampling temperature.
            format_json: If True, request JSON formatted output.

        Returns:
            The full generated text.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2048,
            },
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except httpx.TimeoutException:
            logger.error("Ollama request timed out after %ds", self.timeout)
            raise
        except httpx.HTTPStatusError as e:
            logger.error("Ollama HTTP error: %s", e.response.status_code)
            raise
        except Exception as e:
            logger.error("Ollama request failed: %s", e)
            raise

    async def generate_stream(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Ollama one at a time.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            temperature: Sampling temperature.

        Yields:
            Individual tokens as they're generated.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": 4096,
            },
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                token = data.get("response", "")
                                if token:
                                    yield token
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
        except httpx.TimeoutException:
            logger.error("Ollama stream timed out after %ds", self.timeout)
            raise
        except Exception as e:
            logger.error("Ollama stream failed: %s", e)
            raise

    async def generate_structured(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
    ) -> dict:
        """
        Generate structured JSON output from Ollama.

        Args:
            prompt: The user prompt (should request JSON output).
            system: Optional system prompt.
            temperature: Lower temperature for more deterministic JSON.

        Returns:
            Parsed JSON dictionary.
        """
        raw = await self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            format_json=True,
        )

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from Ollama, attempting extraction")
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

    async def health_check(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                # Check if our model (or its base name) is available
                model_base = self.model.split(":")[0]
                available = any(model_base in m for m in models)
                if not available:
                    logger.warning(
                        "Model '%s' not found. Available: %s",
                        self.model,
                        models,
                    )
                return available
        except Exception as e:
            logger.error("Ollama health check failed: %s", e)
            return False


# Singleton instance
_client: Optional[OllamaClient] = None


def get_llm_client() -> OllamaClient:
    """Get the singleton Ollama client."""
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
