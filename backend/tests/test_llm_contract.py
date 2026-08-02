"""
Contract tests between our LLM client and the installed Groq SDK.

These exist because of an outage that every other test missed. A streaming
request was sending ``stream_options`` as a top-level keyword; groq 0.25.0's
client does not accept it, so `AsyncCompletions.create()` raised TypeError.
TypeError is not retryable, so it propagated instantly and *every* streamed
synthesis returned the error fallback — while the unit tests stayed green,
because they mock the Groq client and a mock happily accepts any keyword the
real client would reject.

So this file deliberately does not mock. It introspects the real, installed SDK
and checks that the arguments we actually send are ones it can actually take.
No network, no API key, no tokens spent — just signatures.
"""

import inspect

import pytest

from app.services.llm import stream_request_kwargs

groq_completions = pytest.importorskip(
    "groq.resources.chat.completions",
    reason="groq SDK not installed in this environment (it is in CI)",
)

CREATE_PARAMS = inspect.signature(groq_completions.AsyncCompletions.create).parameters

MESSAGES = [{"role": "user", "content": "hello"}]


def kwargs():
    return stream_request_kwargs("llama-3.3-70b-versatile", MESSAGES, 0.4, 2000)


class TestStreamRequestKwargs:

    def test_every_argument_is_accepted_by_the_installed_sdk(self):
        """
        The regression guard. If this fails, the streaming path is broken in
        production no matter what the mocked tests say.
        """
        unsupported = [k for k in kwargs() if k not in CREATE_PARAMS]
        assert not unsupported, (
            f"groq's AsyncCompletions.create() does not accept {unsupported}. "
            "A TypeError here is not retryable, so every streamed answer would "
            "fail. Pass the option through extra_body instead."
        )

    def test_usage_is_requested_via_extra_body_not_a_top_level_kwarg(self):
        """
        ``stream_options`` must ride inside extra_body. Passing it at the top
        level is exactly the mistake this file was written for.
        """
        sent = kwargs()
        assert "stream_options" not in sent
        assert sent["extra_body"]["stream_options"] == {"include_usage": True}

    def test_extra_body_is_a_real_sdk_parameter(self):
        """The pass-through we depend on must exist in the pinned SDK."""
        assert "extra_body" in CREATE_PARAMS

    def test_streaming_is_actually_requested(self):
        assert kwargs()["stream"] is True

    def test_model_and_generation_settings_are_passed_through(self):
        sent = stream_request_kwargs("some-model", MESSAGES, 0.7, 1234)
        assert sent["model"] == "some-model"
        assert sent["temperature"] == 0.7
        assert sent["max_tokens"] == 1234
        assert sent["messages"] is MESSAGES
