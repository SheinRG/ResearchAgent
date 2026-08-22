"""
Unit tests for the Langfuse tracing wrapper.

Tracing sits directly on the request path, so the property that matters most is
that it cannot break a request. These tests drive the wrapper against a fake
client that fails in each of the ways a real one can — refusing to start a span,
raising on update, raising on close — and assert the caller never notices.

No Langfuse account or network is involved: the module holds its client in a
module-level global, so a fake slots straight in.
"""

import pytest

from app.config import get_settings
from app.services import tracing
from app.services.usage import TokenUsage


# --- Fakes -----------------------------------------------------------------


class _FakeObservation:
    def __init__(self, recorder, name):
        self._recorder = recorder
        self.name = name
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class _FakeSpanCM:
    """Mimics the context manager returned by start_as_current_observation."""

    def __init__(self, recorder, name, raise_on_exit=False):
        self._recorder = recorder
        self._name = name
        self._raise_on_exit = raise_on_exit
        self.observation = _FakeObservation(recorder, name)
        self.exited_with = None

    def __enter__(self):
        return self.observation

    def __exit__(self, exc_type, exc, tb):
        self.exited_with = exc_type
        self._recorder.closed.append(self._name)
        if self._raise_on_exit:
            raise RuntimeError("exporter is down")
        return False


class _FakeClient:
    def __init__(self, fail_start=False, fail_exit=False):
        self.started = []
        self.closed = []
        self.spans = []
        self._fail_start = fail_start
        self._fail_exit = fail_exit

    def start_as_current_observation(self, *, as_type, name, **kwargs):
        if self._fail_start:
            raise RuntimeError("no exporter configured")
        self.started.append({"as_type": as_type, "name": name, **kwargs})
        cm = _FakeSpanCM(self, name, raise_on_exit=self._fail_exit)
        self.spans.append(cm)
        return cm


@pytest.fixture
def fake_client():
    """Install a fake Langfuse client for the duration of one test."""
    client = _FakeClient()
    tracing._client = client
    tracing._capture_content = True
    tracing._max_content_chars = 2000
    yield client
    tracing._client = None


@pytest.fixture(autouse=True)
def reset_module_state():
    """Every test starts from the disabled default, whatever the last one did."""
    yield
    tracing._client = None
    tracing._propagate_attributes = None
    tracing._capture_content = True
    tracing._max_content_chars = 2000
    # Settings are lru_cached; a test that edited the environment must not leak
    # that into the next one.
    get_settings.cache_clear()


# ===========================================================================
# Disabled by default
# ===========================================================================

class TestDisabled:

    def test_disabled_without_keys(self):
        assert tracing.is_enabled() is False

    def test_span_still_works_and_yields_a_null_observation(self):
        with tracing.span("triage") as observation:
            observation.update(output={"anything": 1}, level="ERROR")
        assert observation is tracing.NULL_OBSERVATION

    def test_generation_still_works(self):
        with tracing.generation("synthesis", model="openai/gpt-oss-120b") as gen:
            gen.update(output="hello", usage_details={"input": 1, "output": 2})

    def test_request_scope_is_transparent(self):
        with tracing.request_scope(user_id="u1", session_id="s1", tags=["research"]):
            pass

    def test_exceptions_propagate_unchanged(self):
        with pytest.raises(ValueError, match="pipeline blew up"):
            with tracing.span("search"):
                raise ValueError("pipeline blew up")

    def test_shutdown_without_a_client_is_a_noop(self):
        tracing.shutdown_tracing()


# ===========================================================================
# Enabled
# ===========================================================================

class TestEnabled:

    def test_span_opens_and_closes(self, fake_client):
        with tracing.span("rerank", input="what is RAG?"):
            pass

        assert fake_client.started[0]["as_type"] == "span"
        assert fake_client.started[0]["name"] == "rerank"
        assert fake_client.closed == ["rerank"]

    def test_generation_records_type_and_model(self, fake_client):
        with tracing.generation("synthesis", model="openai/gpt-oss-120b") as gen:
            gen.update(output="answer", usage_details={"input": 10, "output": 5})

        assert fake_client.started[0]["as_type"] == "generation"
        assert fake_client.started[0]["model"] == "openai/gpt-oss-120b"
        assert fake_client.spans[0].observation.updates[0]["output"] == "answer"

    def test_nested_spans_all_close(self, fake_client):
        with tracing.span("research"):
            with tracing.span("triage"):
                pass
            with tracing.span("search"):
                pass

        assert fake_client.closed == ["triage", "search", "research"]

    def test_error_in_body_marks_span_then_reraises(self, fake_client):
        with pytest.raises(RuntimeError, match="groq exploded"):
            with tracing.span("synthesize"):
                raise RuntimeError("groq exploded")

        update = fake_client.spans[0].observation.updates[0]
        assert update["level"] == "ERROR"
        assert "groq exploded" in update["status_message"]
        assert fake_client.spans[0].exited_with is RuntimeError

    def test_is_enabled_reflects_the_client(self, fake_client):
        assert tracing.is_enabled() is True


# ===========================================================================
# Failure isolation — the property that actually matters
# ===========================================================================

class TestNeverBreaksTheRequest:

    def test_span_that_cannot_start_degrades_to_noop(self):
        tracing._client = _FakeClient(fail_start=True)

        with tracing.span("triage") as observation:
            observation.update(output={"mode": "research"})
        assert observation is tracing.NULL_OBSERVATION

    def test_failure_to_close_is_swallowed(self):
        tracing._client = _FakeClient(fail_exit=True)

        with tracing.span("search"):
            pass  # __exit__ raises inside the SDK; caller must not see it

    def test_failure_to_close_does_not_mask_a_real_error(self):
        """An exporter failure during cleanup must not replace the app's exception."""
        tracing._client = _FakeClient(fail_exit=True)

        with pytest.raises(ValueError, match="real problem"):
            with tracing.span("search"):
                raise ValueError("real problem")

    def test_update_failure_is_swallowed(self, fake_client):
        with tracing.span("rerank") as observation:
            observation._raw.update = lambda **_: (_ for _ in ()).throw(
                RuntimeError("serialization failed")
            )
            observation.update(output={"ranked_chunks": 12})

    def test_init_without_keys_leaves_tracing_off(self, monkeypatch):
        """
        The default deployment has no Langfuse keys. Init must be silent, must
        not raise, and must leave every helper in its no-op state.

        Deliberately not testing the *configured* path here: it would either
        construct a real client that tries to export to Langfuse Cloud from CI,
        or pass vacuously wherever the SDK is absent. The failure modes of a
        configured-but-broken client are covered above with a fake.
        """
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        get_settings.cache_clear()

        tracing.init_tracing()

        assert tracing.is_enabled() is False
        with tracing.span("triage") as observation:
            assert observation is tracing.NULL_OBSERVATION

    def test_shutdown_returns_to_the_disabled_state(self, fake_client):
        assert tracing.is_enabled() is True
        tracing.shutdown_tracing()
        assert tracing.is_enabled() is False


# ===========================================================================
# Content shaping
# ===========================================================================

class TestPreview:

    def test_short_strings_pass_through(self):
        assert tracing.preview("what is RAG?") == "what is RAG?"

    def test_long_strings_are_truncated_with_a_marker(self):
        tracing._max_content_chars = 50
        result = tracing.preview("x" * 500)

        assert len(result) < 500
        assert result.startswith("x" * 50)
        assert "+450 chars" in result

    def test_capture_off_drops_content_entirely(self):
        tracing._capture_content = False
        assert tracing.preview("a user's private question") is None

    def test_capture_off_still_allows_non_content_metadata(self):
        """Counts and timings are not content and must survive the switch."""
        tracing._capture_content = False
        assert tracing.preview(None) is None

    def test_non_string_values_pass_through(self):
        assert tracing.preview({"sources": 8}) == {"sources": 8}


class TestUsageDetails:

    def test_maps_token_usage_to_langfuse_shape(self):
        usage = TokenUsage(
            stage="synthesis",
            model="openai/gpt-oss-120b",
            prompt_tokens=4000,
            completion_tokens=700,
            total_tokens=4700,
        )
        assert tracing.usage_details(usage) == {
            "input": 4000,
            "output": 700,
            "total": 4700,
        }

    def test_none_usage_maps_to_none(self):
        """A call whose usage the API omitted must not report zero tokens."""
        assert tracing.usage_details(None) is None


class TestCostDetails:
    """
    Costs are sent explicitly because Langfuse's own price table has no Groq
    models — without this the dashboard would report $0 for every trace.
    """

    def test_splits_cost_into_input_and_output(self):
        usage = TokenUsage(
            stage="synthesis",
            model="openai/gpt-oss-120b",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            total_tokens=2_000_000,
        )
        details = tracing.cost_details(usage)

        # 1M tokens each way at gpt-oss-120b's published rate, so the numbers
        # are the price table itself. They move whenever MODEL_PRICES does.
        assert details["input"] == pytest.approx(0.15)
        assert details["output"] == pytest.approx(0.60)
        assert details["total"] == pytest.approx(0.75)

    def test_agrees_with_the_figure_shown_in_the_ui(self):
        """The trace and the done bar must not disagree about what a turn cost."""
        from app.services.usage import cost_of

        usage = TokenUsage(
            stage="synthesis",
            model="openai/gpt-oss-120b",
            prompt_tokens=4000,
            completion_tokens=700,
            total_tokens=4700,
        )
        assert tracing.cost_details(usage)["total"] == pytest.approx(
            cost_of(usage.model, usage.prompt_tokens, usage.completion_tokens)
        )

    def test_unpriced_model_reports_nothing_rather_than_zero(self):
        usage = TokenUsage(
            stage="synthesis",
            model="some-future-model",
            prompt_tokens=1000,
            completion_tokens=100,
            total_tokens=1100,
        )
        assert tracing.cost_details(usage) is None

    def test_none_usage_maps_to_none(self):
        assert tracing.cost_details(None) is None
