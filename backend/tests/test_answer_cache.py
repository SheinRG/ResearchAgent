"""
Unit tests for the exact-match answer cache.

Only the pure logic is covered here — key derivation, the cacheability gate, and
the SSE replay contract. Redis get/set are thin wrappers over services.cache and
need no coverage of their own.

The key-derivation tests matter more than they look: this cache short-circuits
the entire pipeline, so a key that is too *loose* serves a user someone else's
answer, and one that is too *tight* never hits at all.
"""

import pytest

from app.services.answer_cache import (
    build_answer_key,
    cached_state_for_save,
    is_cacheable,
    iter_replay_events,
)

MODEL = "llama-3.3-70b-versatile"


def key(query, history=None, user_name="", model=MODEL):
    return build_answer_key(query, history, user_name, model)


# ===========================================================================
# build_answer_key
# ===========================================================================

class TestBuildAnswerKey:

    def test_identical_question_hits(self):
        assert key("What is RAG?") == key("What is RAG?")

    def test_normalizes_case_and_whitespace(self):
        """Trivial reformatting of the same question should still hit."""
        assert key("What is RAG?") == key("  what   IS   rag?  ")

    def test_different_questions_do_not_collide(self):
        assert key("What is RAG?") != key("What is CAG?")

    def test_preferred_name_changes_key(self):
        """The name is injected into the prompt, so it changes the answer."""
        assert key("What is RAG?") != key("What is RAG?", user_name="Raghav")

    def test_model_changes_key(self):
        """Answers from a different synthesis model must not be reused."""
        assert key("What is RAG?") != key("What is RAG?", model="some-other-model")

    def test_same_followup_in_different_conversations_does_not_collide(self):
        """
        The highest-risk collision: a short follow-up like "what about the CEO?"
        is textually identical across unrelated threads, but the correct answer
        depends entirely on what came before it.
        """
        tesla = [{"query": "Tell me about Tesla", "answer": "Tesla is an EV maker."}]
        ford = [{"query": "Tell me about Ford", "answer": "Ford is an automaker."}]
        assert key("What about the CEO?", tesla) != key("What about the CEO?", ford)

    def test_history_presence_changes_key(self):
        history = [{"query": "Earlier question", "answer": "Earlier answer"}]
        assert key("What is RAG?") != key("What is RAG?", history)

    def test_history_beyond_prompt_window_ignored(self):
        """
        Only the last few turns reach the prompt (see format_history), so older
        turns must not affect the key — otherwise a long thread never hits.
        """
        recent = [{"query": f"q{i}", "answer": f"a{i}"} for i in range(6, 10)]
        thread_a = [{"query": f"old{i}", "answer": f"x{i}"} for i in range(6)] + recent
        thread_b = [{"query": f"other{i}", "answer": f"y{i}"} for i in range(6)] + recent
        assert key("next?", thread_a) == key("next?", thread_b)

    def test_returns_hex_digest(self):
        result = key("anything")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


# ===========================================================================
# is_cacheable
# ===========================================================================

class TestIsCacheable:

    GOOD = {"draft_answer": "An answer [1].", "all_sources": [{"url": "https://a.com"}]}

    def test_successful_research_run_is_cacheable(self):
        assert is_cacheable(self.GOOD, has_documents=False) is True

    def test_uploads_bypass(self):
        """The document text is not in the key, so a hit could answer the wrong file."""
        assert is_cacheable(self.GOOD, has_documents=True) is False

    def test_failed_run_not_cached(self):
        assert is_cacheable({**self.GOOD, "error": "boom"}, False) is False

    def test_empty_answer_not_cached(self):
        assert is_cacheable({"draft_answer": "   ", "all_sources": [{"url": "u"}]}, False) is False

    def test_chat_reply_not_cached(self):
        """No sources means a chat reply: cheap to redo, and canned replies read as broken."""
        assert is_cacheable({"draft_answer": "Hey there!", "all_sources": []}, False) is False


# ===========================================================================
# iter_replay_events
# ===========================================================================

FULL_PAYLOAD = {
    "answer": "The answer [1].",
    "sources": [{"url": "https://a.com", "title": "A"}],
    "images": [{"url": "https://img.example/1.png"}],
    "sub_queries": ["sub one", "sub two"],
    "follow_ups": ["follow one"],
    "citations": [{"index": 1}],
    "confidence": 0.74,
}


class TestIterReplayEvents:

    def test_event_order_matches_live_pipeline(self):
        assert [t for t, _ in iter_replay_events(FULL_PAYLOAD)] == [
            "sub_queries", "sources", "images", "phase", "token", "follow_up",
        ]

    def test_sources_claim_authority_over_citation_order(self):
        """
        Without replace=True the UI merges these into whatever it already has,
        breaking the guarantee that marker [i] points at sources[i-1].
        """
        events = dict(iter_replay_events(FULL_PAYLOAD))
        assert events["sources"]["replace"] is True

    def test_answer_emitted_as_single_token(self):
        events = dict(iter_replay_events(FULL_PAYLOAD))
        assert events["token"]["token"] == "The answer [1]."

    def test_empty_sections_omitted(self):
        """A sparse payload must not emit empty sources/images/follow-up events."""
        assert [t for t, _ in iter_replay_events({"answer": "just text"})] == ["phase", "token"]

    def test_answer_always_emitted(self):
        events = dict(iter_replay_events({"answer": "x"}))
        assert "token" in events


# ===========================================================================
# cached_state_for_save
# ===========================================================================

class TestCachedStateForSave:

    def test_maps_payload_onto_graph_state_shape(self):
        """A cache hit is persisted through the same _save_session path as a live run."""
        state = cached_state_for_save(FULL_PAYLOAD)
        assert state["draft_answer"] == "The answer [1]."
        assert state["all_sources"] == FULL_PAYLOAD["sources"]
        assert state["sub_queries"] == FULL_PAYLOAD["sub_queries"]
        assert state["follow_up_suggestions"] == FULL_PAYLOAD["follow_ups"]
        assert state["citations"] == FULL_PAYLOAD["citations"]
        assert state["confidence"] == 0.74

    def test_tolerates_missing_fields(self):
        state = cached_state_for_save({"answer": "x"})
        assert state["draft_answer"] == "x"
        assert state["all_sources"] == []
        assert state["confidence"] == 0.0

    def test_entry_cached_before_integrity_tracking_reads_as_clean(self):
        """
        Payloads stored before invalid_citations existed have no such key. They
        must read as 0 — the same value a healthy answer reports — rather than
        raising or surfacing as a phantom warning in the UI.
        """
        assert cached_state_for_save({"answer": "x"})["invalid_citations"] == 0

    def test_carries_integrity_count_through_a_replay(self):
        payload = dict(FULL_PAYLOAD, invalid_citations=2)
        assert cached_state_for_save(payload)["invalid_citations"] == 2
