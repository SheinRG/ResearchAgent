"""
Unit tests for the eval scorers.

The scorers are the checklist the eval harness applies. They are pure functions,
so they are tested here on fixed text with no pipeline, no keys and no network —
which is the whole reason the checklist was separated from the run: this file
runs on every pull request, while the live run happens weekly.

A scorer that quietly stops detecting something is worse than no scorer, because
the dashboard keeps showing a healthy number. Hence the emphasis below on the
negative cases: text that must NOT be counted as a claim, a table that must NOT
pass as well-formed, a reworded fallback that must still be recognised.
"""

import pytest

from app.agents.messages import (
    CHAT_ERROR_MESSAGE,
    NO_SOURCES_MESSAGE,
    SYNTHESIS_ERROR_MESSAGE,
)
from evals.scorers import (
    citation_coverage,
    claim_units,
    count_list_items,
    count_ordered_items,
    format_compliance,
    has_table,
    invalid_citations,
    is_fallback,
    score_run,
    source_utilization,
    strip_code_blocks,
    table_columns,
)

SOURCES = [
    {"url": "https://a.com", "title": "A", "domain": "a.com"},
    {"url": "https://b.com", "title": "B", "domain": "b.com"},
    {"url": "https://c.com", "title": "C", "domain": "c.com"},
]

TABLE_ANSWER = """Here is the comparison you asked for.

| Tool | Cost | Best for |
| --- | --- | --- |
| Alpha | Free tier available [1] | Small projects [1] |
| Beta | $20 per month [2] | Production workloads [2] |

Both options scale to a few million vectors without much tuning [3].
"""


# ===========================================================================
# claim_units — what counts as something needing a citation
# ===========================================================================

class TestClaimUnits:

    def test_prose_splits_into_sentences(self):
        text = (
            "Retrieval augmented generation grounds a model in external documents. "
            "It reduces hallucination by supplying evidence at inference time."
        )
        assert len(claim_units(text)) == 2

    def test_headings_are_not_claims(self):
        units = claim_units("## Background\n\n### Why it matters")
        assert units == []

    def test_short_scaffolding_lines_are_not_claims(self):
        """'Key points:' announces content; it asserts nothing."""
        assert claim_units("Key points:") == []

    def test_table_header_and_separator_are_not_claims(self):
        units = claim_units("| Tool | Cost |\n| --- | --- |\n")
        assert units == []

    def test_table_data_rows_are_claims(self):
        units = claim_units(
            "| Tool | Cost |\n| --- | --- |\n| Alpha | Free [1] |\n| Beta | $20 [2] |"
        )
        assert len(units) == 2

    def test_list_items_are_claims(self):
        text = (
            "- Prompt injection can reach any tool the agent is allowed to call [1]\n"
            "- Credentials in the environment are readable by any executed code [2]\n"
        )
        assert len(claim_units(text)) == 2

    def test_short_list_items_are_ignored(self):
        assert claim_units("- yes\n- no\n") == []

    def test_code_blocks_are_excluded(self):
        text = (
            "Install it with the package manager of your choice.\n"
            "```bash\npip install something-that-is-quite-long-here\n```\n"
        )
        units = claim_units(text)
        assert all("pip install" not in u for u in units)

    def test_mixed_answer_counts_rows_and_prose_but_not_the_intro(self):
        """
        Two table rows plus the closing sentence — but NOT "Here is the
        comparison you asked for.", which is scaffolding announcing the table
        rather than a claim that needs a source.
        """
        units = claim_units(TABLE_ANSWER)
        assert len(units) == 3
        assert not any(u.startswith("Here is the comparison") for u in units)


class TestStripCodeBlocks:

    def test_removes_fenced_content(self):
        assert "secret" not in strip_code_blocks("before\n```\nsecret\n```\nafter")

    def test_keeps_surrounding_text(self):
        result = strip_code_blocks("before\n```\nx\n```\nafter")
        assert "before" in result and "after" in result


# ===========================================================================
# Citation integrity
# ===========================================================================

class TestInvalidCitations:

    def test_clean_answer_has_none(self):
        assert invalid_citations("A claim [1] and another [2].", SOURCES) == []

    def test_marker_beyond_the_source_list_is_caught(self):
        assert invalid_citations("Invented [7].", SOURCES) == [7]

    def test_empty_answer_has_none(self):
        assert invalid_citations("", SOURCES) == []


class TestCitationCoverage:

    def test_fully_cited_answer(self):
        text = (
            "Retrieval augmented generation grounds a model in documents [1]. "
            "It measurably reduces fabricated detail in long answers [2]."
        )
        assert citation_coverage(text) == (2, 2)

    def test_partially_cited_answer(self):
        text = (
            "Retrieval augmented generation grounds a model in documents [1]. "
            "Many teams adopted it during the past few years without measuring it."
        )
        cited, total = citation_coverage(text)
        assert (cited, total) == (1, 2)

    def test_answer_with_no_claims_reports_zero_total(self):
        """Nothing to cite must read as 0/0, not as a 0% failure."""
        assert citation_coverage("## Heading only") == (0, 0)


class TestSourceUtilization:

    def test_counts_distinct_sources_used(self):
        used, provided = source_utilization("One [1] two [2] one again [1].", SOURCES)
        assert (used, provided) == (2, 3)

    def test_invalid_markers_do_not_inflate_usage(self):
        used, provided = source_utilization("Real [1] invented [9].", SOURCES)
        assert (used, provided) == (1, 3)

    def test_no_sources_reports_zero_denominator(self):
        assert source_utilization("Anything [1].", []) == (0, 0)


# ===========================================================================
# Fallback detection
# ===========================================================================

class TestIsFallback:

    @pytest.mark.parametrize(
        "message",
        [NO_SOURCES_MESSAGE, SYNTHESIS_ERROR_MESSAGE, CHAT_ERROR_MESSAGE],
    )
    def test_detects_every_canned_message(self, message):
        assert is_fallback(message) is True

    def test_real_answer_is_not_a_fallback(self):
        assert is_fallback("Retrieval augmented generation works by [1].") is False

    def test_empty_answer_is_not_a_fallback(self):
        """Empty is its own failure — the 'answered' invariant catches it."""
        assert is_fallback("") is False

    def test_detects_fallback_embedded_in_a_longer_string(self):
        assert is_fallback(f"{NO_SOURCES_MESSAGE}\n\nTry again later.") is True


# ===========================================================================
# Format compliance — the triage → synthesizer contract
# ===========================================================================

class TestTableDetection:

    def test_well_formed_table_detected(self):
        assert has_table(TABLE_ANSWER) is True

    def test_column_count_read_from_the_header(self):
        assert table_columns(TABLE_ANSWER) == 3

    def test_pipes_without_a_separator_row_are_not_a_table(self):
        assert has_table("| Tool | Cost |\n| Alpha | Free |") is False

    def test_prose_is_not_a_table(self):
        assert has_table("No table here at all.") is False

    def test_table_inside_a_code_block_does_not_count(self):
        fenced = "```\n| A | B |\n| --- | --- |\n| 1 | 2 |\n```"
        assert has_table(fenced) is False


class TestListCounting:

    def test_counts_bullets(self):
        assert count_list_items("- one\n- two\n- three") == 3

    def test_counts_numbered_items(self):
        assert count_ordered_items("1. first\n2. second") == 2

    def test_bullets_are_not_counted_as_ordered(self):
        assert count_ordered_items("- one\n- two") == 0

    def test_table_rows_are_not_list_items(self):
        assert count_list_items(TABLE_ANSWER) == 0


class TestFormatCompliance:

    def test_table_promised_and_delivered(self):
        assert format_compliance(TABLE_ANSWER, "table") is True

    def test_table_promised_but_prose_returned(self):
        assert format_compliance("Just a paragraph of prose.", "table") is False

    def test_steps_need_an_ordered_list(self):
        assert format_compliance("1. Install it\n2. Configure it", "steps") is True
        assert format_compliance("- Install it\n- Configure it", "steps") is False

    def test_list_accepts_bullets_or_numbers(self):
        assert format_compliance("- a\n- b", "list") is True
        assert format_compliance("1. a\n2. b", "list") is True

    def test_single_item_is_not_a_list(self):
        assert format_compliance("- only one", "list") is False

    def test_prose_always_passes(self):
        """Prose is the absence of a structural requirement, so nothing to check."""
        assert format_compliance("Anything at all.", "prose") is True

    def test_unknown_format_passes_rather_than_failing_loudly(self):
        assert format_compliance("Anything.", "") is True


# ===========================================================================
# score_run — the whole checklist against one finished run
# ===========================================================================

def state(**overrides):
    base = {
        "draft_answer": "A well cited claim about the subject at hand [1].",
        "all_sources": SOURCES,
        "ranked_chunks": [{"score": 0.91}, {"score": 0.42}],
        "mode": "research",
        "answer_format": {"type": "prose", "reasoning": "", "columns": []},
        "sub_queries": ["a", "b"],
    }
    base.update(overrides)
    return base


class TestScoreRun:

    def test_healthy_run_has_no_failures(self):
        score = score_run(spec={"id": "q1", "query": "x"}, final_state=state())
        assert score.failures() == []
        assert score.answered is True

    def test_fabricated_citation_is_a_hard_failure(self):
        score = score_run(
            spec={"id": "q1", "query": "x"},
            final_state=state(draft_answer="Invented reference [9]."),
        )
        assert any("fabricated" in f for f in score.failures())

    def test_empty_answer_is_a_hard_failure(self):
        score = score_run(
            spec={"id": "q1", "query": "x"}, final_state=state(draft_answer="")
        )
        assert any("empty answer" in f for f in score.failures())

    def test_fallback_fails_unless_the_query_allows_it(self):
        score = score_run(
            spec={"id": "q1", "query": "x"},
            final_state=state(draft_answer=NO_SOURCES_MESSAGE, all_sources=[]),
        )
        assert score.failures() != []
        assert score.failures(allow_fallback=True) == []

    def test_research_answer_without_sources_fails(self):
        score = score_run(
            spec={"id": "q1", "query": "x"},
            final_state=state(all_sources=[]),
        )
        assert any("no sources" in f for f in score.failures())

    def test_chat_reply_is_not_judged_on_format_or_sources(self):
        score = score_run(
            spec={"id": "chat", "query": "hi", "expected_format": "chat"},
            final_state=state(
                draft_answer="Hey! How can I help?",
                all_sources=[],
                mode="chat",
                answer_format={"type": "prose"},
            ),
        )
        assert score.failures() == []
        assert score.format_ok is True
        assert score.triage_format_match is True

    def test_missing_required_term_is_a_hard_failure(self):
        score = score_run(
            spec={"id": "q1", "query": "x", "must_mention": ["retrieval"]},
            final_state=state(),
        )
        assert any("missing required terms" in f for f in score.failures())

    def test_required_term_matches_case_insensitively(self):
        score = score_run(
            spec={"id": "q1", "query": "x", "must_mention": ["CLAIM"]},
            final_state=state(),
        )
        assert score.missing_terms == []

    def test_triage_format_mismatch_is_tracked_but_not_fatal(self):
        """Triage choosing prose where we expected a table is a trend, not a break."""
        score = score_run(
            spec={"id": "q1", "query": "x", "expected_format": "table"},
            final_state=state(),
        )
        assert score.triage_format_match is False
        assert score.failures() == []

    def test_declared_table_not_delivered_is_tracked(self):
        score = score_run(
            spec={"id": "q1", "query": "x", "expected_format": "table"},
            final_state=state(answer_format={"type": "table", "columns": ["A", "B"]}),
        )
        assert score.format_ok is False

    def test_errored_run_short_circuits_to_one_failure(self):
        score = score_run(
            spec={"id": "q1", "query": "x"},
            final_state={},
            error="TimeoutError: too slow",
        )
        assert score.failures() == ["run failed: TimeoutError: too slow"]

    def test_carries_cost_and_latency_through(self):
        score = score_run(
            spec={"id": "q1", "query": "x"},
            final_state=state(),
            latency_ms=8200,
            usage={"total_tokens": 4700, "cost_usd": 0.0031},
        )
        assert score.latency_ms == 8200
        assert score.total_tokens == 4700
        assert score.cost_usd == 0.0031

    def test_as_dict_is_json_serializable(self):
        import json

        score = score_run(spec={"id": "q1", "query": "x"}, final_state=state())
        assert json.loads(json.dumps(score.as_dict()))["id"] == "q1"
