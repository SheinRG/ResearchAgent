"""
Unit tests for the eval runner's decision logic.

The runner decides whether the build passes. That decision has two halves —
hard invariants (never acceptable, whatever history says) and regressions
against a baseline (acceptable until they move too far) — and both are worth
testing directly, because a bug here means either a silent green on a broken
pipeline or a red build every Monday for no reason.

The expensive half of the runner (actually calling the graph) imports its
dependencies inside the function, so everything here runs with no keys and no
model packages installed.
"""

import json

import pytest
import yaml

from evals.run_eval import (
    TOLERANCE,
    compare_to_baseline,
    load_queries,
    render_markdown,
    summarize,
)
from evals.scorers import score_run

SOURCES = [{"url": "https://a.com", "title": "A", "domain": "a.com"}]


def make_score(qid="q", **state_overrides):
    base = {
        "draft_answer": "A properly cited claim about the topic in question [1].",
        "all_sources": SOURCES,
        "ranked_chunks": [{"score": 0.9}],
        "mode": "research",
        "answer_format": {"type": "prose"},
        "sub_queries": ["a"],
    }
    base.update(state_overrides)
    return score_run(
        spec={"id": qid, "query": "x"},
        final_state=base,
        latency_ms=5000,
        usage={"total_tokens": 1000, "cost_usd": 0.002},
    )


# ===========================================================================
# Query set loading and validation
# ===========================================================================

class TestLoadQueries:

    def test_real_query_set_loads(self):
        specs = load_queries()
        assert len(specs) >= 10
        assert all(s.get("id") and s.get("query") for s in specs)

    def test_real_query_set_covers_both_triage_modes(self):
        """A set with no chat questions cannot catch a triage routing regression."""
        categories = {s.get("category") for s in load_queries()}
        assert "chat" in categories
        assert {"table", "list", "steps", "prose"} <= categories

    def test_real_query_set_has_adversarial_cases(self):
        """The fabrication-bait questions are the point; don't let them vanish."""
        specs = load_queries()
        assert any(s.get("allow_fallback") for s in specs)

    def _write(self, tmp_path, data):
        path = tmp_path / "queries.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    def test_duplicate_ids_rejected(self, tmp_path):
        path = self._write(tmp_path, [
            {"id": "dup", "query": "a"},
            {"id": "dup", "query": "b"},
        ])
        with pytest.raises(ValueError, match="duplicate query id"):
            load_queries(path)

    def test_missing_id_rejected(self, tmp_path):
        path = self._write(tmp_path, [{"query": "a"}])
        with pytest.raises(ValueError, match="no id"):
            load_queries(path)

    def test_empty_query_text_rejected(self, tmp_path):
        path = self._write(tmp_path, [{"id": "a", "query": ""}])
        with pytest.raises(ValueError, match="empty query"):
            load_queries(path)

    def test_unknown_format_rejected(self, tmp_path):
        path = self._write(tmp_path, [
            {"id": "a", "query": "x", "expected_format": "spreadsheet"},
        ])
        with pytest.raises(ValueError, match="expected_format"):
            load_queries(path)

    def test_malformed_history_rejected(self, tmp_path):
        path = self._write(tmp_path, [
            {"id": "a", "query": "x", "history": [{"answer": "no query key"}]},
        ])
        with pytest.raises(ValueError, match="history turns"):
            load_queries(path)

    def test_empty_file_rejected(self, tmp_path):
        path = self._write(tmp_path, [])
        with pytest.raises(ValueError, match="non-empty list"):
            load_queries(path)


# ===========================================================================
# Aggregation
# ===========================================================================

class TestSummarize:

    def test_healthy_run_reports_no_hard_failures(self):
        scores = [make_score("q1"), make_score("q2")]
        specs = {"q1": {"id": "q1"}, "q2": {"id": "q2"}}
        summary = summarize(scores, specs)

        assert summary["hard_failures"] == {}
        assert summary["queries"] == 2
        assert summary["total_invalid_citations"] == 0

    def test_fabricated_citations_are_counted_and_flagged(self):
        scores = [make_score("q1", draft_answer="Invented source [9].")]
        summary = summarize(scores, {"q1": {"id": "q1"}})

        assert summary["total_invalid_citations"] == 1
        assert "q1" in summary["hard_failures"]

    def test_allowed_fallback_is_not_a_hard_failure(self):
        from app.agents.messages import NO_SOURCES_MESSAGE

        scores = [make_score("adv", draft_answer=NO_SOURCES_MESSAGE, all_sources=[])]
        summary = summarize(scores, {"adv": {"id": "adv", "allow_fallback": True}})

        assert summary["hard_failures"] == {}

    def test_chat_replies_excluded_from_research_averages(self):
        """A chat reply cites nothing; averaging it in would drag coverage down."""
        chat = make_score(
            "chat", draft_answer="Hey!", all_sources=[], mode="chat",
        )
        research = make_score("q1")
        summary = summarize([chat, research], {"chat": {"id": "chat"}, "q1": {"id": "q1"}})

        assert summary["mean_coverage"] == pytest.approx(research.coverage)

    def test_errored_runs_counted_separately(self):
        errored = score_run(spec={"id": "boom", "query": "x"}, final_state={}, error="Timeout")
        summary = summarize([errored, make_score("q1")], {"boom": {"id": "boom"}, "q1": {"id": "q1"}})

        assert summary["errors"] == 1
        assert "boom" in summary["hard_failures"]

    def test_costs_and_tokens_are_totalled(self):
        summary = summarize(
            [make_score("q1"), make_score("q2")],
            {"q1": {"id": "q1"}, "q2": {"id": "q2"}},
        )
        assert summary["total_cost_usd"] == pytest.approx(0.004)
        assert summary["total_tokens"] == 2000

    def test_summary_is_json_serializable(self):
        summary = summarize([make_score("q1")], {"q1": {"id": "q1"}})
        assert json.loads(json.dumps(summary))["queries"] == 1


# ===========================================================================
# Regression detection
# ===========================================================================

class TestCompareToBaseline:

    def test_no_baseline_means_no_regressions(self):
        summary = summarize([make_score("q1")], {"q1": {"id": "q1"}})
        assert compare_to_baseline(summary, None) == []

    def test_identical_numbers_are_not_a_regression(self):
        summary = summarize([make_score("q1")], {"q1": {"id": "q1"}})
        assert compare_to_baseline(summary, dict(summary)) == []

    def test_small_drop_within_tolerance_is_ignored(self):
        """Temperature 0.4 and a changing web make small movements meaningless."""
        summary = {"mean_coverage": 0.80}
        baseline = {"mean_coverage": 0.80 + TOLERANCE["mean_coverage"] - 0.01}
        assert compare_to_baseline(summary, baseline) == []

    def test_large_drop_is_flagged(self):
        summary = {"mean_coverage": 0.30}
        baseline = {"mean_coverage": 0.85}
        regressions = compare_to_baseline(summary, baseline)

        assert len(regressions) == 1
        assert "mean_coverage" in regressions[0]

    def test_improvement_is_never_a_regression(self):
        assert compare_to_baseline({"mean_coverage": 0.99}, {"mean_coverage": 0.50}) == []

    def test_metric_absent_from_baseline_is_skipped(self):
        """An older baseline missing a newly added metric must not fail the run."""
        assert compare_to_baseline({"mean_coverage": 0.1}, {"queries": 20}) == []

    def test_every_tracked_metric_can_be_flagged(self):
        summary = {metric: 0.0 for metric in TOLERANCE}
        baseline = {metric: 1.0 for metric in TOLERANCE}
        assert len(compare_to_baseline(summary, baseline)) == len(TOLERANCE)


# ===========================================================================
# Reporting
# ===========================================================================

class TestRenderMarkdown:

    def test_clean_run_renders_pass(self):
        scores = [make_score("q1")]
        summary = summarize(scores, {"q1": {"id": "q1"}})
        report = render_markdown(summary, scores, [])

        assert "PASS" in report
        assert "q1" in report

    def test_hard_failure_renders_fail_with_the_reason(self):
        scores = [make_score("q1", draft_answer="Invented [9].")]
        summary = summarize(scores, {"q1": {"id": "q1"}})
        report = render_markdown(summary, scores, [])

        assert "FAIL" in report
        assert "Hard failures" in report
        assert "fabricated citation markers" in report

    def test_regressions_are_listed(self):
        scores = [make_score("q1")]
        summary = summarize(scores, {"q1": {"id": "q1"}})
        report = render_markdown(summary, scores, ["mean_coverage: 30% vs baseline 85%"])

        assert "FAIL" in report
        assert "Regressions" in report
