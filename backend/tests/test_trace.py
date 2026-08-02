"""
Unit tests for the reasoning trace.

The trace is the only place a user can see what the pipeline rejected, so the
tests that matter are about the three-way status: a source the answer cited, one
that reached the prompt and went unused, and one that was ranked but never made
the prompt's budget. Getting those confused would turn the feature into a lie —
it would show sources as "used" that the model never saw.
"""

import pytest

from app.models.schemas import Citation
from app.utils.trace import MAX_TRACE_SOURCES, build_trace, is_empty


def chunk(url, score, text="Some passage text about the topic.", title=None, domain=None):
    return {
        "source_url": url,
        "source_title": title or f"Title for {url}",
        "source_domain": domain or url.replace("https://", "").split("/")[0],
        "score": score,
        "text": text,
    }


def source(url):
    return {
        "url": url,
        "title": f"Title for {url}",
        "domain": url.replace("https://", "").split("/")[0],
    }


def citation(index, url):
    return Citation(
        index=index,
        source_url=url,
        source_title=f"Title for {url}",
        source_domain="example.com",
        claim="A claim.",
    )


A, B, C = "https://a.com", "https://b.com", "https://c.com"


def build(**overrides):
    kwargs = {
        "sub_queries": ["first sub-query", "second sub-query"],
        "all_chunks": [chunk(A, 0.9), chunk(B, 0.5), chunk(C, 0.2)],
        "ranked_chunks": [chunk(A, 0.9), chunk(B, 0.5), chunk(C, 0.2)],
        "cited_sources": [source(A), source(B)],
        "citations": [citation(1, A)],
        "reranker_model": "ms-marco-TinyBERT-L-2-v2",
    }
    kwargs.update(overrides)
    return build_trace(**kwargs)


def by_url(trace):
    return {s["url"]: s for s in trace["sources"]}


# ===========================================================================
# The three states
# ===========================================================================

class TestStatus:

    def test_cited_source_is_marked_cited_with_its_marker(self):
        entry = by_url(build())[A]
        assert entry["status"] == "cited"
        assert entry["citation_index"] == 1

    def test_source_in_the_prompt_but_uncited_is_sent(self):
        """The model saw it and chose not to use it — different from unseen."""
        entry = by_url(build())[B]
        assert entry["status"] == "sent"
        assert entry["citation_index"] is None

    def test_ranked_source_that_missed_the_prompt_is_considered(self):
        entry = by_url(build())[C]
        assert entry["status"] == "considered"
        assert entry["citation_index"] is None

    def test_every_ranked_source_appears(self):
        assert set(by_url(build())) == {A, B, C}


# ===========================================================================
# Aggregation
# ===========================================================================

class TestAggregation:

    def test_multiple_chunks_collapse_to_one_row_per_source(self):
        trace = build(
            ranked_chunks=[chunk(A, 0.9), chunk(A, 0.7), chunk(A, 0.3), chunk(B, 0.5)]
        )
        entries = by_url(trace)
        assert len(trace["sources"]) == 2
        assert entries[A]["chunks"] == 3

    def test_source_scored_by_its_best_chunk(self):
        trace = build(ranked_chunks=[chunk(A, 0.3), chunk(A, 0.95), chunk(A, 0.1)])
        assert by_url(trace)[A]["score"] == pytest.approx(0.95)

    def test_preview_comes_from_the_best_chunk(self):
        trace = build(
            ranked_chunks=[
                chunk(A, 0.2, text="the weak passage"),
                chunk(A, 0.9, text="the strong passage"),
            ]
        )
        assert "strong" in by_url(trace)[A]["preview"]

    def test_sorted_best_first(self):
        scores = [s["score"] for s in build()["sources"]]
        assert scores == sorted(scores, reverse=True)

    def test_chunks_without_a_url_are_skipped(self):
        trace = build(ranked_chunks=[chunk(A, 0.9), {"score": 0.5, "text": "orphan"}])
        assert len(trace["sources"]) == 1


# ===========================================================================
# Counts
# ===========================================================================

class TestCounts:

    def test_counts_describe_the_funnel(self):
        counts = build()["counts"]
        assert counts["chunks_ranked"] == 3
        assert counts["chunks_kept"] == 3
        assert counts["sources_considered"] == 3
        assert counts["sources_sent"] == 2
        assert counts["sources_cited"] == 1

    def test_chunks_ranked_reflects_everything_before_selection(self):
        """The funnel's top number is pre-rerank, not what survived it."""
        trace = build(
            all_chunks=[chunk(A, 0)] * 40,
            ranked_chunks=[chunk(A, 0.9), chunk(B, 0.5)],
        )
        assert trace["counts"]["chunks_ranked"] == 40
        assert trace["counts"]["chunks_kept"] == 2

    def test_counts_stay_truthful_when_the_source_list_is_capped(self):
        """The list is trimmed for payload size; the counts must not be."""
        many = [chunk(f"https://s{i}.com", 0.9 - i / 100) for i in range(40)]
        trace = build(all_chunks=many, ranked_chunks=many, cited_sources=[], citations=[])

        assert len(trace["sources"]) == MAX_TRACE_SOURCES
        assert trace["counts"]["sources_considered"] == 40

    def test_duplicate_citations_of_one_source_count_once(self):
        trace = build(citations=[citation(1, A), citation(1, A)])
        assert trace["counts"]["sources_cited"] == 1


# ===========================================================================
# Edges
# ===========================================================================

class TestEdges:

    def test_no_ranked_chunks_yields_an_empty_source_list(self):
        trace = build(all_chunks=[], ranked_chunks=[], cited_sources=[], citations=[])
        assert trace["sources"] == []
        assert trace["counts"]["sources_considered"] == 0

    def test_sub_queries_are_carried_through(self):
        assert build()["sub_queries"] == ["first sub-query", "second sub-query"]

    def test_reranker_model_is_reported(self):
        assert build()["reranker_model"] == "ms-marco-TinyBERT-L-2-v2"

    def test_long_preview_is_truncated(self):
        trace = build(ranked_chunks=[chunk(A, 0.9, text="x" * 500)])
        preview = by_url(trace)[A]["preview"]
        assert len(preview) < 500
        assert preview.endswith("…")

    def test_preview_whitespace_is_collapsed(self):
        trace = build(ranked_chunks=[chunk(A, 0.9, text="ragged\n\n  text   here")])
        assert by_url(trace)[A]["preview"] == "ragged text here"

    def test_trace_is_json_serializable(self):
        import json

        assert json.loads(json.dumps(build()))["counts"]["sources_cited"] == 1


class TestIsEmpty:

    def test_none_is_empty(self):
        assert is_empty(None) is True

    def test_empty_dict_is_empty(self):
        assert is_empty({}) is True

    def test_trace_with_sources_is_not_empty(self):
        assert is_empty(build()) is False

    def test_failed_search_still_counts_as_showable(self):
        """No sources but real sub-queries answers 'what did you even search?'."""
        trace = build(all_chunks=[], ranked_chunks=[], cited_sources=[], citations=[])
        assert is_empty(trace) is False
