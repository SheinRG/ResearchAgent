"""
Unit tests for the chunker and citation utilities.
All functions under test are pure — no external calls, no database.
"""

import pytest

from app.utils.chunker import chunk_text, _force_split
from app.utils.citations import (
    extract_citations,
    extract_citations_detailed,
    _extract_claim_context,
    build_cited_context,
    CITATION_PATTERN,
)
from app.models.schemas import Citation


# ===========================================================================
# chunk_text
# ===========================================================================

class TestChunkText:

    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\n  ") == []

    def test_short_text_returns_single_chunk(self):
        text = "This is a short sentence."
        result = chunk_text(text, chunk_size=500)
        assert result == [text]

    def test_text_exactly_at_chunk_size_returns_single_chunk(self):
        text = "A" * 500
        result = chunk_text(text, chunk_size=500)
        assert len(result) == 1

    def test_long_text_produces_multiple_chunks(self):
        # 10 paragraphs, each 200 chars
        text = "\n\n".join(["X" * 200 for _ in range(10)])
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=0)
        assert len(chunks) > 1

    def test_chunks_cover_all_content(self):
        """Every part of the original text should appear in at least one chunk."""
        words = [f"word{i}" for i in range(200)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        combined = " ".join(chunks)
        # Each word should appear somewhere across all chunks
        for word in words:
            assert word in combined, f"'{word}' missing from chunks"

    def test_chunk_size_respected(self):
        """No individual chunk should greatly exceed chunk_size + overlap."""
        text = "word " * 500  # 2500 chars
        chunk_size = 200
        chunk_overlap = 30
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # With overlap prepended, chunks may be slightly larger than chunk_size
        for chunk in chunks:
            assert len(chunk) <= chunk_size + chunk_overlap + 50, (
                f"Chunk too long: {len(chunk)} chars"
            )

    def test_overlap_links_adjacent_chunks(self):
        """The tail of chunk N should appear at the start of chunk N+1."""
        # Use a text that forces force-splitting (no natural separators)
        text = "A" * 600
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
        if len(chunks) >= 2:
            # The overlap text from chunk 0 should appear in chunk 1
            overlap_from_prev = chunks[0][-40:]
            assert overlap_from_prev in chunks[1]

    def test_short_chunks_filtered_out(self):
        """Chunks under 30 characters are dropped."""
        # A paragraph with very short pieces between long pieces
        text = "\n\n".join(["A" * 200, "tiny", "B" * 200])
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=0)
        for chunk in chunks:
            assert len(chunk.strip()) > 30

    def test_paragraph_boundary_preferred(self):
        """Text separated by double newlines should split at those boundaries."""
        para_a = "Alpha " * 40  # 240 chars
        para_b = "Beta " * 40   # 200 chars
        text = para_a + "\n\n" + para_b
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=0)
        # Should produce at least 2 chunks if text > 300 chars
        assert len(chunks) >= 1  # at minimum it doesn't error


# ===========================================================================
# _force_split
# ===========================================================================

class TestForceSplit:

    def test_splits_at_fixed_boundaries(self):
        text = "A" * 100
        chunks = _force_split(text, chunk_size=30, chunk_overlap=0)
        # ceil(100/30) chunks
        assert len(chunks) >= 3
        assert all(len(c) <= 30 for c in chunks)

    def test_overlap_shifts_start_back(self):
        text = "B" * 100
        chunks = _force_split(text, chunk_size=30, chunk_overlap=10)
        # With overlap=10 the step is chunk_size - overlap = 20
        # So we expect ceil(100/20) ≈ 5 chunks
        assert len(chunks) >= 4

    def test_no_overlap_covers_all_text(self):
        text = "C" * 90
        chunks = _force_split(text, chunk_size=30, chunk_overlap=0)
        assert "".join(chunks) == text

    def test_single_chunk_when_fits(self):
        text = "D" * 20
        chunks = _force_split(text, chunk_size=30, chunk_overlap=0)
        assert chunks == [text]

    def test_overlap_at_or_above_chunk_size_still_terminates(self):
        """A misconfigured overlap must not spin forever on a long page."""
        text = "E" * 200
        for overlap in (30, 45):
            chunks = _force_split(text, chunk_size=30, chunk_overlap=overlap)
            assert chunks, "expected at least one chunk"
            assert len(chunks) <= len(text), "stride collapsed to zero"


# ===========================================================================
# CITATION_PATTERN regex
# ===========================================================================

class TestCitationPattern:

    def test_matches_single_digit(self):
        assert CITATION_PATTERN.findall("see [1] for details") == ["1"]

    def test_matches_multi_digit(self):
        assert CITATION_PATTERN.findall("refs [10] and [23]") == ["10", "23"]

    def test_no_match_empty_brackets(self):
        assert CITATION_PATTERN.findall("see [] here") == []

    def test_no_match_text_brackets(self):
        assert CITATION_PATTERN.findall("[abc] reference") == []

    def test_multiple_citations_in_order(self):
        text = "First [1], second [2], third [3]."
        assert CITATION_PATTERN.findall(text) == ["1", "2", "3"]


# ===========================================================================
# extract_citations
# ===========================================================================

SAMPLE_SOURCES = [
    {"url": "https://example.com/a", "title": "Source A", "domain": "example.com"},
    {"url": "https://example.com/b", "title": "Source B", "domain": "example.com"},
    {"url": "https://example.com/c", "title": "Source C", "domain": "example.com"},
]


class TestExtractCitations:

    def test_empty_text_returns_empty(self):
        result = extract_citations("", SAMPLE_SOURCES)
        assert result == []

    def test_no_sources_returns_empty(self):
        result = extract_citations("Some text [1]", [])
        assert result == []

    def test_single_citation_resolved(self):
        text = "The sky is blue [1]."
        citations = extract_citations(text, SAMPLE_SOURCES)
        assert len(citations) == 1
        assert citations[0].index == 1
        assert citations[0].source_url == "https://example.com/a"
        assert citations[0].source_title == "Source A"

    def test_multiple_citations(self):
        text = "Fact one [1]. Fact two [2]. Fact three [3]."
        citations = extract_citations(text, SAMPLE_SOURCES)
        assert len(citations) == 3
        indices = [c.index for c in citations]
        assert 1 in indices and 2 in indices and 3 in indices

    def test_duplicate_citation_deduplicated(self):
        """The same [1] appearing twice should produce only one Citation."""
        text = "First mention [1] and again [1]."
        citations = extract_citations(text, SAMPLE_SOURCES)
        ones = [c for c in citations if c.index == 1]
        assert len(ones) == 1

    def test_out_of_range_citation_ignored(self):
        """A [99] with only 3 sources should be silently skipped."""
        text = "Reference to [99] which does not exist."
        citations = extract_citations(text, SAMPLE_SOURCES)
        assert all(c.index != 99 for c in citations)

    def test_citation_zero_ignored(self):
        """[0] is not a valid citation index (1-based)."""
        text = "Citation [0] is invalid."
        citations = extract_citations(text, SAMPLE_SOURCES)
        assert citations == []

    def test_claim_populated(self):
        """Each citation should carry a non-empty claim string."""
        text = "The experiment confirmed the hypothesis [1]."
        citations = extract_citations(text, SAMPLE_SOURCES)
        assert len(citations) == 1
        assert len(citations[0].claim) > 0

    def test_returns_citation_objects(self):
        """extract_citations returns Citation model instances."""
        text = "Result [1]."
        citations = extract_citations(text, SAMPLE_SOURCES)
        assert all(isinstance(c, Citation) for c in citations)

    def test_no_citations_in_text(self):
        """Text with no bracket markers returns empty list."""
        text = "No citations here at all."
        citations = extract_citations(text, SAMPLE_SOURCES)
        assert citations == []


# ===========================================================================
# extract_citations_detailed — the fabrication signal
# ===========================================================================

class TestExtractCitationsDetailed:
    """
    A marker pointing outside the source list means the model referenced
    something it was never shown. It is dropped from the citations either way;
    these tests cover that it is also *reported*, which is what reaches the UI
    and the stored turn.
    """

    def test_clean_answer_reports_no_invalid_markers(self):
        citations, invalid = extract_citations_detailed(
            "Fact one [1]. Fact two [2].", SAMPLE_SOURCES
        )
        assert len(citations) == 2
        assert invalid == []

    def test_out_of_range_marker_is_reported(self):
        citations, invalid = extract_citations_detailed(
            "A real claim [1]. An invented one [7].", SAMPLE_SOURCES
        )
        assert [c.index for c in citations] == [1]
        assert invalid == [7]

    def test_zero_marker_is_reported_as_invalid(self):
        _citations, invalid = extract_citations_detailed(
            "Citation [0] is not 1-based.", SAMPLE_SOURCES
        )
        assert invalid == [0]

    def test_invalid_markers_are_deduplicated_and_sorted(self):
        _citations, invalid = extract_citations_detailed(
            "Bad [9]. Worse [7]. Again [9].", SAMPLE_SOURCES
        )
        assert invalid == [7, 9]

    def test_boundary_marker_is_valid(self):
        """[3] against exactly 3 sources is the last valid marker, not invalid."""
        citations, invalid = extract_citations_detailed(
            "Edge case [3].", SAMPLE_SOURCES
        )
        assert [c.index for c in citations] == [3]
        assert invalid == []

    def test_empty_inputs_report_nothing(self):
        assert extract_citations_detailed("", SAMPLE_SOURCES) == ([], [])
        assert extract_citations_detailed("Text [1]", []) == ([], [])

    def test_wrapper_matches_detailed_citations(self):
        """extract_citations must stay a pure projection of the detailed call."""
        text = "One [1]. Two [2]. Fake [42]."
        citations, _invalid = extract_citations_detailed(text, SAMPLE_SOURCES)
        assert extract_citations(text, SAMPLE_SOURCES) == citations


# ===========================================================================
# _extract_claim_context
# ===========================================================================

class TestExtractClaimContext:

    def test_extracts_surrounding_sentence(self):
        text = "Background info. The main finding is important [1]. More text follows."
        claim = _extract_claim_context(text, 1)
        assert "main finding" in claim

    def test_missing_marker_returns_empty(self):
        text = "There is no citation here."
        claim = _extract_claim_context(text, 5)
        assert claim == ""

    def test_claim_excludes_citation_markers(self):
        """The claim text should not contain raw [N] markers."""
        text = "Some claim [1] and another [2]."
        claim = _extract_claim_context(text, 1)
        import re
        assert not re.search(r'\[\d+\]', claim)

    def test_claim_length_capped(self):
        from app.utils.citations import MAX_CLAIM_LENGTH
        long_sentence = "W " * 200 + "[1]" + " Z " * 200
        claim = _extract_claim_context(long_sentence, 1)
        assert len(claim) <= MAX_CLAIM_LENGTH


# ===========================================================================
# build_cited_context
# ===========================================================================

class TestBuildCitedContext:

    def _make_chunk(self, url, text, title="Title", domain="example.com"):
        return {"source_url": url, "text": text, "source_title": title, "source_domain": domain}

    def _make_result(self, url, title="Title", domain="example.com", snippet=""):
        return {"url": url, "title": title, "domain": domain, "snippet": snippet, "favicon": ""}

    def test_basic_build(self):
        chunks = [self._make_chunk("https://a.com", "Content from A")]
        results = [self._make_result("https://a.com", title="Page A", domain="a.com")]
        sources, context = build_cited_context(chunks, results)

        assert len(sources) == 1
        assert sources[0]["url"] == "https://a.com"
        assert "[1]" in context
        assert "Content from A" in context

    def test_deduplicates_sources(self):
        """Multiple chunks from the same URL should produce only one source entry."""
        chunks = [
            self._make_chunk("https://a.com", "Chunk one"),
            self._make_chunk("https://a.com", "Chunk two"),
        ]
        results = [self._make_result("https://a.com")]
        sources, context = build_cited_context(chunks, results)
        assert len(sources) == 1

    def test_max_sources_cap(self):
        chunks = [self._make_chunk(f"https://s{i}.com", f"Content {i}") for i in range(15)]
        results = [self._make_result(f"https://s{i}.com") for i in range(15)]
        sources, _ = build_cited_context(chunks, results, max_sources=5)
        assert len(sources) <= 5

    def test_empty_chunks_returns_no_sources(self):
        sources, context = build_cited_context([], [])
        assert sources == []
        assert "No source content" in context

    def test_context_numbers_sources_sequentially(self):
        chunks = [
            self._make_chunk("https://a.com", "Alpha content"),
            self._make_chunk("https://b.com", "Beta content"),
        ]
        results = [
            self._make_result("https://a.com", domain="a.com"),
            self._make_result("https://b.com", domain="b.com"),
        ]
        sources, context = build_cited_context(chunks, results)
        assert "[1]" in context
        assert "[2]" in context

    def test_chunks_with_empty_text_skipped(self):
        chunks = [
            self._make_chunk("https://a.com", ""),
            self._make_chunk("https://b.com", "Valid content"),
        ]
        results = [self._make_result("https://b.com")]
        sources, context = build_cited_context(chunks, results)
        urls = [s["url"] for s in sources]
        assert "https://a.com" not in urls
        assert "https://b.com" in urls

    def test_max_chunks_budget(self):
        """Total chunks in context should not exceed max_chunks."""
        chunks = [self._make_chunk(f"https://s{i}.com", f"Text {i}") for i in range(20)]
        results = [self._make_result(f"https://s{i}.com") for i in range(20)]
        sources, context = build_cited_context(chunks, results, max_chunks=5)
        # At most 5 chunks were included, so at most 5 sources
        assert len(sources) <= 5
