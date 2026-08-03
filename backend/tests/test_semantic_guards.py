"""
Unit tests for the semantic cache's safety rails.

These matter more than the similarity math. Cosine similarity is arithmetic —
it either works or it obviously doesn't. The guards are the part standing
between a fast answer and a confidently wrong one, and every case below is a
pair of questions that a good embedding model rates as nearly identical while
the correct answers differ completely.

If a guard is loosened, this file is where it shows up as a user-visible bug
rather than as a number moving on a dashboard.
"""

import pytest

from app.utils.semantic_guards import (
    content_words,
    numbers_in,
    polarity_words,
    rejection_reason,
    token_overlap,
)


# Mirrors semantic_min_token_overlap so these tests exercise the shipped value.
DEFAULT_OVERLAP = 0.3


def accepts(a, b, min_overlap=DEFAULT_OVERLAP):
    return rejection_reason(a, b, min_overlap) is None


def rejects(a, b, min_overlap=DEFAULT_OVERLAP):
    return rejection_reason(a, b, min_overlap) is not None


# ===========================================================================
# The dangerous pairs
# ===========================================================================

class TestOppositesAreRejected:
    """
    Superlatives and negations leave the topic untouched while inverting the
    answer. This is the single failure mode the guards exist for.
    """

    def test_best_versus_worst(self):
        assert rejects(
            "best vector databases for a RAG application",
            "worst vector databases for a RAG application",
        )

    def test_negation_flips_meaning(self):
        assert rejects(
            "why you should use pgvector for production",
            "why you should not use pgvector for production",
        )

    def test_pros_versus_cons(self):
        assert rejects(
            "advantages of using LangChain in production",
            "disadvantages of using LangChain in production",
        )

    def test_cheapest_versus_priciest(self):
        assert rejects(
            "cheapest hosting for a small side project",
            "priciest hosting for a small side project",
        )

    def test_rejection_reason_names_the_polarity_words(self):
        reason = rejection_reason("best python orm", "worst python orm")
        assert "polarity" in reason
        assert "best" in reason and "worst" in reason


class TestNumbersAreRejected:
    """
    A changed digit barely moves a vector but completely changes the question.
    Versions, years, and price ceilings all live here.
    """

    def test_different_python_versions(self):
        assert rejects(
            "what are the new features in Python 3.11",
            "what are the new features in Python 3.12",
        )

    def test_different_price_ceilings(self):
        assert rejects(
            "best noise cancelling headphones under 200 dollars",
            "best noise cancelling headphones under 500 dollars",
        )

    def test_different_years(self):
        assert rejects(
            "state of EU AI regulation in 2025",
            "state of EU AI regulation in 2026",
        )

    def test_number_present_in_only_one_question(self):
        assert rejects(
            "how do I upgrade to Next.js 16",
            "how do I upgrade to Next.js",
        )

    def test_same_numbers_are_fine(self):
        assert accepts(
            "what changed in Python 3.12",
            "what is new in Python 3.12",
        )

    def test_reason_names_the_differing_numbers(self):
        reason = rejection_reason("Python 3.11 features", "Python 3.12 features")
        assert "numbers" in reason
        assert "3.11" in reason or "3.12" in reason


class TestComparisonOrder:
    """"Is X better than Y" and "is Y better than X" embed almost identically."""

    def test_reversed_comparison_is_rejected(self):
        assert rejects(
            "postgres vs mongodb for analytics workloads",
            "mongodb vs postgres for analytics workloads",
        )

    def test_same_order_is_accepted(self):
        assert accepts(
            "postgres vs mongodb for analytics workloads",
            "postgres vs mongodb for analytics",
        )


class TestTokenOverlap:
    """A blunt backstop for two unrelated questions that happen to score high."""

    def test_unrelated_questions_are_rejected(self):
        assert rejects(
            "how do I configure a Cloudflare tunnel",
            "what causes the northern lights",
        )

    def test_low_overlap_reason_reports_the_score(self):
        reason = rejection_reason(
            "how do I configure a Cloudflare tunnel",
            "what causes the northern lights",
        )
        assert "overlap" in reason

    def test_threshold_is_respected(self):
        a, b = "vector database pricing", "vector database costs"
        assert accepts(a, b, min_overlap=0.3)
        assert rejects(a, b, min_overlap=0.95)

    def test_short_paraphrases_need_the_lower_default(self):
        """
        Why the floor is 0.3 and not higher. Short questions keep very few
        content words after stopword removal, so one swapped synonym costs a
        third of the score even though the questions mean the same thing.
        """
        a, b = "what changed in Python 3.12", "what is new in Python 3.12"
        assert token_overlap(a, b) == pytest.approx(1 / 3, abs=0.01)
        assert accepts(a, b)                      # at the shipped 0.3
        assert rejects(a, b, min_overlap=0.5)     # at the stricter floor


# ===========================================================================
# Genuine paraphrases must still pass
# ===========================================================================

class TestParaphrasesAreAccepted:
    """A guard that rejects everything is as useless as no cache at all."""

    @pytest.mark.parametrize("a,b", [
        (
            "what is retrieval augmented generation",
            "explain retrieval augmented generation",
        ),
        (
            "how does the reranker score passages",
            "how does the reranker score passages exactly",
        ),
        (
            "tips for reducing docker image size",
            "tips reducing docker image size",
        ),
    ])
    def test_paraphrase_passes(self, a, b):
        assert accepts(a, b, min_overlap=0.4)

    def test_identical_questions_pass(self):
        assert accepts("what is pgvector", "what is pgvector")


class TestEdges:

    def test_empty_question_is_rejected(self):
        assert rejects("", "what is pgvector")
        assert rejects("what is pgvector", "   ")

    def test_case_and_spacing_do_not_matter(self):
        assert accepts("What Is  PGVector", "what is pgvector")


# ===========================================================================
# The pieces
# ===========================================================================

class TestHelpers:

    def test_numbers_include_versions_and_decimals(self):
        assert numbers_in("Python 3.11 vs 3.9 in 2026") == {"3.11", "3.9", "2026"}

    def test_numbers_empty_when_none_present(self):
        assert numbers_in("what is pgvector") == set()

    def test_polarity_words_detected(self):
        assert "worst" in polarity_words("the worst database")
        assert polarity_words("what is a database") == set()

    def test_content_words_drop_stopwords(self):
        words = content_words("what is the best way to do this")
        assert "best" in words
        assert "what" not in words and "the" not in words

    def test_content_words_drop_bare_numbers(self):
        assert "2026" not in content_words("regulation in 2026")

    def test_token_overlap_identical_is_one(self):
        assert token_overlap("vector database", "vector database") == 1.0

    def test_token_overlap_disjoint_is_zero(self):
        assert token_overlap("vector database", "northern lights") == 0.0

    def test_token_overlap_empty_is_zero(self):
        assert token_overlap("", "vector database") == 0.0
