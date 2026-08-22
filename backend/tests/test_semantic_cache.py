"""
Unit tests for the semantic cache's vector math, bucketing and TTL policy.

The guards live in test_semantic_guards.py. This file covers the parts that
decide *which* answers are even candidates: normalization and cosine similarity,
the bucket key that keeps exact-match concerns out of the similarity comparison,
and how long an answer is allowed to live.

No Redis and no embedding provider — everything here is arithmetic and hashing.
"""

import math

import pytest

from app.config import get_settings
from app.services.answer_cache import build_bucket_key, ttl_for
from app.services.embeddings import cosine, normalize

MODEL = "openai/gpt-oss-120b"


# ===========================================================================
# Vector math
# ===========================================================================

class TestNormalize:

    def test_produces_a_unit_vector(self):
        result = normalize([3.0, 4.0])
        assert math.sqrt(sum(v * v for v in result)) == pytest.approx(1.0)

    def test_preserves_direction(self):
        result = normalize([3.0, 4.0])
        assert result[0] == pytest.approx(0.6)
        assert result[1] == pytest.approx(0.8)

    def test_zero_vector_survives_instead_of_dividing_by_zero(self):
        assert normalize([0.0, 0.0]) == [0.0, 0.0]

    def test_already_normalized_vector_is_unchanged(self):
        assert normalize([1.0, 0.0]) == pytest.approx([1.0, 0.0])


class TestCosine:
    """
    Vectors are normalized at store time, so cosine is a plain dot product.
    These tests all use unit vectors for that reason.
    """

    def test_identical_vectors_score_one(self):
        v = normalize([0.3, 0.9, 0.1])
        assert cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self):
        assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_similar_vectors_score_near_one(self):
        a = normalize([1.0, 0.9])
        b = normalize([1.0, 0.85])
        assert cosine(a, b) > 0.99

    def test_mismatched_dimensions_score_zero_rather_than_raising(self):
        """
        A vector stored by a different embedding model must not take a request
        down — it is simply not similar to anything.
        """
        assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0

    def test_empty_vectors_score_zero(self):
        assert cosine([], [1.0]) == 0.0
        assert cosine([1.0], []) == 0.0

    def test_realistic_dimension_count(self):
        """384 dims is the configured size; make sure nothing chokes on it."""
        a = normalize([0.05] * 384)
        assert cosine(a, a) == pytest.approx(1.0)


# ===========================================================================
# Bucketing
# ===========================================================================

class TestBuildBucketKey:
    """
    Everything that must match EXACTLY belongs in the bucket, so cosine
    similarity is only ever asked to judge the question itself.
    """

    def test_same_inputs_give_the_same_bucket(self):
        assert build_bucket_key("", MODEL) == build_bucket_key("", MODEL)

    def test_preferred_name_changes_the_bucket(self):
        """
        Answers may address the user by name. Sharing a bucket would let one
        user be greeted by another user's name.
        """
        assert build_bucket_key("Raghav", MODEL) != build_bucket_key("", MODEL)
        assert build_bucket_key("Raghav", MODEL) != build_bucket_key("Priya", MODEL)

    def test_model_change_changes_the_bucket(self):
        """A model swap must not resurrect answers written by the old one."""
        assert build_bucket_key("", MODEL) != build_bucket_key("", "openai/gpt-oss-20b")

    def test_name_whitespace_is_ignored(self):
        assert build_bucket_key("  Raghav  ", MODEL) == build_bucket_key("Raghav", MODEL)

    def test_key_is_namespaced_and_bounded(self):
        key = build_bucket_key("", MODEL)
        assert key.startswith("research:answer_index:")
        assert len(key) < 80


# ===========================================================================
# TTL policy
# ===========================================================================

class TestTtlFor:
    """
    Before triage emitted time_sensitive, every answer expired on the short TTL
    because nothing could tell a stock price from a definition.
    """

    def test_time_sensitive_answers_get_the_short_ttl(self):
        assert ttl_for(True) == get_settings().answer_cache_ttl

    def test_evergreen_answers_live_much_longer(self):
        settings = get_settings()
        assert ttl_for(False) == settings.answer_cache_evergreen_ttl
        assert ttl_for(False) > ttl_for(True)

    def test_the_gap_is_worth_having(self):
        """If the two TTLs were close, the flag would not be earning its call."""
        assert ttl_for(False) >= ttl_for(True) * 4


# ===========================================================================
# Configured defaults
# ===========================================================================

class TestDefaults:

    def test_semantic_cache_is_off_by_default(self):
        """
        Unlike the other optional integrations, this one can serve an answer to
        a question the user did not ask. It gets switched on deliberately.
        """
        assert get_settings().semantic_cache_enabled is False

    def test_threshold_is_high(self):
        assert get_settings().semantic_similarity_threshold >= 0.85

    def test_semantic_hits_expire_faster_than_exact_ones(self):
        """
        A near-match to a time-sensitive answer compounds two kinds of drift —
        the answer's age and the difference in intent — so it must be fresher.
        """
        settings = get_settings()
        assert settings.semantic_time_sensitive_max_age < settings.answer_cache_ttl
