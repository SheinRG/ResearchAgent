"""
Safety rails for semantic cache hits.

Embeddings encode *topic*, not *direction*. "best vector database" and "worst
vector database" are about the same subject, so their vectors sit almost on top
of one another — while the correct answers are opposites. Cosine similarity
alone will happily serve one for the other, with citations, which for a research
tool is worse than being slow.

So a high score is necessary but not sufficient. Every candidate must also clear
these checks, all of which are cheap string work:

    numbers    a digit in one question that is absent from the other means the
               questions are about different things — versions, years, prices,
               limits. "Python 3.11 features" vs "3.12" barely moves a vector.
    polarity   negations and superlatives flip the meaning while leaving the
               topic untouched.
    order      "is X better than Y" and "is Y better than X" are near-identical
               vectors asking opposite questions. Both the "X vs Y" and the
               "X <comparative> than Y" phrasings are recognised.
    overlap    a floor on shared meaningful words, as a blunt backstop against
               two unrelated questions that happen to score highly.

Every function here is pure and returns a reason string, so a rejection can be
logged and counted rather than disappearing silently.
"""

from __future__ import annotations

import re
from typing import Optional

# Digits, decimals, and versions: 3.11, 2026, 200, v2
_NUMBER = re.compile(r"\d+(?:\.\d+)*")

# Words that invert meaning without moving the topic. Superlatives are included
# because "best"/"worst" is the canonical failure this module exists for.
_POLARITY = frozenset({
    "not", "no", "never", "without", "except", "excluding", "avoid",
    "best", "worst", "most", "least", "cheapest", "priciest",
    "fastest", "slowest", "largest", "smallest", "highest", "lowest",
    "more", "less", "fewer", "better", "worse", "faster", "slower",
    "pros", "cons", "advantages", "disadvantages", "benefits", "drawbacks",
    "should", "shouldn't", "can", "can't", "cannot", "isn't", "aren't",
    "before", "after", "under", "over", "above", "below",
})

# Ignored when measuring overlap: present in almost every question, so they
# inflate similarity between questions that share nothing meaningful.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "i", "me", "my", "we", "our", "you", "your", "it", "its", "they", "them",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "and", "or", "but", "if", "then", "than", "as", "of", "at", "by", "for",
    "with", "about", "into", "to", "from", "in", "on", "out", "up", "down",
    "so", "that", "this", "these", "those", "there", "here",
    "get", "got", "make", "made", "use", "used", "using", "need", "want",
    "please", "tell", "explain", "give", "show", "some", "any", "all",
})

_WORD = re.compile(r"[a-z0-9']+")

# Comparatives that need two sides: "is X faster than Y". Swapping the sides
# barely moves the vector but inverts the answer, exactly like "X vs Y", so the
# order has to be checked the same way. Kept as an explicit list rather than an
# "-er" suffix rule, which would read "other than" as a comparison.
_COMPARATIVES = frozenset({
    "better", "worse", "faster", "slower", "cheaper", "pricier", "safer",
    "bigger", "smaller", "larger", "stronger", "weaker", "easier", "harder",
    "simpler", "lighter", "heavier", "newer", "older", "quicker", "cleaner",
    "more", "less", "fewer",
})

# Degree modifiers sit between a comparative and the thing being compared
# ("much faster than"), so they are skipped when walking out to the two sides.
_MODIFIERS = frozenset({
    "much", "far", "way", "lot", "bit", "even", "still", "really", "quite",
    "slightly", "significantly", "considerably", "noticeably",
})


def numbers_in(text: str) -> set[str]:
    """Every number in the text, normalized so 3.10 and 3.1 stay distinct."""
    return set(_NUMBER.findall((text or "").lower()))


def words_in(text: str) -> list[str]:
    """Lowercased word tokens, in order."""
    return _WORD.findall((text or "").lower())


def content_words(text: str) -> set[str]:
    """Meaningful words only — stopwords and pure numbers removed."""
    return {
        w for w in words_in(text)
        if w not in _STOPWORDS and not w.isdigit() and len(w) > 1
    }


def polarity_words(text: str) -> set[str]:
    """Meaning-flipping words present in the text."""
    return {w for w in words_in(text) if w in _POLARITY}


def token_overlap(a: str, b: str) -> float:
    """
    Jaccard overlap of the two questions' meaningful words, 0.0 to 1.0.

    Two questions with no shared content words score 0 no matter what the
    embedding thinks of them.
    """
    words_a, words_b = content_words(a), content_words(b)
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _side_word(words: list[str], index: int, step: int) -> Optional[str]:
    """
    The nearest meaningful word from ``index``, walking in direction ``step``.

    Stopwords and degree modifiers are skipped so that "is the postgres much
    faster than a mongodb" compares postgres against mongodb rather than "the"
    against "a".
    """
    i = index + step
    while 0 <= i < len(words):
        word = words[i]
        if word not in _STOPWORDS and word not in _MODIFIERS:
            return word
        i += step
    return None


def _comparative_before(words: list[str], than_index: int) -> Optional[int]:
    """
    Index of the comparative that makes a "than" a comparison, if there is one.

    Looks back a short window so that both "better than" and "more secure than"
    are caught, while "other than" and "rather than" are not.
    """
    for i in range(max(0, than_index - 3), than_index):
        if words[i] in _COMPARATIVES:
            return i
    return None


def _comparison_order(text: str) -> Optional[tuple[str, str]]:
    """
    The two sides of a comparison, in the order they were asked.

    Recognises "X vs Y" and "X <comparative> than Y". Returns None when the
    question is not a comparison.
    """
    words = words_in(text)
    for i, word in enumerate(words):
        if word in ("vs", "versus"):
            left, right = _side_word(words, i, -1), _side_word(words, i, 1)
            if left and right:
                return left, right
            continue

        if word == "than":
            marker = _comparative_before(words, i)
            if marker is None:
                continue
            left, right = _side_word(words, marker, -1), _side_word(words, i, 1)
            if left and right:
                return left, right
    return None


def rejection_reason(
    query: str,
    candidate: str,
    min_token_overlap: float = 0.5,
) -> Optional[str]:
    """
    Why ``candidate``'s answer must not be reused for ``query``.

    Returns None when the candidate passes every check — the only case in which
    a cached answer may be served for a differently-worded question.

    Args:
        query: The question being asked now.
        candidate: The question whose answer is stored.
        min_token_overlap: Floor on shared meaningful words.
    """
    if not (query or "").strip() or not (candidate or "").strip():
        return "empty question"

    query_numbers, candidate_numbers = numbers_in(query), numbers_in(candidate)
    if query_numbers != candidate_numbers:
        differing = sorted(query_numbers ^ candidate_numbers)
        return f"different numbers ({', '.join(differing)})"

    query_polarity, candidate_polarity = polarity_words(query), polarity_words(candidate)
    if query_polarity != candidate_polarity:
        differing = sorted(query_polarity ^ candidate_polarity)
        return f"different polarity ({', '.join(differing)})"

    query_order, candidate_order = _comparison_order(query), _comparison_order(candidate)
    if query_order and candidate_order and query_order != candidate_order:
        return "comparison asked in the opposite order"

    overlap = token_overlap(query, candidate)
    if overlap < min_token_overlap:
        return f"low word overlap ({overlap:.2f} < {min_token_overlap:.2f})"

    return None
