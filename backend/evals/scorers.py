"""
The inspector's checklist — pure scoring functions over a finished answer.

Every function here takes text and data and returns a number. No network, no
model, no clock. That is the point: the checklist can be unit-tested on fixed
inputs and run on every pull request, while the expensive part (actually running
the pipeline against live APIs) happens on a schedule.

What these measure is FAITHFULNESS and STRUCTURAL INTEGRITY, not truth. An
answer can be perfectly faithful to a source that is wrong, and nothing here
would notice. Judging truth needs ground-truth labels this project does not
have; judging whether the agent followed its own process needs only the data the
pipeline already produces.

Several scorers are deliberately heuristic — sentence splitting and "is this a
factual claim" have no exact answer. They are calibrated to be stable rather
than precise: the number matters as a trend against its own history, not as an
absolute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.agents.messages import FALLBACK_MESSAGES
from app.utils.citations import extract_citations_detailed

# A GFM table separator row: | --- | :---: | ---: |
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_BULLET = re.compile(r"^\s*[-*+]\s+\S")
_ORDERED = re.compile(r"^\s*\d+[.)]\s+\S")
_FENCE = re.compile(r"^\s*```")
_CITATION = re.compile(r"\[(\d+)\]")

# Sentence boundary: . ! or ? followed by whitespace and a capital/digit/quote.
# Crude on purpose — a full sentence tokenizer is a dependency and a decision
# this harness does not need to make.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[])")

# Fragments shorter than this are treated as scaffolding ("Key points:",
# "In summary") rather than claims that need a source.
_MIN_CLAIM_CHARS = 45

VALID_FORMATS = ("table", "list", "steps", "prose")


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

def strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks — code is not a claim and is never cited."""
    out: list[str] = []
    in_fence = False
    for line in (text or "").splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return "\n".join(out)


def claim_units(text: str) -> list[str]:
    """
    Split an answer into units that ought to carry a citation.

    A "unit" is one prose sentence, one list item, or one table data row —
    whatever the smallest standalone assertion is for that piece of formatting.
    Headings, table header/separator rows, and short scaffolding lines are
    excluded: they announce content rather than assert anything.
    """
    units: list[str] = []
    lines = strip_code_blocks(text).splitlines()

    prose_buffer: list[str] = []
    table_header_seen = False

    def flush_prose() -> None:
        if not prose_buffer:
            return
        paragraph = " ".join(prose_buffer).strip()
        prose_buffer.clear()
        if not paragraph:
            return
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if len(sentence) >= _MIN_CLAIM_CHARS:
                units.append(sentence)

    for line in lines:
        stripped = line.strip()

        if not stripped:
            flush_prose()
            table_header_seen = False
            continue

        if _HEADING.match(line):
            flush_prose()
            continue

        if _TABLE_SEPARATOR.match(line):
            # The separator confirms the row above it was a header, not data.
            table_header_seen = True
            continue

        if _TABLE_ROW.match(line):
            flush_prose()
            if not table_header_seen:
                # Header row (or a table with no separator yet) — not a claim.
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if any(len(c) >= 2 for c in cells):
                units.append(stripped)
            continue

        if _BULLET.match(line) or _ORDERED.match(line):
            flush_prose()
            item = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", stripped)
            if len(item) >= _MIN_CLAIM_CHARS:
                units.append(item)
            continue

        prose_buffer.append(stripped)

    flush_prose()
    return units


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def invalid_citations(answer: str, sources: list) -> list[int]:
    """
    Citation markers pointing at a source the model was never given.

    Delegates to the same function the pipeline uses, so the eval cannot drift
    from what production actually counts.
    """
    _citations, invalid = extract_citations_detailed(answer or "", sources or [])
    return invalid


def is_fallback(answer: str) -> bool:
    """Whether the pipeline returned a canned apology instead of an answer."""
    text = (answer or "").strip()
    if not text:
        return False
    return any(message in text for message in FALLBACK_MESSAGES)


def citation_coverage(answer: str) -> tuple[int, int]:
    """
    How much of the answer carries a citation.

    Returns (cited_units, total_units). A unit is one sentence, list item or
    table row (see :func:`claim_units`). 100% is neither expected nor desirable
    — an opening summary sentence reasonably goes uncited — so this is a trend
    metric, not a gate.
    """
    units = claim_units(answer)
    if not units:
        return 0, 0
    cited = sum(1 for unit in units if _CITATION.search(unit))
    return cited, len(units)


def source_utilization(answer: str, sources: list) -> tuple[int, int]:
    """
    How many of the sources handed to the model were actually cited.

    Returns (used, provided). Persistently low utilization is a cost signal as
    much as a quality one: it means retrieval and reranking are paying for
    sources the synthesizer ignores.
    """
    provided = len(sources or [])
    if not provided:
        return 0, 0
    citations, _invalid = extract_citations_detailed(answer or "", sources)
    return len({c.index for c in citations}), provided


def has_table(answer: str, min_columns: int = 2) -> bool:
    """Whether the answer contains a well-formed GFM table."""
    lines = strip_code_blocks(answer).splitlines()
    for i, line in enumerate(lines[:-1]):
        if _TABLE_ROW.match(line) and _TABLE_SEPARATOR.match(lines[i + 1]):
            columns = len([c for c in line.strip().strip("|").split("|")])
            if columns >= min_columns:
                return True
    return False


def table_columns(answer: str) -> int:
    """Column count of the first well-formed table, or 0 if there is none."""
    lines = strip_code_blocks(answer).splitlines()
    for i, line in enumerate(lines[:-1]):
        if _TABLE_ROW.match(line) and _TABLE_SEPARATOR.match(lines[i + 1]):
            return len(line.strip().strip("|").split("|"))
    return 0


def count_list_items(answer: str) -> int:
    """Bullet or numbered items, excluding table rows."""
    count = 0
    for line in strip_code_blocks(answer).splitlines():
        if _TABLE_ROW.match(line):
            continue
        if _BULLET.match(line) or _ORDERED.match(line):
            count += 1
    return count


def count_ordered_items(answer: str) -> int:
    """Numbered items only — what distinguishes 'steps' from a plain list."""
    return sum(
        1
        for line in strip_code_blocks(answer).splitlines()
        if _ORDERED.match(line) and not _TABLE_ROW.match(line)
    )


def format_compliance(answer: str, declared_format: str) -> bool:
    """
    Whether the answer delivered the shape the triage stage promised.

    This is a contract check between two stages, not an opinion about which
    format is nicer: triage tells the synthesizer what to produce, and this
    asks whether it complied.

    "prose" always passes. It is the absence of a structural requirement, so
    there is nothing to verify without judging the writing.
    """
    fmt = (declared_format or "").strip().lower()
    if fmt == "table":
        return has_table(answer)
    if fmt == "steps":
        return count_ordered_items(answer) >= 2
    if fmt == "list":
        return count_list_items(answer) >= 2
    return True


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

@dataclass
class QueryScore:
    """Every measurement for a single evaluated query."""

    id: str
    query: str
    category: str = ""

    # Hard invariants
    answered: bool = False
    fallback: bool = False
    invalid_citation_markers: list[int] = field(default_factory=list)

    # Trend metrics
    cited_units: int = 0
    total_units: int = 0
    sources_used: int = 0
    sources_provided: int = 0
    format_ok: bool = True

    # Observed pipeline behaviour
    mode: str = ""
    declared_format: str = ""
    expected_format: str = ""
    triage_format_match: bool = True
    sub_queries: int = 0
    ranked_chunks: int = 0
    top_rerank_score: float = 0.0
    missing_terms: list[str] = field(default_factory=list)

    # Cost / latency
    latency_ms: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    error: str = ""

    @property
    def coverage(self) -> float:
        return self.cited_units / self.total_units if self.total_units else 0.0

    @property
    def utilization(self) -> float:
        return self.sources_used / self.sources_provided if self.sources_provided else 0.0

    def failures(self, allow_fallback: bool = False) -> list[str]:
        """
        Hard invariant violations — the things that fail the build.

        Everything else is a trend compared against the baseline. These are
        absolutes: an invented citation or a crashed run is never acceptable,
        whatever last week's numbers were.
        """
        problems: list[str] = []
        if self.error:
            problems.append(f"run failed: {self.error}")
            return problems
        if not self.answered:
            problems.append("empty answer")
        if self.invalid_citation_markers:
            problems.append(
                f"fabricated citation markers: {self.invalid_citation_markers}"
            )
        if self.fallback and not allow_fallback:
            problems.append("returned a fallback message instead of an answer")
        if self.mode == "research" and not self.fallback and self.sources_provided == 0:
            problems.append("research answer with no sources")
        if self.missing_terms:
            problems.append(f"missing required terms: {self.missing_terms}")
        return problems

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "category": self.category,
            "answered": self.answered,
            "fallback": self.fallback,
            "invalid_citation_markers": self.invalid_citation_markers,
            "coverage": round(self.coverage, 4),
            "cited_units": self.cited_units,
            "total_units": self.total_units,
            "utilization": round(self.utilization, 4),
            "sources_used": self.sources_used,
            "sources_provided": self.sources_provided,
            "format_ok": self.format_ok,
            "mode": self.mode,
            "expected_format": self.expected_format,
            "declared_format": self.declared_format,
            "triage_format_match": self.triage_format_match,
            "sub_queries": self.sub_queries,
            "ranked_chunks": self.ranked_chunks,
            "top_rerank_score": round(self.top_rerank_score, 4),
            "missing_terms": self.missing_terms,
            "latency_ms": self.latency_ms,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "error": self.error,
        }


def score_run(
    *,
    spec: dict,
    final_state: dict,
    latency_ms: int = 0,
    usage: Optional[dict] = None,
    error: str = "",
) -> QueryScore:
    """
    Apply the whole checklist to one completed run.

    ``spec`` is the query definition; ``final_state`` is what the graph
    accumulated. Kept separate from the runner so it can be exercised on
    recorded states with no pipeline in sight.
    """
    score = QueryScore(
        id=str(spec.get("id", "")),
        query=str(spec.get("query", "")),
        category=str(spec.get("category", "")),
        expected_format=str(spec.get("expected_format", "")),
        error=error,
    )
    if error:
        return score

    answer = final_state.get("draft_answer") or ""
    sources = final_state.get("all_sources") or []
    ranked = final_state.get("ranked_chunks") or []

    score.answered = bool(answer.strip())
    score.fallback = is_fallback(answer)
    score.invalid_citation_markers = invalid_citations(answer, sources)

    score.cited_units, score.total_units = citation_coverage(answer)
    score.sources_used, score.sources_provided = source_utilization(answer, sources)

    score.mode = final_state.get("mode", "research")
    score.declared_format = (final_state.get("answer_format") or {}).get("type", "")
    score.sub_queries = len(final_state.get("sub_queries") or [])
    score.ranked_chunks = len(ranked)
    score.top_rerank_score = float(ranked[0].get("score", 0.0)) if ranked else 0.0

    # Chat replies have no format contract and no sources to cite.
    if score.mode == "chat":
        score.format_ok = True
        score.triage_format_match = score.expected_format in ("", "chat", "prose")
    else:
        score.format_ok = format_compliance(answer, score.declared_format)
        score.triage_format_match = (
            not score.expected_format or score.declared_format == score.expected_format
        )

    required = spec.get("must_mention") or []
    lowered = answer.lower()
    score.missing_terms = [t for t in required if str(t).lower() not in lowered]

    score.latency_ms = latency_ms
    score.total_tokens = (usage or {}).get("total_tokens", 0)
    score.cost_usd = (usage or {}).get("cost_usd", 0.0)
    return score
