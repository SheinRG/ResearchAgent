"""
Turn a finished research run into labelled-judge *candidates*.

The citation judge needs pairs of (one answer sentence, the evidence behind the
source it cited). Nothing in the product persists the second half: the database
keeps a `SearchResult` snippet, the trace keeps a 180-char preview, and the full
chunk text the model actually read exists only in graph state while the run is
in flight. So it gets captured here, at the one moment it is available.

**These are candidates, not labels.** A cited pair is not automatically
"supported" — the whole premise of the judge is that models cite things their
source does not say — and a swapped pair is not automatically "unsupported",
because a second source from the same run often supports the sentence too.
Auto-labelling either would bake the assumption being tested into the data.
Every record therefore carries ``label: null``, and the pair type is a hint to
whoever (or whatever) labels it, not an answer.

Swapped pairs exist because a set of only-cited pairs is overwhelmingly
supported, and a judge measured on those alone has no measurable precision.
They are free — no extra search credits, no extra tokens — and they are exactly
the case a lexical overlap check gets wrong: same run, same topic, shared
vocabulary, different evidence.

Deterministic by construction: the swap for a given sentence is chosen from a
seed derived from the query id and the sentence, so re-running capture over the
same answers reproduces the same dataset rather than a similar one.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable, Optional

from app.utils.citations import select_evidence

from evals.scorers import claim_units

__all__ = ["capture_pairs", "write_pairs", "PairSink"]

_CITATION = re.compile(r"\[(\d+)\]")

# Evidence shorter than this is a scrap — a nav fragment or a cookie banner that
# survived extraction. Judging a sentence against it teaches nothing except that
# the chunker occasionally emits noise.
_MIN_EVIDENCE_CHARS = 80


def _seed_for(query_id: str, sentence: str) -> int:
    """Stable per-sentence seed, so a re-capture reproduces the same swaps."""
    digest = hashlib.sha256(f"{query_id}\x00{sentence}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _pair_id(query_id: str, sentence: str, source_url: str, pair_type: str) -> str:
    raw = f"{query_id}\x00{sentence}\x00{source_url}\x00{pair_type}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _record(
    *,
    query_id: str,
    query: str,
    sentence: str,
    cited_indices: list[int],
    index: int,
    source: dict,
    chunks: list[str],
    pair_type: str,
) -> dict:
    return {
        "pair_id": _pair_id(query_id, sentence, source.get("url", ""), pair_type),
        "query_id": query_id,
        "query": query,
        "sentence": sentence,
        # What the sentence actually cited, so a labeller can see whether the
        # claim was meant to rest on this source or on a different one.
        "cited_indices": cited_indices,
        "citation_index": index,
        "pair_type": pair_type,
        "source_url": source.get("url", ""),
        "source_title": source.get("title", ""),
        "source_domain": source.get("domain", ""),
        "evidence": "\n".join(chunks),
        "evidence_chunks": len(chunks),
        # Filled in by labelling — never by capture. See the module docstring.
        "label": None,
    }


def capture_pairs(
    *,
    query_id: str,
    query: str,
    answer: str,
    ranked_chunks: list[dict],
    cited_sources: list[dict],
    max_sources: int = 8,
    max_chunks: int = 12,
    swaps_per_sentence: int = 1,
) -> list[dict]:
    """
    Build judge candidates from one finished run.

    Args:
        query_id: Stable id from the eval set; keys the dataset.
        query: The question, for context while labelling.
        answer: The generated markdown answer, with its [n] markers.
        ranked_chunks: Final-state ranked chunks — the only place full chunk
            text exists.
        cited_sources: The canonical ordered source list; ``[i]`` in the answer
            is ``cited_sources[i - 1]``.
        max_sources / max_chunks: Must match the settings the run used, or the
            recovered evidence is not what the model read.
        swaps_per_sentence: How many same-run non-cited sources to pair each
            sentence with, as negative candidates.

    Returns:
        One record per (sentence, source) pair, both cited and swapped.
    """
    if not answer or not cited_sources:
        return []

    order, chunks_by_url = select_evidence(ranked_chunks, max_sources, max_chunks)
    if not order:
        return []

    # select_evidence and cited_sources are built from the same inputs by the
    # same rules, but a run that fell back to raw search results (the
    # synthesizer's no-usable-chunks path) has sources with no chunks behind
    # them. Those cannot be judged, so pairing is capped at what both agree on.
    usable = min(len(order), len(cited_sources))

    records: list[dict] = []

    for sentence in claim_units(answer):
        cited = sorted({
            int(m) for m in _CITATION.findall(sentence)
            if 1 <= int(m) <= usable
        })
        if not cited:
            continue  # uncited sentences are a coverage problem, not a support one

        for index in cited:
            chunks = chunks_by_url.get(order[index - 1], [])
            if len("\n".join(chunks)) < _MIN_EVIDENCE_CHARS:
                continue
            records.append(_record(
                query_id=query_id,
                query=query,
                sentence=sentence,
                cited_indices=cited,
                index=index,
                source=cited_sources[index - 1],
                chunks=chunks,
                pair_type="cited",
            ))

        # Negative candidates: same run, different source.
        alternatives = [i for i in range(1, usable + 1) if i not in cited]
        if not alternatives or swaps_per_sentence < 1:
            continue

        rng = random.Random(_seed_for(query_id, sentence))
        picks = rng.sample(alternatives, min(swaps_per_sentence, len(alternatives)))
        for index in picks:
            chunks = chunks_by_url.get(order[index - 1], [])
            if len("\n".join(chunks)) < _MIN_EVIDENCE_CHARS:
                continue
            records.append(_record(
                query_id=query_id,
                query=query,
                sentence=sentence,
                cited_indices=cited,
                index=index,
                source=cited_sources[index - 1],
                chunks=chunks,
                pair_type="swapped",
            ))

    return records


def write_pairs(records: Iterable[dict], path: Path) -> int:
    """
    Append records to a JSONL file, skipping pair_ids already in it.

    Append rather than overwrite because capture accumulates across runs — the
    in-domain set is built up over many eval runs, not produced by one. The
    pair_id check makes re-running the same queries idempotent instead of
    duplicating every sentence.

    Returns:
        How many records were actually new.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    seen.add(json.loads(line)["pair_id"])
                except (json.JSONDecodeError, KeyError):
                    continue  # a truncated tail should not stop the append

    written = 0
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            if record["pair_id"] in seen:
                continue
            seen.add(record["pair_id"])
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


class PairSink:
    """
    Collects candidates across a concurrent run, then writes them once.

    The eval set runs several queries at a time; appending to the file from each
    would interleave partial lines. Collecting in memory is safe here — a full
    run is a few hundred records — and keeps the write a single ordered step.
    """

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self.records: list[dict] = []

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def add(self, records: list[dict]) -> None:
        if self.enabled:
            self.records.extend(records)

    def flush(self) -> int:
        if not self.enabled or not self.records:
            return 0
        return write_pairs(self.records, self.path)
