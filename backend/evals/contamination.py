"""
Keep the gold set out of the training data.

The gold set is the only artifact licensed to produce a quotable number, and
the pre-registered bar is a comparison against baselines that never saw it.
A training pair that overlaps gold breaks that comparison in the one direction
that flatters the fine-tune, which is exactly the direction this project has
spent its effort refusing to be flattered in.

The subtle part is that *pair-id* uniqueness is not enough. Capture emits many
pairs per query, and pairs from one query share retrieved evidence chunks: the
same 174-word chunk backs a cited pair and a swapped one, and backs several
sentences of the same answer. Training on pair A and testing on pair B is
therefore training on the test set's evidence whenever A and B come from the
same run. Measured on the current capture: of the 111 candidates outside the
gold-200, 97 reuse a gold evidence chunk and 81 reuse a gold sentence
verbatim. Exactly **one** is clean on both.

So the guard rejects on four grounds, cheapest first:

1. ``pair_id`` — the same pair.
2. ``query_id`` — the same pipeline run. The blunt one, and the one that
   actually does the work: same run means same retrieved corpus.
3. Normalised evidence text — the same chunk reached through a different run.
4. Near-duplicate sentence — the same claim, reworded. Paraphrase is what an
   exact hash misses, so this tier uses the repo's existing lexical overlap
   rather than a second notion of similarity.

The manifest carries **hashes only, never text**, for the same reason
``label.py`` writes only ids and labels: the file is committed, and a
committed copy of the gold sentences is a copy of the answer key.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from app.utils.semantic_guards import content_words, token_overlap

# Above this Jaccard overlap of content words, two sentences are treated as the
# same claim reworded. Deliberately loose: a false positive costs one training
# pair out of thousands, a false negative costs the headline number's meaning.
#
# 0.70 rather than a rounder 0.80 because `content_words` does not stem. A real
# paraphrase of a 7-word sentence that changes only `requires` -> `required`
# scores 0.75: the inflected pair lands in neither intersection, costing two
# tokens of union. Short sentences are hit hardest, and answer sentences are
# short (median 30 words, most of them stopwords).
NEAR_DUPLICATE = 0.70

_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Casefold and collapse whitespace, so reformatting cannot defeat a hash."""
    return _WHITESPACE.sub(" ", text or "").strip().casefold()


def fingerprint(text: str) -> str:
    """Stable short digest of normalised text."""
    return hashlib.sha1(_normalise(text).encode("utf-8")).hexdigest()[:16]


def build_manifest(gold: list[dict], *, seed: int, limit: int) -> dict:
    """
    Describe the gold set without reproducing it.

    Sorted lists, so the committed file diffs cleanly and re-exporting an
    unchanged gold set is a no-op rather than a spurious commit.
    """
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "limit": limit,
        "n": len(gold),
        "note": "Hashes only. Regenerate with: python -m evals.label --manifest",
        "pair_ids": sorted(p["pair_id"] for p in gold),
        "query_ids": sorted({p["query_id"] for p in gold}),
        "evidence_sha": sorted({fingerprint(p["evidence"]) for p in gold}),
        "sentence_sha": sorted({fingerprint(p["sentence"]) for p in gold}),
        # Kept in the clear because the near-duplicate tier needs to compare
        # against something. Content words of a sentence are not the sentence,
        # and they are already derivable from the pair file this sits beside.
        "sentence_words": sorted(
            " ".join(sorted(content_words(p["sentence"]))) for p in gold
        ),
    }


def load_manifest(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def reasons(pair: dict, manifest: dict) -> list[str]:
    """
    Every way this pair touches the gold set. Empty means safe to train on.

    Returns all matching reasons rather than the first, because when a filter
    rejects most of a dataset the useful question is *which* tier is doing it —
    a query-id collision is a sourcing mistake, a near-duplicate is a
    paraphrase problem, and they call for different fixes.
    """
    found: list[str] = []
    if pair.get("pair_id") in set(manifest.get("pair_ids", ())):
        found.append("pair_id")
    if pair.get("query_id") in set(manifest.get("query_ids", ())):
        found.append("query_id")
    if fingerprint(pair.get("evidence", "")) in set(manifest.get("evidence_sha", ())):
        found.append("evidence")
    if fingerprint(pair.get("sentence", "")) in set(manifest.get("sentence_sha", ())):
        found.append("sentence")
    elif _near_duplicate(pair.get("sentence", ""), manifest):
        found.append("sentence~")
    return found


def _near_duplicate(sentence: str, manifest: dict) -> bool:
    candidate = " ".join(sorted(content_words(sentence)))
    if not candidate:
        return False
    return any(
        token_overlap(candidate, known) >= NEAR_DUPLICATE
        for known in manifest.get("sentence_words", ())
    )


def partition(
    pairs: Iterable[dict], manifest: dict
) -> tuple[list[dict], list[tuple[dict, list[str]]]]:
    """Split into (trainable, rejected-with-reasons)."""
    clean: list[dict] = []
    dirty: list[tuple[dict, list[str]]] = []
    for pair in pairs:
        why = reasons(pair, manifest)
        if why:
            dirty.append((pair, why))
        else:
            clean.append(pair)
    return clean, dirty


def assert_clean(pairs: Iterable[dict], manifest: dict) -> None:
    """
    Hard stop before training. Raises rather than filtering.

    A silent filter here would let a contaminated build succeed with a quietly
    smaller dataset; the training notebook should refuse to run instead.
    """
    _, dirty = partition(pairs, manifest)
    if dirty:
        tally: dict[str, int] = {}
        for _, why in dirty:
            for reason in why:
                tally[reason] = tally.get(reason, 0) + 1
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(tally.items()))
        raise ValueError(
            f"{len(dirty)} training pair(s) overlap the gold set ({breakdown}). "
            "Training on these would invalidate the pre-registered comparison."
        )
