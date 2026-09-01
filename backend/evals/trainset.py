"""
Turn public NLI / fact-verification data into citation-support training pairs.

This module is the *contract* half of the training set: what an example is,
what the public labels mean once translated, and what may never enter. The
downloading half lives in ``finetune/build_dataset.py`` because it needs
``datasets`` and a GPU box; everything here is stdlib so CI covers it.

Nothing here is pure plumbing. Two decisions in this file change what the
model is taught the task *means*, and both are recorded rather than assumed.

**`neutral` maps to `unsupported`, not to a third class or a dropped row.**
NLI has three answers — entailment, contradiction, neutral. Citation support
has two, and the question is not "does the evidence disagree?" but "does the
evidence *establish* the claim?". A citation pointing at a chunk that neither
confirms nor denies the sentence is a bad citation, and it is the single most
common failure the judge exists to catch — the synthesizer attaching `[3]` to
a plausible sentence the chunk simply does not cover. Folding neutral into
`unsupported` is therefore the mapping that matches the deployed question.
Dropping neutral rows instead would train a contradiction detector and call it
a support judge, and it would look *better* on public data while being useless
on ours.

**Evidence length is a known distribution gap, not a bug to hide.** Public
premises are one or two sentences; the real evidence chunks average 174 words.
The public slice teaches the *relation*, the in-domain slice teaches the
*format*. The writeup says so, and the ablation that drops the public slice is
what measures whether the transfer actually happened.

Licence note: only permissively-licensed, ungated sources belong here, because
the pre-registration promises a notebook a stranger can re-run. Non-commercial
corpora are deliberately absent — **ANLI** (CC BY-NC 4.0) and **SciFact**
(CC BY-NC 2.0) are the two that would otherwise be obvious picks. Adding a
mapping for one is how it ends up in a build, so they have none. See
``finetune/LICENCES.md``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional

from evals.contamination import fingerprint, reasons
from evals.label import SUPPORTED, UNSUPPORTED

# Every public label this project knows how to translate, per source. Anything
# absent raises rather than defaulting: a silently mishandled label is a silently
# mislabelled training set, and the whole point of the module docstring above is
# that these choices are deliberate.
#
# ``None`` means "drop this row" and is reserved for rows the *source* marks as
# having no gold label (SNLI/MNLI use -1 when annotators failed to agree).
LABEL_MAP: dict[str, dict[object, Optional[str]]] = {
    "fever": {
        "SUPPORTS": SUPPORTED,
        "REFUTES": UNSUPPORTED,
        "NOT ENOUGH INFO": UNSUPPORTED,
    },
    "vitaminc": {
        "SUPPORTS": SUPPORTED,
        "REFUTES": UNSUPPORTED,
        "NOT ENOUGH INFO": UNSUPPORTED,
    },
    # SNLI/MNLI ship integer labels; the strings are accepted too so a caller
    # reading a raw JSONL dump does not have to convert first.
    "mnli": {
        0: SUPPORTED, "entailment": SUPPORTED,
        1: UNSUPPORTED, "neutral": UNSUPPORTED,
        2: UNSUPPORTED, "contradiction": UNSUPPORTED,
        -1: None,
    },
}
LABEL_MAP["snli"] = LABEL_MAP["mnli"]


class UnknownLabel(ValueError):
    """A source emitted a label the mapping does not cover."""


def map_label(source: str, raw: object) -> Optional[str]:
    """
    Translate one source label. ``None`` means the row has no gold label.

    Raises on an unrecognised label rather than guessing, because the failure
    mode of guessing is a training set that looks fine and teaches the wrong
    task.
    """
    try:
        table = LABEL_MAP[source]
    except KeyError:
        raise UnknownLabel(
            f"no label mapping for source {source!r}; "
            f"known: {', '.join(sorted(LABEL_MAP))}"
        ) from None
    if raw not in table:
        raise UnknownLabel(f"source {source!r} emitted unmapped label {raw!r}")
    return table[raw]


def to_example(
    *, sentence: str, evidence: str, label: str, source: str, origin_label: object
) -> dict:
    """
    One training row, in the shape the judge is prompted in.

    ``origin_label`` is kept so the neutral-to-unsupported decision stays
    auditable after the fact: without it, a reviewer cannot tell a genuine
    contradiction from a neutral row, and cannot re-derive the set under a
    different mapping without re-downloading everything.
    """
    if label not in (SUPPORTED, UNSUPPORTED):
        raise ValueError(f"invalid label {label!r}")
    return {
        "sentence": " ".join((sentence or "").split()),
        "evidence": " ".join((evidence or "").split()),
        "label": label,
        "source": source,
        "origin_label": origin_label,
    }


def example_key(example: dict) -> str:
    """Identity of a training row: the pair, not the row's provenance."""
    return hashlib.sha1(
        (fingerprint(example["sentence"]) + fingerprint(example["evidence"])).encode()
    ).hexdigest()[:16]


def build(
    rows: Iterable[dict],
    manifest: dict,
    *,
    source: str,
    seen: Optional[set[str]] = None,
) -> tuple[list[dict], dict]:
    """
    Map, gate, and de-duplicate one source into training examples.

    ``rows`` are dicts with ``sentence``/``evidence``/``label`` already
    extracted from whatever the upstream schema was — keeping the upstream
    field names out of here is what lets this module stay dependency-free.

    ``seen`` carries de-duplication state across sources, so the same pair
    arriving from two corpora is counted once. Passing it in rather than
    hiding it in a global keeps the build reproducible.

    Returns (examples, report). The report is not decoration: a slice that
    silently loses 90% of its rows to the gate is the bug this whole module
    exists to make visible.
    """
    seen = seen if seen is not None else set()
    kept: list[dict] = []
    report = Counter()

    for row in rows:
        report["rows"] += 1
        label = map_label(source, row["label"])
        if label is None:
            report["dropped_no_gold"] += 1
            continue

        example = to_example(
            sentence=row["sentence"],
            evidence=row["evidence"],
            label=label,
            source=source,
            origin_label=row["label"],
        )
        if not example["sentence"] or not example["evidence"]:
            report["dropped_empty"] += 1
            continue

        why = reasons(example, manifest)
        if why:
            report["dropped_contaminated"] += 1
            for reason in why:
                report[f"contaminated_{reason}"] += 1
            continue

        key = example_key(example)
        if key in seen:
            report["dropped_duplicate"] += 1
            continue
        seen.add(key)

        kept.append(example)
        report[label] += 1

    report["kept"] = len(kept)
    return kept, dict(report)


def balance(
    examples: list[dict], *, seed: int = 0, ratio: float = 1.0
) -> list[dict]:
    """
    Downsample the majority class to at most ``ratio`` times the minority.

    Needed because the mapping above is inherently lopsided: folding both
    `neutral` and `contradiction` into `unsupported` turns a balanced
    three-way corpus into roughly 1:2. Left alone, the model learns that
    guessing `unsupported` is usually right — which is precisely the
    degenerate strategy macro-F1 was chosen to expose at eval time. Fixing it
    at eval time only is fixing it too late.

    Deterministic: shuffles with a seeded RNG so a rebuild is reproducible.
    """
    if ratio < 1:
        # Below 1 the cap falls under the minority count and the function
        # quietly truncates *both* classes, which is never what a caller
        # asking to "balance" wants.
        raise ValueError("ratio must be >= 1 (majority per minority)")
    by_label: dict[str, list[dict]] = {}
    for example in examples:
        by_label.setdefault(example["label"], []).append(example)
    if len(by_label) < 2:
        return list(examples)

    minority = min(len(v) for v in by_label.values())
    cap = int(minority * ratio)
    rng = random.Random(seed)

    out: list[dict] = []
    for label in sorted(by_label):
        group = by_label[label]
        rng.shuffle(group)
        out.extend(group[:cap] if len(group) > cap else group)
    rng.shuffle(out)
    return out


def compose(examples: list[dict]) -> dict:
    """Human-readable summary — what went in, by source and by label."""
    by_source = Counter(e["source"] for e in examples)
    by_label = Counter(e["label"] for e in examples)
    supported = by_label.get(SUPPORTED, 0)
    return {
        "total": len(examples),
        "by_source": dict(sorted(by_source.items())),
        "by_label": dict(sorted(by_label.items())),
        "supported_share": round(supported / len(examples), 3) if examples else 0.0,
    }


def write_jsonl(examples: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
