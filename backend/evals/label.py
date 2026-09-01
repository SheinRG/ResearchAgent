"""
Hand-label captured pairs into the gold set.

The gold set is the only thing in this project licensed to produce a quotable
number. Everything else — the 70B teacher's labels, the token-overlap baseline,
the trained judge — gets measured *against* it, so its credibility is the
credibility of the whole result. Three choices protect that, and each of them
costs something:

**The labeller is blind.** You see the sentence and the evidence. You do not see
whether capture called the pair `cited` or `swapped`, and you do not see the
query, the source title, or the domain. The pair type is a hint about the answer
and would anchor you toward it. The rest is information the judge structurally
cannot access — labelling with it produces a standard no model reading only
sentence-plus-evidence could ever meet, which would look like model failure
rather than what it is.

**Presentation is shuffled.** Capture emits a sentence's cited and swapped pairs
next to each other. Labelling them back to back means judging the second
relative to the first rather than on its own. The shuffle is seeded, so the
order is reproducible.

**Your own consistency is measured.** ``--recheck`` re-presents pairs you have
already labelled, blind, and reports how often you agree with yourself. That
number is the ceiling: a judge scoring above your self-agreement is fitting your
noise, not learning the task. Publishing a model score without it is quoting a
measurement with no error bar.

The label file holds only pair ids and labels — no sentences, no evidence, no
scraped text. That keeps it committable, unlike the candidates it refers to.

Usage:
    python -m evals.label --pairs evals/data/pairs.jsonl        # label
    python -m evals.label --stats                               # progress + balance
    python -m evals.label --recheck 40                          # self-agreement pass
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from evals.contamination import build_manifest

EVALS_DIR = Path(__file__).parent
DEFAULT_PAIRS = EVALS_DIR / "data" / "pairs.jsonl"
DEFAULT_LABELS = EVALS_DIR / "gold.jsonl"
# Lives under finetune/ because that is who reads it: the training-data build
# needs to know what gold is without importing the backend.
DEFAULT_MANIFEST = EVALS_DIR.parent.parent / "finetune" / "gold_manifest.json"

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
UNCLEAR = "unclear"

# `unclear` is a real outcome, not a cop-out: evidence that is genuinely
# ambiguous should be excluded from the gold set rather than forced into a
# binary that then gets treated as ground truth. It is dropped from scoring and
# reported separately, so a high unclear rate is visible as a data problem
# instead of hiding as label noise.
VALID_LABELS = (SUPPORTED, UNSUPPORTED, UNCLEAR)

_KEYS = {
    "s": SUPPORTED,
    "u": UNSUPPORTED,
    "k": UNCLEAR,
}


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def load_candidates(path: Path) -> list[dict]:
    """Read capture output. Missing file is an error worth naming clearly."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No candidates at {path}. Produce some first:\n"
            f"    python -m evals.run_eval --capture {path}"
        )
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_labels(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Read the label file.

    Returns:
        (first_pass, recheck_pass), each mapping pair_id -> record. Later
        records for the same pair and pass win, so a correction is just another
        append rather than an edit.
    """
    first: dict[str, dict] = {}
    recheck: dict[str, dict] = {}
    path = Path(path)
    if not path.exists():
        return first, recheck

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            target = recheck if int(record.get("pass", 1)) >= 2 else first
            target[record["pair_id"]] = record
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue  # a truncated tail must not make the file unreadable
    return first, recheck


def append_label(
    path: Path,
    pair_id: str,
    label: str,
    *,
    note: str = "",
    pass_no: int = 1,
) -> None:
    """Append one label. Append-only: the file is a log, not a database."""
    if label not in VALID_LABELS:
        raise ValueError(f"invalid label {label!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "pair_id": pair_id,
        "label": label,
        "note": note,
        "pass": pass_no,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def pending(candidates: list[dict], labelled: dict[str, dict], seed: int = 0) -> list[dict]:
    """
    Unlabelled candidates, shuffled deterministically.

    Shuffled because capture emits a sentence's cited and swapped pairs
    adjacently, and labelling them back to back judges the second relative to
    the first. Seeded so a resumed session continues the same order.
    """
    remaining = [c for c in candidates if c.get("pair_id") not in labelled]
    random.Random(seed).shuffle(remaining)
    return remaining


def recheck_sample(
    candidates: list[dict],
    labelled: dict[str, dict],
    count: int,
    seed: int = 0,
) -> list[dict]:
    """Already-labelled pairs to re-present, chosen without regard to label."""
    done = [c for c in candidates if c.get("pair_id") in labelled]
    rng = random.Random(seed)
    rng.shuffle(done)
    return done[:count]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def label_stats(candidates: list[dict], labelled: dict[str, dict]) -> dict:
    """Progress and class balance, plus how the labels fell across pair types."""
    counts = {label: 0 for label in VALID_LABELS}
    by_type: dict[str, dict[str, int]] = {}

    by_id = {c["pair_id"]: c for c in candidates}
    for pair_id, record in labelled.items():
        label = record.get("label")
        if label in counts:
            counts[label] += 1
        pair_type = by_id.get(pair_id, {}).get("pair_type", "unknown")
        by_type.setdefault(pair_type, {label: 0 for label in VALID_LABELS})
        if label in counts:
            by_type[pair_type][label] += 1

    scored = counts[SUPPORTED] + counts[UNSUPPORTED]
    return {
        "candidates": len(candidates),
        "labelled": len(labelled),
        "remaining": max(0, len(candidates) - len(labelled)),
        "counts": counts,
        "scoreable": scored,
        "supported_rate": (counts[SUPPORTED] / scored) if scored else 0.0,
        "unclear_rate": (counts[UNCLEAR] / len(labelled)) if labelled else 0.0,
        "by_pair_type": by_type,
    }


def agreement(first: dict[str, dict], second: dict[str, dict]) -> dict:
    """
    How often the two passes agree, over pairs labelled in both.

    Reported over *all* overlapping pairs including `unclear`, because
    "I called it supported yesterday and unclear today" is a disagreement — and
    dropping it would flatter the number.
    """
    shared = sorted(set(first) & set(second))
    if not shared:
        return {"compared": 0, "agreed": 0, "rate": 0.0, "disagreements": []}

    agreed = 0
    disagreements: list[dict] = []
    for pair_id in shared:
        a = first[pair_id].get("label")
        b = second[pair_id].get("label")
        if a == b:
            agreed += 1
        else:
            disagreements.append({"pair_id": pair_id, "first": a, "second": b})

    return {
        "compared": len(shared),
        "agreed": agreed,
        "rate": agreed / len(shared),
        "disagreements": disagreements,
    }


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int, indent: str = "  ") -> str:
    paragraphs = (text or "").splitlines() or [""]
    out: list[str] = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(
            paragraph, width=width,
            initial_indent=indent, subsequent_indent=indent,
        ) or [indent])
    return "\n".join(out)


def render(candidate: dict, position: int, total: int, width: int = 88) -> str:
    """
    One pair, as the labeller sees it.

    Deliberately withholds pair_type, query, source title and domain — see the
    module docstring. Changing that changes what the gold set means.
    """
    rule = "-" * width
    return (
        f"\n{rule}\n"
        f"  {position}/{total}\n"
        f"{rule}\n"
        f"\n  SENTENCE\n{_wrap(candidate.get('sentence', ''), width - 4, '    ')}\n"
        f"\n  EVIDENCE\n{_wrap(candidate.get('evidence', ''), width - 4, '    ')}\n"
        f"\n{rule}\n"
        "  Does the evidence support the sentence?\n"
        "  [s] supported   [u] unsupported   [k] unclear   [q] save and quit\n"
    )


def run_session(
    candidates: list[dict],
    labels_path: Path,
    *,
    pass_no: int = 1,
    seed: int = 0,
    limit: Optional[int] = None,
    prompt: Callable[[str], str] = input,
    write: Callable[[str], None] = lambda s: print(s, end=""),
) -> int:
    """
    Label until the queue empties, the limit is hit, or the user quits.

    ``prompt`` and ``write`` are injected so the loop is testable without a
    terminal. Every answer is written immediately rather than batched at the
    end: an hour of labelling must not be lost to a closed window.

    Returns:
        How many labels were recorded this session.
    """
    queue = candidates[:limit] if limit else candidates
    total = len(queue)
    if not total:
        write("Nothing to label.\n")
        return 0

    width = min(88, max(60, shutil.get_terminal_size((88, 24)).columns - 2))
    recorded = 0

    for position, candidate in enumerate(queue, 1):
        write(render(candidate, position, total, width))

        while True:
            answer = prompt("  > ").strip().lower()
            if answer in ("q", "quit"):
                write(f"\nSaved {recorded} label(s) to {labels_path}\n")
                return recorded
            if answer in _KEYS:
                append_label(
                    labels_path,
                    candidate["pair_id"],
                    _KEYS[answer],
                    pass_no=pass_no,
                )
                recorded += 1
                break
            write("  Enter s, u, k, or q.\n")

    write(f"\nDone — {recorded} label(s) recorded in {labels_path}\n")
    return recorded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_stats(stats: dict) -> None:
    counts = stats["counts"]
    print(
        f"\n  {stats['labelled']}/{stats['candidates']} labelled "
        f"({stats['remaining']} remaining)"
    )
    print(
        f"  supported {counts[SUPPORTED]}   unsupported {counts[UNSUPPORTED]}   "
        f"unclear {counts[UNCLEAR]}"
    )
    if stats["scoreable"]:
        print(f"  class balance: {stats['supported_rate']:.0%} supported")
    if stats["unclear_rate"] > 0.15:
        print(
            f"  NOTE: {stats['unclear_rate']:.0%} unclear — that is high enough to "
            "suspect the evidence, not the labeller."
        )

    if stats["by_pair_type"]:
        print("\n  by capture type (not shown while labelling):")
        for pair_type, breakdown in sorted(stats["by_pair_type"].items()):
            print(
                f"    {pair_type:<8} supported {breakdown[SUPPORTED]:>4}   "
                f"unsupported {breakdown[UNSUPPORTED]:>4}   "
                f"unclear {breakdown[UNCLEAR]:>4}"
            )
    print()


def _print_agreement(result: dict) -> None:
    if not result["compared"]:
        print(
            "\n  No pairs labelled twice yet. Run --recheck N after a labelling "
            "session to measure your own consistency.\n"
        )
        return
    print(
        f"\n  Self-agreement: {result['agreed']}/{result['compared']} "
        f"= {result['rate']:.0%}"
    )
    print(
        "  This is the ceiling. A judge scoring above it is fitting label noise,\n"
        "  not learning the task — quote it alongside any model number.\n"
    )
    for item in result["disagreements"][:10]:
        print(f"    {item['pair_id']}  {item['first']} -> {item['second']}")
    if len(result["disagreements"]) > 10:
        print(f"    ... and {len(result['disagreements']) - 10} more")
    print()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover — not a TextIOWrapper
        pass

    parser = argparse.ArgumentParser(description="Hand-label citation-judge pairs.")
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS), help="Capture output (JSONL).")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="Label file (JSONL).")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many pairs.")
    parser.add_argument("--seed", type=int, default=0, help="Presentation-order seed.")
    parser.add_argument("--stats", action="store_true", help="Show progress and stop.")
    parser.add_argument(
        "--recheck", type=int, default=0, metavar="N",
        help="Re-label N already-labelled pairs, blind, to measure self-agreement.",
    )
    parser.add_argument(
        "--agreement", action="store_true",
        help="Report self-agreement from existing recheck labels and stop.",
    )
    parser.add_argument(
        "--manifest", nargs="?", const=str(DEFAULT_MANIFEST), metavar="PATH",
        help="Export the gold set's fingerprints (hashes only) for the "
             "training-data contamination guard, and stop.",
    )
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    labels_path = Path(args.labels)

    try:
        candidates = load_candidates(pairs_path)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if args.manifest:
        # Deliberately independent of how many pairs are *labelled*: the gold
        # set is defined by the seeded order and the limit, so the guard can be
        # exported (and the training data built) while labelling is still in
        # progress. What must never drift is the --limit used here and there.
        gold = pending(candidates, {}, seed=args.seed)[: args.limit or None]
        manifest = build_manifest(gold, seed=args.seed, limit=args.limit or len(gold))
        out = Path(args.manifest)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(
            f"  Gold fingerprints for {manifest['n']} pair(s) "
            f"across {len(manifest['query_ids'])} quer(ies) -> {out}\n"
            f"  Hashes only; no sentences or evidence are written."
        )
        return 0

    first, recheck = load_labels(labels_path)

    if args.stats:
        _print_stats(label_stats(candidates, first))
        _print_agreement(agreement(first, recheck))
        return 0

    if args.agreement:
        _print_agreement(agreement(first, recheck))
        return 0

    if args.recheck:
        queue = recheck_sample(candidates, first, args.recheck, seed=args.seed + 1)
        if not queue:
            print("Nothing labelled yet to re-check.", file=sys.stderr)
            return 1
        print(
            f"\nRe-checking {len(queue)} already-labelled pairs, blind.\n"
            "Answer as if you had never seen them — the point is to find out how "
            "often you agree with yourself."
        )
        run_session(queue, labels_path, pass_no=2, seed=args.seed)
        first, recheck = load_labels(labels_path)
        _print_agreement(agreement(first, recheck))
        return 0

    queue = pending(candidates, first, seed=args.seed)
    if not queue:
        print("Every candidate is labelled.")
        _print_stats(label_stats(candidates, first))
        return 0

    run_session(
        queue, labels_path,
        pass_no=1, seed=args.seed,
        limit=args.limit or None,
    )
    first, _ = load_labels(labels_path)
    _print_stats(label_stats(candidates, first))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
