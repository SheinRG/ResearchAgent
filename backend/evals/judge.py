"""
Baseline judges for the citation-support task, and the scoring that compares
them to the gold set.

The pre-registered bar (``finetune/PREREGISTRATION.md``) is stated in terms of
four baselines. This module produces predictions for the three that need no
GPU — the two lexical-overlap variants and the prompted Groq judges — and
scores *any* prediction file against the gold labels, so the zero-shot and
fine-tuned Qwen runs (produced elsewhere, in the ``finetune/`` environment)
just have to drop a file in the same format into ``results/judge/``.

Two properties matter more than anything the code does:

**Predictions are label-independent.** A judge sees exactly what the blind
labeller sees — the sentence and the evidence, nothing else. No pair type, no
query, no source title. Predictions can therefore be produced before, during,
or after labelling without contaminating either side, and are cached to disk
so a Groq run is never paid for twice.

**Thresholds for the lexical baselines are chosen by oracle.** The continuous
overlap scores are cached now; at scoring time each variant gets the cutoff
that maximises its own gold macro-F1. That deliberately flatters the baseline
— beating a flattered baseline is the conservative claim, and the bar is
against the stronger of the two variants.

Usage (from ``backend/``):
    python -m evals.judge --lexical                  # cache overlap scores (no network)
    python -m evals.judge --predict reference        # prompted gpt-oss-20b
    python -m evals.judge --predict teacher          # prompted gpt-oss-120b
    python -m evals.judge --score                    # compare everything to gold
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from app.utils.semantic_guards import content_words, token_overlap

from evals.label import (
    SUPPORTED,
    UNSUPPORTED,
    agreement,
    load_candidates,
    load_labels,
)

EVALS_DIR = Path(__file__).parent
DEFAULT_PAIRS = EVALS_DIR / "data" / "pairs.jsonl"
DEFAULT_LABELS = EVALS_DIR / "gold.jsonl"
PRED_DIR = EVALS_DIR / "results" / "judge"

# The prompted-judge aliases, resolved to concrete models via settings so the
# ids live in one place. The filename each writes is derived from the model id,
# not the alias — a results directory full of "reference.jsonl" would stop
# meaning anything the next time Groq retires a model.
ALIASES = ("reference", "teacher")

# Groq's free tier caps tokens per MINUTE (8k), and that cap — not model speed
# — is the real ceiling. The client-side budget stays under it so the run
# paces itself instead of collecting 429s and burning retry budget.
_TPM_BUDGET = 6000

_VERDICT = re.compile(r"\b(unsupported|supported)\b", re.IGNORECASE)

_SYSTEM = (
    "You judge whether a piece of evidence supports a sentence from a "
    "research answer. Reply with exactly one word: supported or unsupported."
)


def build_prompt(sentence: str, evidence: str) -> str:
    """
    The judge's entire view of a pair.

    Deliberately only the sentence and the evidence — the same blindness the
    hand-labeller works under (see ``label.py``). Adding the query, the source
    title, or the pair type would hand the model information the task defines
    as unavailable, and its score would stop being comparable to gold.
    """
    return (
        f"EVIDENCE:\n{evidence}\n\n"
        f"SENTENCE:\n{sentence}\n\n"
        "Does the evidence support the sentence? A sentence is supported only "
        "if the evidence states or directly entails its claim, including any "
        "numbers, dates, and polarity. Being on the same topic is not enough.\n"
        "Answer with exactly one word: supported or unsupported."
    )


def parse_verdict(text: str) -> Optional[str]:
    """
    Pull the verdict out of a model reply.

    The *last* match wins because reasoning models think out loud —
    "...this looks supported at first, but the number differs, so:
    unsupported" must parse as unsupported. The regex is word-bounded because
    "unsupported" contains "supported" as a substring.
    """
    matches = _VERDICT.findall(text or "")
    if not matches:
        return None
    return matches[-1].lower()


def coverage(sentence: str, evidence: str) -> float:
    """
    Fraction of the sentence's content words present in the evidence.

    The second lexical variant. Jaccard divides by the union, so long evidence
    drags every score toward zero regardless of support; coverage only asks
    how much of the *sentence* the evidence lexically accounts for, which is
    the stronger version of the overlap idea and the fairer incumbent to beat.
    """
    words_s, words_e = content_words(sentence), content_words(evidence)
    if not words_s:
        return 0.0
    return len(words_s & words_e) / len(words_s)


# ---------------------------------------------------------------------------
# Prediction storage
# ---------------------------------------------------------------------------

def _pred_path(judge: str) -> Path:
    safe = re.sub(r"[^a-z0-9._-]+", "-", judge.lower())
    return PRED_DIR / f"{safe}.jsonl"


def load_predictions(path: Path) -> dict[str, dict]:
    """pair_id -> record. Later records win, so a rerun corrects in place."""
    out: dict[str, dict] = {}
    path = Path(path)
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            out[record["pair_id"]] = record
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # a truncated tail must not make the file unreadable
    return out


def append_prediction(path: Path, record: dict) -> None:
    """Append-only, written immediately: a 30-minute Groq run must survive a
    closed laptop lid with everything it already paid for on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record(pair_id: str, judge: str, verdict: Optional[str], score: float,
            raw: str = "") -> dict:
    record = {
        "pair_id": pair_id,
        "judge": judge,
        "verdict": verdict,
        "score": score,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # The raw reply is kept only when it failed to parse, which is the only
    # time it says anything `verdict` doesn't. Storing it for every pair would
    # also risk carrying quoted evidence — third-party page text — into a file
    # that is committed, which is exactly what the data/ ignore rule exists to
    # prevent.
    if verdict is None and raw:
        record["raw"] = raw[:200]
    return record


# ---------------------------------------------------------------------------
# Lexical baselines (no network)
# ---------------------------------------------------------------------------

def predict_lexical(candidates: list[dict]) -> dict[str, int]:
    """
    Cache both overlap variants for every pair. Scores only — the verdict is
    decided at scoring time by the oracle threshold, so nothing here needs
    rerunning when gold labels land.
    """
    written = {}
    for judge, fn in (
        ("lexical-jaccard", token_overlap),
        ("lexical-coverage", coverage),
    ):
        path = _pred_path(judge)
        have = load_predictions(path)
        count = 0
        for cand in candidates:
            if cand["pair_id"] in have:
                continue
            score = fn(cand.get("sentence", ""), cand.get("evidence", ""))
            append_prediction(path, _record(cand["pair_id"], judge, None, score))
            count += 1
        written[judge] = count
    return written


# ---------------------------------------------------------------------------
# Prompted Groq judges
# ---------------------------------------------------------------------------

def _resolve_alias(alias: str) -> str:
    from app.config import get_settings

    settings = get_settings()
    if alias == "reference":
        return settings.groq_model
    if alias == "teacher":
        return settings.groq_synth_model
    return alias  # allow an explicit model id


async def predict_groq(
    candidates: list[dict],
    model: str,
    *,
    limit: int = 0,
    tpm_budget: int = _TPM_BUDGET,
) -> int:
    from app.services.llm import get_llm_client

    client = get_llm_client()
    path = _pred_path(model)
    have = load_predictions(path)
    todo = [c for c in candidates if c["pair_id"] not in have]
    if limit:
        todo = todo[:limit]
    if not todo:
        print(f"  {model}: nothing to do ({len(have)} cached)")
        return 0

    print(f"  {model}: {len(todo)} pairs to judge ({len(have)} cached) -> {path}")

    window_start = time.monotonic()
    window_tokens = 0
    done = 0
    for cand in todo:
        prompt = build_prompt(cand.get("sentence", ""), cand.get("evidence", ""))
        # Rough count (chars/4) plus generous headroom for the model's
        # reasoning tokens, which bill against the same cap.
        estimate = len(prompt) // 4 + len(_SYSTEM) // 4 + 300

        if window_tokens + estimate > tpm_budget:
            wait = max(0.0, 60.0 - (time.monotonic() - window_start))
            if wait:
                print(f"    ...TPM budget spent, sleeping {wait:.0f}s "
                      f"({done}/{len(todo)} done)")
                await asyncio.sleep(wait)
            window_start = time.monotonic()
            window_tokens = 0
        window_tokens += estimate

        reply = await client.generate(
            prompt,
            system=_SYSTEM,
            temperature=0.0,
            model=model,
            # Reasoning models spend completion tokens thinking before the
            # one-word answer; a tight cap would truncate mid-thought and
            # return nothing parseable.
            max_tokens=512,
            stage="judge",
        )
        verdict = parse_verdict(reply)
        score = 1.0 if verdict == SUPPORTED else 0.0
        append_prediction(path, _record(cand["pair_id"], model, verdict, score, reply))
        done += 1
        if done % 25 == 0:
            print(f"    {done}/{len(todo)}")

    print(f"  {model}: {done} new predictions")
    return done


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def macro_f1(truths: list[str], preds: list[Optional[str]]) -> float:
    """Unweighted mean of the two classes' F1. A judge that answers
    'supported' to everything lands near 0.45 here, not near the 0.8 accuracy
    the class balance would gift it."""
    scores = []
    for cls in (SUPPORTED, UNSUPPORTED):
        tp = sum(1 for t, p in zip(truths, preds) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(truths, preds) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(truths, preds) if t == cls and p != cls)
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return sum(scores) / len(scores)


def oracle_threshold(scored: list[tuple[float, str]]) -> tuple[float, float]:
    """
    (best threshold, macro-F1 at it) for a continuous-score judge.

    Candidate cutoffs are the midpoints between adjacent observed scores —
    sweeping anything finer cannot change any prediction. Oracle by design:
    see the module docstring.
    """
    if not scored:
        return 0.0, 0.0
    values = sorted({s for s, _ in scored})
    cuts = [0.0] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [values[-1] + 1e-9]
    truths = [t for _, t in scored]
    best = (0.0, 0.0)
    for cut in cuts:
        preds = [SUPPORTED if s >= cut else UNSUPPORTED for s, _ in scored]
        f1 = macro_f1(truths, preds)
        if f1 > best[1]:
            best = (cut, f1)
    return best


def score_judges(
    candidates: list[dict],
    gold: dict[str, dict],
    pred_dir: Path = PRED_DIR,
) -> list[dict]:
    """
    Every prediction file in ``pred_dir`` against the gold labels.

    ``unclear`` pairs are excluded (they are a data problem, reported by
    ``label.py --stats``, not a judging problem). Judges with a continuous
    score and no verdict get the oracle-threshold treatment.
    """
    ids = {c["pair_id"] for c in candidates}
    labelled = {
        pid: rec["label"] for pid, rec in gold.items()
        if pid in ids and rec.get("label") in (SUPPORTED, UNSUPPORTED)
    }

    rows = []
    for path in sorted(Path(pred_dir).glob("*.jsonl")):
        preds = load_predictions(path)
        overlap = [pid for pid in labelled if pid in preds]
        if not overlap:
            rows.append({"judge": path.stem, "n": 0, "macro_f1": 0.0, "note": "no scored pairs"})
            continue

        has_verdicts = any(preds[pid].get("verdict") for pid in overlap)
        if has_verdicts:
            truths = [labelled[pid] for pid in overlap]
            verdicts = [preds[pid].get("verdict") for pid in overlap]
            rows.append({
                "judge": path.stem,
                "n": len(overlap),
                "macro_f1": macro_f1(truths, verdicts),
                "unparseable": sum(1 for v in verdicts if v not in (SUPPORTED, UNSUPPORTED)),
            })
        else:
            scored = [(float(preds[pid].get("score", 0.0)), labelled[pid]) for pid in overlap]
            cut, f1 = oracle_threshold(scored)
            rows.append({
                "judge": path.stem,
                "n": len(scored),
                "macro_f1": f1,
                "oracle_threshold": round(cut, 3),
            })
    return rows


def _print_scores(rows: list[dict], gold_first: dict, gold_recheck: dict) -> None:
    if not rows:
        print("\n  No prediction files in", PRED_DIR)
        print("  Produce some: python -m evals.judge --lexical / --predict reference\n")
        return

    print(f"\n  {'judge':<28} {'n':>5}  {'macro-F1':>8}  notes")
    for row in rows:
        notes = []
        if "oracle_threshold" in row:
            notes.append(f"oracle cut {row['oracle_threshold']}")
        if row.get("unparseable"):
            notes.append(f"{row['unparseable']} unparseable")
        if row.get("note"):
            notes.append(row["note"])
        print(f"  {row['judge']:<28} {row['n']:>5}  {row['macro_f1']:>8.3f}  {', '.join(notes)}")

    self_agree = agreement(gold_first, gold_recheck)
    if self_agree["compared"]:
        print(f"\n  Labeller self-agreement (the ceiling): "
              f"{self_agree['rate']:.0%} over {self_agree['compared']} rechecked pairs")
    else:
        print("\n  No self-agreement measured yet — run: python -m evals.label --recheck 40")
        print("  Quote no model number without it.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover — not a TextIOWrapper
        pass

    parser = argparse.ArgumentParser(description="Citation-judge baselines and scoring.")
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS), help="Capture output (JSONL).")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="Gold label file (JSONL).")
    parser.add_argument("--lexical", action="store_true",
                        help="Cache both lexical-overlap scores for every pair. No network.")
    parser.add_argument("--predict", metavar="MODEL",
                        help="Run a prompted Groq judge: 'reference', 'teacher', or a model id.")
    parser.add_argument("--limit", type=int, default=0, help="Cap pairs for a smoke run.")
    parser.add_argument("--score", action="store_true",
                        help="Score every cached prediction file against gold.")
    args = parser.parse_args()

    try:
        candidates = load_candidates(Path(args.pairs))
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    did_something = False

    if args.lexical:
        written = predict_lexical(candidates)
        for judge, count in written.items():
            print(f"  {judge}: {count} new scores cached")
        did_something = True

    if args.predict:
        model = _resolve_alias(args.predict)
        asyncio.run(predict_groq(candidates, model, limit=args.limit))
        did_something = True

    if args.score:
        first, recheck = load_labels(Path(args.labels))
        rows = score_judges(candidates, first)
        _print_scores(rows, first, recheck)
        did_something = True

    if not did_something:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
