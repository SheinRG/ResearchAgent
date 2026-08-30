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
    pending,
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

# Per-minute pacing. This is NOT the constraint that actually bites: a full
# 311-pair run died at 225 on tokens per DAY (200k on the free tier, ~890
# tokens a pair), having never once hit the per-minute ceiling. Pacing at 6000
# stretched that run to 47 minutes and bought nothing, so the budget now sits
# high enough to stay out of the way while still smoothing bursts.
#
# The daily cap is the one to plan around: it is why `--limit` exists and why
# the run order matches the labeller's. Roughly 220 pairs per model per day.
_TPM_BUDGET = 14000

# The gpt-oss models reason before answering, and the reasoning bills against
# the completion budget. At 512 the model spent the whole allowance thinking
# and returned an *empty* string — measured, not assumed: the same pair
# answered correctly at 1024. A cap that truncates costs the judge a point it
# had earned, which would understate the baseline and flatter anything
# measured against it, so the ceiling is set well clear of what was needed.
_MAX_TOKENS = 2048

# Reasoning tokens a pair is assumed to spend, for pacing only. Deliberately
# nearer what the probe showed than the one-word answer would suggest —
# underestimating here means pacing into 429s.
_REASONING_ALLOWANCE = 600

_VERDICT = re.compile(r"\b(unsupported|supported)\b", re.IGNORECASE)

_SYSTEM = (
    "You judge whether a piece of evidence supports a sentence from a "
    "research answer. Reply with exactly one word: supported or unsupported."
)


def selection_order(candidates: list[dict], seed: int = 0) -> list[dict]:
    """
    Candidates in the order the labeller will meet them.

    Delegates to ``label.pending`` rather than re-deriving the shuffle, for the
    same reason capture reuses ``select_evidence``: two implementations of the
    same order are two things that can drift.

    This is what makes ``--limit N`` mean the same set of pairs here as it does
    in ``label.py``. Judging in file order instead spends a hard daily token
    budget on pairs nobody will ever label — measured, when it happened: 79 of
    225 predictions landed outside the 200 the labeller would see, while 54
    pairs inside it had no prediction at all.
    """
    return pending(candidates, {}, seed=seed)


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

def _is_daily_cap(error: Exception) -> bool:
    """
    Whether this is the per-*day* token cap rather than a passing 429.

    Both arrive as 429s and the client's retry loop treats them alike, which is
    right for the per-minute limit and useless for this one: the daily cap
    clears tomorrow. Matching on the message is unlovely, but the distinction
    exists only in the text — the status code is identical either way.
    """
    text = str(error).lower()
    return "tokens per day" in text or "tpd" in text


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
        # Rough count (chars/4) plus headroom for the model's reasoning
        # tokens, which bill against the same cap.
        estimate = len(prompt) // 4 + len(_SYSTEM) // 4 + _REASONING_ALLOWANCE

        if window_tokens + estimate > tpm_budget:
            wait = max(0.0, 60.0 - (time.monotonic() - window_start))
            if wait:
                print(f"    ...TPM budget spent, sleeping {wait:.0f}s "
                      f"({done}/{len(todo)} done)")
                await asyncio.sleep(wait)
            window_start = time.monotonic()
            window_tokens = 0
        window_tokens += estimate

        try:
            reply = await client.generate(
                prompt,
                system=_SYSTEM,
                temperature=0.0,
                model=model,
                max_tokens=_MAX_TOKENS,
                stage="judge",
            )
        except Exception as e:
            # The daily token cap is not a transient failure the retry loop can
            # wait out — it clears tomorrow, not in seconds. Everything judged
            # so far is already on disk, so the useful thing is to say where it
            # stopped and stop, rather than bury that under a traceback.
            if _is_daily_cap(e):
                print(f"\n  Daily token cap reached for {model}. "
                      f"{done}/{len(todo)} judged this run; {len(have) + done} cached total.")
                print("  Everything judged so far is saved. Re-run the same command "
                      "tomorrow to continue where it stopped.")
                return done
            raise

        verdict = parse_verdict(reply)

        # An empty reply is the signature of reasoning that ran out of budget
        # rather than of a model with no opinion, so it gets one more try with
        # room to finish. Recording the truncation as a wrong answer would be
        # scoring the harness, and it would understate a baseline the
        # fine-tune is required to beat.
        if verdict is None:
            window_tokens += _MAX_TOKENS
            reply = await client.generate(
                prompt,
                system=_SYSTEM,
                temperature=0.0,
                model=model,
                max_tokens=_MAX_TOKENS * 2,
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


def verdict_on_bar(rows: list[dict], reference_model: str = "") -> dict:
    """
    Read the pre-registered bar off the scored baselines.

    Everything here is stated in ``finetune/PREREGISTRATION.md`` and is
    deliberately computed rather than remembered: a bar the tool restates from
    the numbers is one nobody can quietly round in their favour later.

    The gate is the part that matters now. If the prompted reference judge is
    already near-perfect on gold, "match the bigger model" has no headroom left
    to demonstrate and the project is supposed to stop and rethink rather than
    spend training compute proving nothing.
    """
    by_judge = {row["judge"]: row for row in rows if row.get("n")}

    lexical = {name: row for name, row in by_judge.items() if name.startswith("lexical-")}
    strongest = max(lexical.values(), key=lambda r: r["macro_f1"], default=None)

    reference = None
    if reference_model:
        reference = by_judge.get(_pred_path(reference_model).stem)
    zero_shot = next(
        (row for name, row in by_judge.items() if "qwen" in name and "zero" in name),
        None,
    )

    out: dict = {
        "strongest_lexical": strongest,
        "reference": reference,
        "zero_shot": zero_shot,
        "targets": {},
        "gate": None,
    }
    if strongest:
        out["targets"]["beat_lexical"] = strongest["macro_f1"] + 0.15
    if zero_shot:
        out["targets"]["beat_zero_shot"] = zero_shot["macro_f1"] + 0.10
    if reference:
        out["targets"]["stretch_reference"] = reference["macro_f1"] - 0.03
        # ">= ~0.94" in the pre-registration. Stated as a strict comparison so
        # the tool never has to decide what "about" means.
        out["gate"] = {
            "reference_f1": reference["macro_f1"],
            "no_headroom": reference["macro_f1"] >= 0.94,
        }
    return out


def _print_bar(bar: dict) -> None:
    strongest, reference = bar["strongest_lexical"], bar["reference"]

    if strongest:
        print(f"  Strongest lexical baseline: {strongest['judge']} "
              f"at {strongest['macro_f1']:.3f}")
    if not reference:
        print("  Reference judge not scored yet — the Milestone 1 gate is still open.")
        print("    python -m evals.judge --predict reference\n")
        return

    targets = bar["targets"]
    print("\n  Pre-registered bar for the trained judge (finetune/PREREGISTRATION.md):")
    if "beat_lexical" in targets:
        print(f"    must hit   >= {targets['beat_lexical']:.3f}  "
              f"(strongest lexical + 15 pts)")
    if "beat_zero_shot" in targets:
        print(f"    must hit   >= {targets['beat_zero_shot']:.3f}  "
              f"(zero-shot Qwen3-1.7B + 10 pts)")
    else:
        print("    must hit   -- pending zero-shot Qwen3-1.7B, produced in finetune/")
    if "stretch_reference" in targets:
        print(f"    stretch    >= {targets['stretch_reference']:.3f}  "
              f"(within 3 pts of prompted {reference['judge']})")

    gate = bar["gate"]
    print(f"\n  Milestone 1 gate: reference judge scores {gate['reference_f1']:.3f} on gold.")
    if gate["no_headroom"]:
        print("    STOP AND REVISIT. At >= 0.94 the prompted model has essentially")
        print("    solved the task, so 'match the bigger model' has no headroom to")
        print("    demonstrate and training compute would prove nothing.")
    else:
        print("    Headroom exists — proceed to Milestone 2 (training data).")
    print()


def discriminate(candidates: list[dict], pred_dir: Path = PRED_DIR) -> list[dict]:
    """
    How differently each judge treats cited pairs versus swapped ones.

    The one check on the data plan that needs no gold labels. The swapped pairs
    are the free negatives the whole training set leans on, and `README.md`
    states the condition they have to meet: if they come back mostly
    *supported*, they are not negatives and the plan needs revisiting.

    A judge that answers the same way to both is also visible here, and would
    mean the pairs carry no signal to learn from — worth knowing before anyone
    spends an hour labelling rather than after.
    """
    by_id = {c["pair_id"]: c for c in candidates}
    rows = []
    for path in sorted(Path(pred_dir).glob("*.jsonl")):
        preds = load_predictions(path)
        buckets: dict[str, list[Optional[str]]] = {}
        for pair_id, record in preds.items():
            pair_type = by_id.get(pair_id, {}).get("pair_type", "unknown")
            buckets.setdefault(pair_type, []).append(record.get("verdict"))
        if not any(v for verdicts in buckets.values() for v in verdicts):
            continue  # lexical files carry scores, not verdicts

        rates = {
            pair_type: (
                sum(1 for v in verdicts if v == SUPPORTED) / len(verdicts),
                len(verdicts),
            )
            for pair_type, verdicts in buckets.items()
        }
        cited = rates.get("cited", (0.0, 0))
        swapped = rates.get("swapped", (0.0, 0))
        rows.append({
            "judge": path.stem,
            "rates": rates,
            "gap": cited[0] - swapped[0],
        })
    return rows


def _print_discrimination(rows: list[dict]) -> None:
    if not rows:
        print("\n  No judge with verdicts yet — run --predict first.\n")
        return

    print("\n  Support rate by capture type (no gold labels needed):\n")
    for row in rows:
        print(f"  {row['judge']}")
        for pair_type, (rate, n) in sorted(row["rates"].items()):
            print(f"    {pair_type:<8} n={n:<5} judged supported {rate:>5.0%}")
        print(f"    gap (cited - swapped): {row['gap']:+.0%}")

    print("\n  What to read here:")
    print("    Swapped pairs mostly SUPPORTED would mean the free negatives are")
    print("    not negatives, and the data plan needs revisiting.")
    print("    A gap near zero would mean the pairs carry no signal to learn.")
    print("  Judge opinion, not ground truth — only the gold set settles it.\n")


def _print_scores(rows: list[dict], gold_first: dict, gold_recheck: dict) -> None:
    if not rows:
        print("\n  No prediction files in", PRED_DIR)
        print("  Produce some: python -m evals.judge --lexical / --predict reference\n")
        return

    # Every judge scoring zero over zero pairs means the gold set is missing,
    # not that the judges failed. Saying "run --predict" here would send
    # someone to re-run predictions that already exist.
    if not any(row.get("n") for row in rows):
        print(f"\n  {len(rows)} prediction file(s) cached, but nothing to score them "
              "against yet.")
        print("  The gold set is hand-labelled and cannot be generated — it is what")
        print("  every number here gets measured by:\n")
        print("    python -m evals.label --limit 200   # label, blind; resumable")
        print("    python -m evals.label --recheck 40  # then measure self-agreement")
        print("    python -m evals.judge --score       # then this becomes a result\n")
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

    print()
    _print_bar(verdict_on_bar(rows, _resolve_alias("reference")))

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
    parser.add_argument("--limit", type=int, default=0,
                        help="Judge only the first N in the labeller's order — "
                             "use the same N as `label --limit N`.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Selection-order seed. Must match label.py's.")
    parser.add_argument("--score", action="store_true",
                        help="Score every cached prediction file against gold.")
    parser.add_argument("--diagnose", action="store_true",
                        help="Cited-vs-swapped support rates. Checks the data plan "
                             "without gold labels.")
    args = parser.parse_args()

    try:
        candidates = load_candidates(Path(args.pairs))
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    # Judge in the order the labeller labels, so --limit picks the same pairs
    # in both tools and a scarce daily token budget is spent on pairs that will
    # actually have a gold label to be scored against.
    candidates = selection_order(candidates, seed=args.seed)

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

    if args.diagnose:
        _print_discrimination(discriminate(candidates))
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
