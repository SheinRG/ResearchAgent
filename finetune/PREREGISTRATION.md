# Pre-registration: citation-support judge

Committed **before** any training run, and the results get published either way.
That is the whole point of this file: the success bar cannot move after the
numbers exist. If the trained judge misses the bar, the writeup says so.

## The task

Given one answer sentence and the evidence chunk behind the source it cites,
emit a single token — `supported` or `unsupported` — with the token's softmax
probability kept as a continuous confidence score.

This replaces the lexical-overlap approach (`semantic_guards.token_overlap`
applied to sentence-vs-evidence), which the roadmap already argues is
untrustworthy for exactly the case that matters: same run, same topic, shared
vocabulary, different evidence.

## The model

- **Base:** Qwen3-1.7B — Apache-2.0 and ungated, so the notebook is
  re-runnable by strangers without a licence click-through.
- **Method:** 4-bit QLoRA. At 1.7B parameters, 4-bit is a *choice* (recipe
  transfer and batch size), not a memory necessity — the writeup says so
  rather than implying the model wouldn't fit otherwise.
- **Deployment:** offline / eval-time only. It is never wired into the
  serving path: the production box is a 512MB starter instance and `torch`
  is deliberately absent from `backend/requirements.txt`.

## The data

| Slice | Size | Source |
| --- | --- | --- |
| Public NLI / attribution pairs | ~5–8k | Existing open datasets |
| In-domain distilled pairs | ~1–2k | Labelled by `openai/gpt-oss-120b` (the teacher) over captured pipeline runs |
| Hard negatives | free | Same-run chunk swaps from `evals/capture.py` |
| **Gold set (held out)** | **~200** | **Hand-labelled, blind, via `evals/label.py`** |

The gold set is never trained on and is the only artifact licensed to produce
a quotable number. Its protocol (blind labeller, seeded shuffle, measured
self-agreement, `unclear` excluded from scoring and reported separately) lives
in `backend/evals/label.py` and is part of this pre-registration.

### Contamination rule

"Never trained on" is stricter than "no shared pair ids", and the difference
is not academic. Capture emits many pairs per query and pairs from one query
share retrieved evidence chunks, so a pair with a fresh id can still carry the
gold set's evidence. Measured on the first capture: of the 111 candidates
outside the gold-200, **97 reuse a gold evidence chunk, 81 reuse a gold
sentence verbatim, and 0 are clean** — all 15 captured queries are touched by
the gold set.

A training pair is therefore disqualified if it shares **any** of: `pair_id`,
`query_id`, normalised evidence text, or a near-duplicate sentence
(content-word Jaccard ≥ 0.70) with gold. `query_id` is the binding one — same
run means same retrieved corpus.

Enforced by `backend/evals/contamination.py` against `finetune/gold_manifest.json`
(hashes only, no evidence, no labels), which is regenerated with
`python -m evals.label --manifest`. The training build calls `assert_clean`
and **raises** rather than filtering: a silent drop would turn contamination
into a quietly smaller dataset nobody audits.

Consequence, recorded here rather than discovered later: the in-domain slice
cannot be salvaged from the existing capture. It requires new capture runs on
queries disjoint from the 15 already used.

## The baselines

Four, all evaluated on the same gold set, all predictions produced *before*
gold labels are consulted:

1. **Lexical overlap** — the incumbent. Both variants are computed:
   Jaccard (`token_overlap(sentence, evidence)`, unmodified) and sentence-side
   coverage (`|s∩e| / |s|`, which does not punish long evidence). Each gets
   its **oracle threshold** — the cutoff that maximises its own gold macro-F1.
   That deliberately flatters the baseline; the bar below is against the
   **stronger** of the two variants. Beating a flattered baseline is the
   conservative claim.
2. **Zero-shot Qwen3-1.7B** — the base model, prompted, untrained. Isolates
   what the fine-tune added.
3. **Prompted `openai/gpt-oss-20b`** — the reference judge (what a competent
   prompted small model does).
4. **Prompted `openai/gpt-oss-120b`** — the teacher. Measures the ceiling
   distillation could carry.

## The bar

Metric: **macro-F1** over {supported, unsupported} on the gold set,
`unclear` pairs excluded. Every number is quoted next to the labeller's
self-agreement rate, which is the measurement ceiling.

- **Must hit** (both, or the result is reported as a miss):
  - gold macro-F1 ≥ strongest lexical baseline **+ 15 points**
  - gold macro-F1 ≥ zero-shot Qwen3-1.7B **+ 10 points**
- **Stretch:** within **3 points** of prompted `openai/gpt-oss-20b`.

### Gate (Milestone 1 exit)

If the prompted reference judge already scores ~0.94+ macro-F1 on gold, the
"match the bigger model" framing has no headroom and the plan gets revisited
before any training compute is spent.

## Honest-wording constraints

- **2026-08-22:** Groq decommissioned the original teacher
  (`llama-3.3-70b-versatile`) and reference judge (`llama-3.1-8b-instant`)
  mid-project; `gpt-oss-120b` / `gpt-oss-20b` are the replacements.
- `gpt-oss-20b` is MoE — ~20B total, ~3.6B active parameters. Any
  "N× smaller" headline is therefore banned: the writeup states both
  parameter counts and lets the reader divide.
- Purpose is a portfolio proof with one honestly-measured win, not a shipped
  user-facing feature.

## Milestones

0. Tier 1 safety items — **done** (`df54351`).
1. Measurement spine — capture (`d1a2632`), blind labelling tool (`746e15c`),
   this pre-registration, baseline predictions, gold labels, baseline scores.
   Ends at the gate above.
2. Training data (public + distilled + swaps).
3. QLoRA training + ablations (this directory gets its own requirements file;
   `backend/requirements.txt` stays untouched).
4. Wire into eval harness + writeup.

## Addendum, 2026-11-19: the zero-shot protocol

Written before the zero-shot run and before any gold label exists. The
baselines section above names zero-shot Qwen3-1.7B but never said *how* it is
run, and each choice below moves its score, so the choices are fixed here
rather than picked after seeing what they produce. Implemented in
`finetune/01_zero_shot_baseline.ipynb`.

- **Same prompt.** System and user prompt are imported from
  `backend/evals/judge.py` (`_SYSTEM`, `build_prompt`), not copied. The
  zero-shot model is asked exactly what gpt-oss-20b/120b were asked.
- **Thinking off** (`enable_thinking=False`). The fine-tuned judge answers in
  one token with no reasoning, so its baseline has to be measured in that
  format. A reasoning baseline would make the +10 bar compare two different
  tasks.
- **4-bit NF4, the same quantisation the adapter runs on.** The difference
  between baseline and fine-tune is then the adapter and nothing else. Loading
  the baseline in bf16 would quietly credit (or blame) the adapter for the
  quantisation.
- **Score and verdict.** One forward pass. The probability of the first token
  of each answer is summed over case and leading-space variants, giving
  `score = P(supported) / (P(supported) + P(unsupported))`; the verdict is
  whichever is larger. The probability mass on the two answers together is
  recorded as a diagnostic: low mass means the model wanted to say something
  else, and the verdict is then a forced choice.
- **Scored on the verdict, like the prompted judges.** The oracle threshold
  stays reserved for judges with no decision of their own (the lexical
  variants). The fine-tuned judge is held to the same fixed rule, so the
  comparison is symmetric. The oracle-threshold figure for zero-shot is
  computable from the committed scores and is reported next to the headline
  as a sensitivity check, never as the bar.
