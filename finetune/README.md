# Citation-support judge — fine-tune

A QLoRA fine-tune of Qwen3-1.7B that reads one answer sentence and the evidence
chunk it cites, and emits a single token: `supported` or `unsupported`.

It exists to replace `semantic_guards.token_overlap()` in the eval harness,
which `ROADMAP.md` argues is untrustworthy for the case that actually matters —
same run, same topic, shared vocabulary, *different evidence*. It is
**eval-time only** and never wired into serving: the production box is a 512MB
Render instance and `torch` is deliberately absent from
`backend/requirements.txt`.

Read [`PREREGISTRATION.md`](PREREGISTRATION.md) first. It fixes the success bar
before training, and it was committed before any of this ran.

## The split, and why it is where it is

| Half | Lives in | Why |
| --- | --- | --- |
| **Contract** — label mapping, gold gate, de-duplication, class balance | `backend/evals/trainset.py`, `backend/evals/contamination.py` | Stdlib-only, so CI runs it. These are the decisions that change what the model learns, and they are the ones worth having tests on. |
| **Plumbing** — downloads, GPU, notebook | here | Needs `datasets` and a GPU; cannot run in the backend suite. |

`build_dataset.py` imports the first from the second rather than vendoring a
copy, so the CI-tested rules and the Colab build cannot drift.

## Running it

```bash
# 1. Export what "gold" means (hashes only — no evidence, no labels).
cd backend && python -m evals.label --manifest --limit 200

# 2. Build the public slice. Refuses to run without the manifest.
cd ../finetune && pip install -r requirements.txt
python build_dataset.py --per-source 4000
```

The build writes `data/public.jsonl` (gitignored — CC BY-SA, see
[`LICENCES.md`](LICENCES.md)) and `data/public.report.json` (committed: row
counts, per-source drops, class balance).

### Zero-shot baseline

The untrained Qwen3-1.7B is the fourth baseline, and it needs a GPU, so it runs in Colab rather than here. It must run before any gold label exists; the prediction timestamps are the audit trail. The protocol is fixed in the zero-shot addendum in [`PREREGISTRATION.md`](PREREGISTRATION.md).

1. Open `01_zero_shot_baseline.ipynb` in Colab on a T4 runtime.
2. Upload your local `backend/evals/data/pairs.jsonl` when asked, then run all.
3. It downloads the predictions (`qwen-qwen3-1.7b-zero-shot.jsonl`) and a run report (`qwen3-1.7b-zero-shot.json`). Put the first in `backend/evals/results/judge/` and the second in `finetune/reports/`, and commit both before labelling.

## The contamination rule

`assert_clean` **raises** rather than filtering. A training pair is
disqualified if it shares `pair_id`, `query_id`, normalised evidence text, or a
near-duplicate sentence (content-word Jaccard ≥ 0.70) with the gold set.

`query_id` is the tier that does the work. Pairs from one pipeline run share
retrieved evidence chunks, so a fresh pair id proves nothing. Measured on the
first capture: of 111 candidates outside the gold-200, **97 reuse a gold
evidence chunk, 81 reuse a gold sentence verbatim, and 0 survive the full
rule.** The in-domain slice therefore has to come from new capture runs on
queries disjoint from the 15 already used.

## Status

- [x] Milestone 0 — Tier 1 safety items
- [ ] Milestone 1 — measurement spine. Tooling and the lexical and both prompted
      baselines done at 200/200; the zero-shot Qwen3-1.7B baseline has a notebook
      (`01_zero_shot_baseline.ipynb`) awaiting its Colab run, which must happen
      before labelling. **Blocked on ~200 hand labels**, which cannot be delegated to a
      model without changing what the gold set means. Ends at the gate in
      `PREREGISTRATION.md`.
- [ ] Milestone 2 — training data. Public slice built; in-domain slice waits on
      the gate, since it costs Tavily credits that are wasted if the gate says
      the framing has no headroom.
- [ ] Milestone 3 — QLoRA + ablations
- [ ] Milestone 4 — wire into the harness + writeup
