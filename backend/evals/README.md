# Eval harness

A fixed set of research questions, run through the real pipeline on a schedule
and scored the same way every time, so a change to a prompt or a model shows up
as a number instead of a vibe.

Think of it as a health inspector: it doesn't taste the food and tell you it's
delicious — it checks that the kitchen followed its own process. It measures
**faithfulness and structural integrity**, not truth. An answer can be perfectly
faithful to a source that is wrong, and nothing here will notice. Judging truth
needs ground-truth labels this project doesn't have; judging whether the agent
did what it said it did needs only data the pipeline already produces.

## The four parts

| File | Role |
| --- | --- |
| `queries.yaml` | The questions. Same ones every run, so runs are comparable. |
| `scorers.py` | The checklist. Pure functions — no network, no model, no clock. |
| `run_eval.py` | The runner. Executes the set, aggregates, compares to the baseline. |
| `baseline.json` | Last accepted numbers. Regressions are measured against this. |
| `capture.py` | Side errand: records citation-judge candidates while a run happens. |
| `label.py` | Hand-labels those candidates into the gold set. |
| `judge.py` | Baseline judges for the citation task, and the scoring that ranks them. |

The split between the checklist and the runner is the important one: the
scorers are tested on every pull request (`tests/test_scorers.py`,
`tests/test_eval_runner.py`) with no API keys, while the expensive live run
happens weekly.

## Running it

```bash
cd backend

python -m evals.run_eval --validate          # parse the query set, no API calls
python -m evals.run_eval                     # full run (needs real API keys)
python -m evals.run_eval --only prose-rag    # one or a few queries
python -m evals.run_eval --update-baseline   # accept these numbers as the new bar
python -m evals.run_eval --capture evals/data/pairs.jsonl   # also record judge candidates
```

Needs `GROQ_API_KEY` and `TAVILY_API_KEY` (plus `SERPER_API_KEY` for images and
the fallback search path). Exit code is 0 when every hard invariant held and no
tracked metric regressed, 1 otherwise.

In CI it runs **weekly** (Mondays 07:00 UTC) via `.github/workflows/eval.yml`,
plus a **Run workflow** button for on-demand runs. Not on pull requests: GitHub
withholds secrets from fork PRs, so a per-PR eval would fail for anyone outside
the repo — and it costs real Tavily credits.

## Capturing citation-judge candidates

`--capture` writes a JSONL record for every (answer sentence, cited source)
pair the run produced. It exists because **the evidence behind a citation is
never persisted**: the database keeps a search snippet, the trace keeps a
180-char preview, and the full chunk text the model actually read lives only in
graph state while the run is in flight. A run is the only chance to record it.

Each record pairs one sentence with the exact text that sat behind its `[n]`
marker in the synthesis prompt — recovered through the same
`citations.select_evidence()` the synthesizer uses, not a re-implementation of
its rules, so the two cannot drift.

Two kinds of record come out, and **neither is labelled**:

| `pair_type` | What it is |
| --- | --- |
| `cited` | The sentence and the source it actually cited. |
| `swapped` | The same sentence against a *different* source from the same run. |

`label` is always `null`. A cited pair is not automatically supported — models
citing things their source does not say is the entire problem being measured —
and a swapped pair is not automatically unsupported, because another source
from the same run often supports the sentence too. Labelling either
automatically would assume the answer to the question the judge exists to ask.

Swapped pairs are there because a set of only-cited pairs is overwhelmingly
supported, and a judge measured on those alone has no measurable precision.
They cost nothing — no extra search credits, no extra tokens — and they are
exactly the case a lexical overlap check gets wrong: same run, same topic,
shared vocabulary, different evidence.

Appending is idempotent by `pair_id` and the swap choice is seeded per
sentence, so re-running the same queries reproduces the dataset rather than
duplicating or perturbing it. The in-domain set is meant to accumulate across
many runs.

## Building the gold set

`label.py` turns captured candidates into hand-labelled ground truth. The gold
set is the only artifact licensed to produce a quotable number — everything
else (the teacher's labels, the token-overlap baseline, the trained judge) is
measured *against* it — so three rules protect it.

```bash
python -m evals.label --pairs evals/data/pairs.jsonl   # label; resumable, quit any time
python -m evals.label --stats                          # progress, balance, breakdown
python -m evals.label --recheck 40                     # self-agreement pass
```

**The labeller is blind.** You see the sentence and the evidence — exactly what
the judge sees. You do not see whether capture called the pair `cited` or
`swapped`, because that is a hint about the answer and would anchor you toward
it. You also do not see the query, source title, or domain: labelling with
information the model cannot access produces a standard no model could meet,
which then reads as model failure rather than as a broken measurement.

**Presentation is shuffled**, seeded so a resumed session keeps the same order.
Capture emits a sentence's cited and swapped pairs adjacently, and labelling
them back to back means judging the second relative to the first.

**Your own consistency is measured.** `--recheck N` re-presents pairs you have
already labelled, blind, and reports how often you agree with yourself. That
number is the ceiling: a judge scoring above your self-agreement is fitting
label noise, not learning the task. Quote it next to any model score — a result
without it is a measurement with no error bar.

`unclear` is a real answer, not a cop-out. Genuinely ambiguous evidence should
be excluded from scoring rather than forced into a binary that is then treated
as truth. It is reported separately, so a high unclear rate shows up as a data
problem instead of hiding as label noise.

`gold.jsonl` holds only pair ids and labels — no sentences, no evidence, no
scraped text — so it is committable even though the candidates it refers to are
not. The `--stats` breakdown by capture type is the check on the data plan: if
swapped pairs come back mostly *supported*, the free negatives are not
negatives and the plan needs revisiting.

## Baselines for the citation judge

`judge.py` answers the question the fine-tune exists to ask: *is a trained
judge actually better than what we already have?* It produces predictions for
the baselines named in `finetune/PREREGISTRATION.md` and scores anything in
`results/judge/` against the gold set.

```bash
python -m evals.judge --lexical                          # both overlap scores; no network
python -m evals.judge --predict reference --limit 200    # prompted gpt-oss-20b
python -m evals.judge --predict teacher --limit 200      # prompted gpt-oss-120b
python -m evals.judge --predict reference --limit 5      # smoke run
python -m evals.judge --score                            # everything vs gold
```

**Use the same `--limit` here as in `label.py`.** Both walk the candidates in
the same seeded order, so `--limit 200` judges exactly the 200 pairs the
labeller will label. Judging in file order instead — which is what happened
first — spent 79 of 225 predictions on pairs that would never be labelled while
leaving 54 labelled pairs unjudged, which on a daily token cap is a day lost.

**The binding constraint is tokens per _day_, not per minute.** The free tier
allows 200k per model per day and a pair costs ~890, so roughly **220 pairs per
model per day** — a full 311-pair run cannot finish in one sitting. The
per-minute pacing exists only to smooth bursts; it was never what stopped a
run. Hitting the daily cap is not an error: the run reports where it stopped
and saves everything judged so far, and re-running the same command tomorrow
continues from there.

**A judge sees exactly what the labeller sees** — the sentence and the
evidence, and nothing else. No query, no source title, no pair type. A judge
given more context than the human had is answering an easier question, and its
score stops being comparable to gold.

**The lexical baselines are deliberately flattered.** Two variants are
computed: Jaccard (`token_overlap`, the incumbent as written) and sentence-side
coverage (`|s∩e| / |s|`, which doesn't punish long evidence). Neither gets a
fixed cutoff — each is scored at the *oracle threshold*, the one maximising its
own gold macro-F1, chosen after seeing the labels. That is the best case the
incumbent could possibly have, and the pre-registered bar is measured against
the stronger of the two. Beating a baseline tuned in its own favour is the
conservative claim; beating one with a threshold picked in advance would not be.

**macro-F1, not accuracy.** The candidate set is heavily supported-leaning, so
accuracy would hand "always say supported" ~0.8 and flatter every judge equally.
Macro-F1 puts that strategy below 0.5.

Predictions are cached per pair and appended immediately, so an interrupted run
resumes without re-paying for what it finished, and a rerun corrects in place
rather than duplicating. Groq calls are paced under a client-side per-minute
token budget — the free tier's cap is the real ceiling on this, not model speed.

Unlike the candidates they refer to, prediction files **are committed**: they
carry no scraped text, and timestamps earlier than the gold labels are the
evidence that predictions were made without the answers in hand.

### Checking the data plan before labelling

```bash
python -m evals.judge --diagnose
```

`--diagnose` reports each judge's support rate for `cited` pairs against
`swapped` ones. It needs no gold labels, which makes it the one check worth
running *before* committing an hour to labelling:

- **Swapped pairs mostly supported** would mean the free negatives are not
  negatives, and the data plan needs revisiting.
- **A gap near zero** would mean the pairs carry no signal to learn from.

On the first run both prompted judges cleared it comfortably — swapped pairs
judged supported 0–3% of the time against 42% for cited, a gap of roughly 40
points — so the same-run chunk swaps really are negatives.

The other half of that result is worth sitting with: **only 42% of pairs the
synthesizer actually cited were judged supported.** If the gold labels agree,
that is the project's premise stated as a number rather than a suspicion. It is
judge opinion until then.

## What it measures

**Hard invariants** — these fail the build regardless of history:

- no citation marker pointing at a source that was never retrieved
- a non-empty answer
- no canned fallback message (unless the query sets `allow_fallback`)
- a research answer has at least one source
- every `must_mention` term present

**Tracked metrics** — compared to the baseline, failing only past a tolerance:

| Metric | What a drop means |
| --- | --- |
| Citation coverage | Claims are appearing without sources behind them. |
| Source utilization | Retrieval and reranking are paying for sources the answer ignores. |
| Format compliance | The synthesizer stopped honouring triage's format decision. |
| Triage format match | Triage's format choices drifted from what the set expects. |

Retrieval health (sources found, chunks after rerank, top rerank score) and
cost/latency are recorded but not gated — they're there to tell you *why* a
quality metric moved, and to catch a search provider degrading before answers
visibly break.

## Two things to know

**It goes through the graph, not HTTP.** The answer cache lives in the endpoint,
in front of the graph, so an HTTP-based eval would replay cached answers and
score the cache. Going graph-direct always measures the pipeline.

**The search cache still applies** — it's inside the researcher node, one hour
TTL. A re-run within the hour measures synthesis against identical retrieval,
which is cheaper and less noisy but is *not* a test of retrieval. Flush Redis
first if that's what you want. CI has no Redis at all, so scheduled runs always
do real retrieval.

## Editing the query set

Adding questions is cheap and encouraged — twenty catches breakage, not a 2%
quality change, and a wider set is worth more than repeat runs of the same
question. Changing or removing an existing `id` breaks comparability with past
runs for that entry, so do it deliberately.

The adversarial entries (`allow_fallback: true`) are the ones most worth
keeping: they ask about things that don't exist, and the correct behaviour is
admitting it rather than inventing a citation.

## Cost per run

Roughly **$0.06 of Groq** and **40–80 Tavily credits** for twenty queries,
in about 2–4 minutes at concurrency 4. Weekly is ~240 credits/month against the
1,000 free tier, leaving room for manual runs.

Concurrency is deliberately modest: the pipeline makes three or more model calls
per query, and a run that spends its time collecting 429s measures nothing.
