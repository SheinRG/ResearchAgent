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
```

Needs `GROQ_API_KEY` and `TAVILY_API_KEY` (plus `SERPER_API_KEY` for images and
the fallback search path). Exit code is 0 when every hard invariant held and no
tracked metric regressed, 1 otherwise.

In CI it runs **weekly** (Mondays 07:00 UTC) via `.github/workflows/eval.yml`,
plus a **Run workflow** button for on-demand runs. Not on pull requests: GitHub
withholds secrets from fork PRs, so a per-PR eval would fail for anyone outside
the repo — and it costs real Tavily credits.

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
