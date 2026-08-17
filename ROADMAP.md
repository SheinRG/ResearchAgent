# goon.ai — Roadmap

Written 2026-08-05, from a full audit of the codebase (product, platform, OSS,
and retrieval/eval). This is the "why" behind the next 90 days, so the reasoning
survives even when the conversation that produced it doesn't.

> Detailed notes on unfixed hardening work live in `SECURITY-NOTES.local.md`,
> which is deliberately **not** committed — this repository is public.

---

## 1. Goals and constraints

Three goals, pursued at once:

- **JOB** — the project is proof of skill for a senior / AI-engineering role.
- **USERS** — real people using it, not just demo clicks.
- **OSS** — stars, and ideally contributors.

Explicitly **not** monetising for now.

Constraints, which drive every decision below:

| Constraint | Consequence |
|---|---|
| **$0 budget, free tiers only** | Anything with a per-user marginal cost is dead on arrival. Cost control is the load-bearing feature, not a nice-to-have. |
| **10–20 hrs/week, solo** | ~130–260 hrs across 13 weeks. The plan below is ~112 hrs — deliberately at the bottom of the range, because solo estimates slip. |
| **~4.5s latency target** | Already bought by removing the planner/reflector loop (~25s). Not to be spent again. |

**The standing tension:** "real users" and "$0" pull against each other. Every
user costs Groq tokens and Tavily credits. This is why the demo is recorded
rather than live, and why the spend ceiling is Tier 0.

## 2. Strategy

> **"The research agent that shows its work."**

We do not out-position Perplexica (33k stars, fully local via Ollama + SearXNG,
no API keys). That is a different and larger problem. The honest lane is
**verifiability** — the Trace tab, citation integrity, scored evals, per-turn
cost accounting. Almost no Perplexity clone has any of it.

This lane is chosen because **credibility is the only asset that serves all
three goals with the same work.** Hiring managers probe rigour; HN and
r/LocalLLaMA reward rigour over polish; users trust a tool that shows sources.

**The prioritisation rule: work serving 2+ goals gets built; work serving 1 waits.**

## 3. Two decisions worth not re-litigating

### 3.1 The demo is live, behind a fuse

*Revised 2026-08-12. This section previously read "the demo is recorded, not
live." The objection that drove that call was arithmetic, and it was correct —
what changed is that the arithmetic now has a guard behind it rather than a
prohibition in front of it. The original reasoning is kept below because it is
still what constrains the design.*

The biggest product gap was real: `frontend/src/app/page.js` redirected to
`/login` before anything rendered, so a "watch it work" product made you
register before you could watch it work.

The naive fix — unlimited anonymous queries — is still a trap on $0. One run
bills up to 4 Tavily credits against a **1,000/month** free tier, so ~250 runs
is the entire month. A generous per-visitor allowance is the version that dies
fastest: at 30 free queries, **nine visitors** spend the month. And 30 is
self-defeating on its own terms — nobody runs 30 research queries on a
stranger's demo, so it is a wall nobody reaches, and a signup gate nobody
reaches converts nobody.

> A signup gate says you made a product call someone disagrees with.
> A dead demo says you can't run a service.

**Decision:** anonymous visitors get **3 real queries**, and three independent
limits keep that survivable. Each does a different job, which is why none of
them can be dropped:

| Limit | Job | Default |
|---|---|---|
| Per visitor (cookie) | *Conversion.* Where someone hits the wall while still interested. | 3 / 24h |
| Per IP | Blunts cookie-clearing. Loose, because offices share an address. | 10 / 24h |
| Global daily + monthly credits | *Cost.* Not bypassable by minting identities. | 120/day, 600/month |
| Anonymous sub-pool | Stops a demo spike starving signed-in users. | 40/day |

**The answer cache is what makes 600/month enough.** Logged-out traffic is
highly correlated — everyone tries the same few questions — and a cache hit
costs zero credits and zero tokens. The daily allowance is really a
*cache-warming* budget; the cache does the serving. This is also why a cache
hit spends a visitor's free query but never touches the credit pool: the
per-visitor number is a product decision, the pool is a cost one.

On breach it **degrades to cache-only with an honest banner, never a 503** —
every answer already cached still works. The guard reads Postgres
(`research_queries.search_credits`, added for this) as ground truth with Redis
only a 60s cache in front, and **denies live research when it cannot establish
spend at all**. An unreadable budget is not an unspent one.

The recorded fixtures are still worth building, for the other two things they
buy: **the corpus for PR-level evals and the source footage for the README
GIF** — plus a third now, pre-warming the cache before a launch so the first
visitors hit it instead of the credit pool.

Known limits, verified in code:
- `ResearchQuery` has **no images column**, so a restored/shared session has an
  empty Images tab. Accept it — images aren't the differentiator.
- ~~`sub_queries` is stored but is not returned by `GET /api/sessions/{id}`.~~
  **Overstated, and now done.** The decomposition was never actually lost: it is
  also written inside `trace`, which *is* returned, and `TraceView` renders it.
  The real gap was narrower — turns stored before the Trace tab existed have
  `trace == {}`, and for those the column was the only surviving copy. It is now
  returned at the top level of each turn as well.
- `trace` **is** persisted and **is** returned. The actual differentiator
  survives the shared-session path intact.

### 3.2 Sessions and files get opposite answers

These were the same "public by unguessable UUID" design. They are not the same
risk and must not share a fate.

| Endpoint | Decision | Reasoning |
|---|---|---|
| `GET /api/sessions/{id}` | **Keep public**, but make it explicit and revocable | The content is a research answer on a public topic. It's a real growth loop *and* the demo mechanism. Sessions private by default; an explicit Share action mints a separate revocable token. |
| `GET /api/files/{id}` | **Gated** — done, see §4 | The content is raw uploaded bytes: resumes, contracts, notes. No growth-loop argument applies. |

"We had an unguessable-UUID capability and split it into an explicit revocable
grant" is a senior answer. "Every session is public if you know the URL" is not.

## 4. Done

- **Search retries + honest credit accounting.** Serper and Tavily returned
  `[]` on the first timeout or 5xx, which the pipeline cannot tell apart from
  "the web has nothing on this" — one blip produced a confident answer with no
  sources and no visible error. Both now retry with exponential backoff under a
  wall-clock deadline (`services/retry.py`, shared with the Groq client, which
  had the only copy of this logic). The subtle half: credit counting moved from
  `researcher.py` into the provider clients and now fires **per HTTP attempt**.
  Counted per *call*, a flapping provider would have billed three lookups while
  the guard counted one — a leak with no alarm on it. Image search is
  deliberately not retried: it is decorative, and a retry spends a credit an
  answer could have used.
- **Per-IP registration limit** (Tier 1). `/register` was unbounded. Not a
  budget-bypass fix — the ceilings in `services/budget.py` are global and cannot
  be bypassed by minting accounts — but a signup loop could burn the shared
  daily allowance and drop every real visitor into degraded mode. 5/IP/rolling
  day, counting *attempts* rather than successes (a script hammering existing
  addresses never creates a row), checked before any DB work or bcrypt hash, and
  fails closed to a process-local counter when Redis is down.
- **`bc78d0a`** — Redis reconnect backoff. `_redis_available` was a one-way
  latch: the first failed connect disabled the cache for the life of the
  process, silently taking the answer cache, the Redis rate-limit path and the
  health counters with it. Now backs off 5s → 300s and recovers on its own.
- **`311a4d6`** — Uploaded files can no longer be served as renderable HTML on
  the API origin. MIME is derived from the validated extension; `/api/files/{id}`
  requires auth + ownership; served as `attachment` with `nosniff`; the viewer
  fetches with an `Authorization` header and builds its own blob URL.
  *Behaviour change:* a shared session no longer serves its attached documents
  to the recipient. That's correct — the answer is shareable, the file isn't.
- **Anonymous demo mode + the spend ceiling** — the landing page no longer
  redirects to `/login`. Logged-out visitors get 3 real queries, guarded by
  `services/budget.py` (Postgres-backed, fails closed, degrades to cache-only)
  and `services/anonymous.py` (per-visitor and per-IP allowances, both
  fail-closed). Tier 0 item 2 is subsumed by this. See §3.1 for the sizing and
  why 3 rather than 30.
- **`LICENSE` (MIT)** — the repo was legally all-rights-reserved while the README
  advertised MIT. Every OSS item was blocked behind this.
- **The delete button deletes.** `SessionHeader.js` toasted "Session deleted" and
  navigated away without ever calling the API, so the thread came back on the
  next sidebar fetch. It now calls `DELETE /api/sessions/{id}`, waits for the
  server, refreshes the sidebar, and only then navigates. It is hidden unless the
  visitor is signed in and the thread has a stored id — the endpoint requires
  auth plus ownership, so offering it otherwise only ever produced a 404.
- **Upload size guard reads in chunks.** `await file.read()` ran *before* the
  size check, so a 500MB POST was fully resident before it could be rejected.
  Peak memory per upload is now bounded at 5MB + 64KB. Covered by
  `tests/test_upload_limits.py`, which asserts on bytes read, not just the 413.
- **`sub_queries` returned from `GET /api/sessions/{id}`** — see the correction
  in §3.1; the gap was smaller than this document claimed.

## 5. Tier 0 — before anything goes public (~1 hr remaining)

None of this makes the project better. All of it stops the project from being
broken, illegal, or exploitable.

| # | Item | Hrs | Goals | Why |
|---|---|---|---|---|
| 1 | ~~Add a real `LICENSE` file (MIT)~~ — **done**, see §4 | — | — | The repo was legally all-rights-reserved while the README advertised MIT. |
| 2 | ~~Global daily spend ceiling~~ — **done**, see §4 | — | — | Built as `services/budget.py` when anonymous demo mode needed it. The design notes below still describe what shipped. |
| 3 | ~~Harden upload limits~~ — **done**, see §4 | — | — | Memory-footprint hardening on a small instance. |
| 4 | ~~Fix the fake delete button~~ — **done**, see §4 | — | — | A delete that lies is worse than no delete. |
| 5 | Repo description, topics, social preview image | 1 | OSS | The description reads "A Perplexity AI clone" — that sentence costs stars on every link preview before anyone opens the README. **Only remaining Tier 0 item, and it is GitHub-side, not code.** |

### The spend ceiling must not be Redis-only

The obvious design (`INCRBYFLOAT` a daily Redis key) **fails open on two
independent paths**, which is the opposite of what a budget guard is for:

1. Redis is optional and degrades to `None` everywhere — a ceiling reading a
   missing counter would see "$0 spent" and authorise unlimited spend.
2. `render.yaml` sets `maxmemoryPolicy: allkeys-lru`, which evicts *any* key,
   including a no-TTL cost counter, so the budget would silently reset mid-day.

It's a smoke alarm wired to the same fuse as the thing it's watching. Both are
reasons the naive version is not worth building — not descriptions of a shipped
guard.

**Correct design:** `ResearchQuery.cost_usd` already exists and is already
written per turn, so `SELECT SUM(cost_usd) WHERE created_at > today` is ground
truth that cannot be evicted. Redis caches that sum for ~60s; on a `None` from
Redis, fall through to Postgres, **never** to "unspent". If Postgres is also
unreachable, deny new research. On breach, **degrade to cache-only with an
honest banner — never hard-503.** Separately, switch to `volatile-lru`.

**One correction from building it:** dollars turned out to be the *backstop*,
not the ceiling. Groq's free tier rate-limits and recovers within the hour;
a spent Tavily credit is gone until the month rolls over. So the guard gates on
a new `research_queries.search_credits` column — counted, not inferred from
`len(sub_queries)`, because a search served from the Redis cache is billed
nothing — and keeps the dollar ceiling only to catch `GROQ_SYNTH_MODEL` being
pointed somewhere expensive.

*"We degrade instead of falling over" is a materially better interview answer
than "we rate-limit."*

## 6. Tiers 1–3

### Tier 1 — weeks 2–4: make it visible and safe to share (~19 hrs remaining)

| Item | Hrs | Goals |
|---|---|---|
| Record 6–8 demo fixtures (SSE captures) — do this first, three things depend on it | 6 | ALL |
| ~~Anonymous demo mode (replay fixtures, zero cost)~~ — **done**, and live rather than replayed; see §3.1 and §4 | — | — |
| README rewrite: GIF above the fold, live demo link in the first 3 lines, one blunt differentiation paragraph, evals pulled out of the war-story section into a "why this is production-grade" block near the top | 8 | ALL |
| Explicit revocable share tokens for sessions (§3.2) | 5 | USERS, JOB |
| ~~Return `sub_queries` from `GET /api/sessions/{id}`~~ — **done**, see §4 | — | — |
| ~~Retry + backoff on Serper/Tavily~~ — **done**, see §4 | — | — |
| ~~Per-IP registration limit~~ — **done**, see §4 | — | — |
| ~~Tag `v0.1.0`~~ — **done** | — | OSS |

> Note on the per-IP limit: it is **not** needed to stop budget bypass — a
> *global* ceiling is by construction not bypassable by creating accounts. The
> real reason is denial-of-service: one abuser burns the global budget and every
> legitimate visitor hits degraded mode.

### Tier 2 — weeks 5–9: build the actual differentiator (~38 hrs)

| Item | Hrs | Goals |
|---|---|---|
| PR-level fixture eval in CI (replay the graph against recorded fixtures, no API keys, works on forks) | 18 | JOB, OSS |
| Citation precision — see the caveat below | 13 | ALL |
| Wire `time_sensitive` into Tavily retrieval params | 3 | USERS |
| Real confidence + weak-evidence abstention | 5 | JOB, USERS |
| Router/auth integration tests (`TestClient`) | 12 | JOB |

**Why the eval work outranks everything else here:** `ci.yml` currently only
runs `--validate`, which lints the YAML. Real evals are Monday-cron only, so a
prompt regression ships and sits undetected for up to a week.

**`time_sensitive` is the cheapest correctness win in the audit.** Triage
already computes the flag, and it *only* ever sets a cache TTL — it never
reaches Tavily's query params or the reranker. "Latest EU AI regulation" gets
byte-identical retrieval to a dictionary definition.

**Confidence today is `0.5 + 0.08 × source_count`** — eight terrible sources
score identically to eight excellent ones. The rerank scores needed to fix this
are already in state and unused. Abstention likewise fires only on *zero*
sources, never *weak* ones.

**Integration tests are 12h, not 6h**, because there is no `conftest.py`, no
`TestClient` usage anywhere, no `aiosqlite`, and no DB service in CI. The
ownership checks in `sessions.py` and `notes.py` are correct **only by
inspection** — including the one added in `311a4d6`.

#### Caveat: do not ship a citation badge backed by token overlap

The plan is to reuse `semantic_guards.token_overlap()` to check a cited chunk
supports its sentence. That is a **lexical** check: it will mark a correctly
paraphrased citation unsupported, and an unrelated chunk sharing vocabulary
supported. A user-facing "citation confidence" badge backed by that is a trust
signal that is itself untrustworthy — and it is exactly what a sharp interviewer
will probe, turning the strongest asset into the weakest moment.

**Split it:**
- **Ship** token overlap as an *eval-time* metric, with its limitations stated
  plainly in the eval README. Defensible precisely *because* you name what it
  can't do.
- **Gate** the user-facing badge behind a real entailment check — a batched 8B
  judge run after the stream completes, off the critical path. If that isn't
  built, ship the Trace surfacing **without** a confidence badge.

### Tier 3 — weeks 10–13: distribute (~30 hrs)

| Item | Hrs | Goals |
|---|---|---|
| CONTRIBUTING.md (repo-specific), 2 issue templates, PR template, 5 real good-first-issues | 6 | OSS |
| One launch: Show HN **or** r/LocalLLaMA — not both the same week | 4 | OSS, USERS |
| Budget real hours for responding to inbound | 10 | OSS, USERS |
| Technical writeup: the polarity guardrail | 6 | JOB, OSS |
| Contradiction detection in the UI (reuse `semantic_guards` polarity) | 8 | ALL |

## 7. GitHub / OSS specifics

**The contributor funnel doesn't exist yet** — no CONTRIBUTING, no templates, no
PR template. But the codebase has unusually good bones for one:

- **`backend/evals/queries.yaml` is a near-perfect first contribution.** "Add 3
  eval queries covering X" needs no architecture knowledge, is validated by CI,
  and directly grows the credibility asset. Almost no clone can offer this,
  because almost no clone has an eval harness.
- **`semantic_guards.py`'s four rails** — "add a 5th guardrail" is
  self-contained and testable.
- **Do not** label the LangGraph node wiring or the cache TTL logic
  good-first-issue. Mislabelling a hard issue burns a contributor's one shot at
  trusting your labels.

**The single highest-value line in CONTRIBUTING** will be: *tests run with
mocked keys — you do not need live API keys to run pytest.* That's true
(`ci.yml` passes `GROQ_API_KEY: test`) and completely unstated. Most people
assume they need three paid API keys and quit before starting. Add the frontend
rebuild gotcha (`docker compose build frontend`) too, and state that the
planner/reflector removal is settled ground so nobody's first PR proposes
reviving it.

**Sequence contributor scaffolding *after* the README work.** Scaffolding with
no traffic is scaffolding nobody climbs.

**Badges worth having:** CI status, the weekly eval run, license (once real).
**Noise:** "PRs welcome", contributor count, tech-stack shield walls, star count.

**The 20 `keep streak alive` commits:** leave the history alone, just stop
making them. Force-pushing rewritten history on a repo with `autoDeploy: true`
risks the deploy pipeline for a cosmetic problem. Same for DEVLOG.md's padded
entries 11–40. If you want to be proactive, one self-aware line in CONTRIBUTING
reads better than a scrub would.

**Realistic ceiling, stated plainly:** 50–250 stars, 0–5 external PRs, a handful
of people who self-host past the demo click. A well-executed Show HN from an
unknown account is a coin flip — 30–150 stars if it catches, a real chance of
zero. That is a good outcome for a solo project. If the honest goal is the job,
judge the 30 days by *"would a senior engineer skimming this for 90 seconds come
away impressed"* — not by stars.

## 8. Kill list

Explicitly not building, with reasons — so these don't get relitigated.

**Deprioritised goal:** Stripe, plan tiers, billing, usage-aggregation
reporting. (The cost accounting makes this cheap *later*, which is exactly why
it can wait.)

**Costs money on $0:** digest emails, followed topics, scheduled research
(needs an email provider and a cron worker — and retention work before you have
users is theatre); a staging environment; enabling the semantic cache in
production (needs a paid embedding key — keep it as the *story*, not the running
feature); multi-provider LLM fallback.

**Bad effort/payoff solo:** multi-tenancy and team workspaces (touches every
router's authorization, zero validated pull); API-as-product with keys and an
SDK (no consumers, and the idiosyncratic SSE contract makes a usable client
library the real hidden cost); full LLM-as-judge with a human-labelled golden
set (do the cheap deterministic scorers first and see if they suffice);
hand-maintained CHANGELOG (use auto-generated release notes from tags).

**Relitigates a settled decision:** HyDE / query rewriting (adds an LLM round
trip; decomposition already exists); verifier or critic agents, reflection
loops, multi-hop agentic retrieval — all re-introduce the ~25s already measured
and removed. Per-claim entailment *inline* — same objection; it belongs in eval.

**Risk exceeds reward:** rewriting git history; circuit breakers (premature at
this QPS — retries already exist for Groq).

**Polish before proof:** folders/collections, session search, source-comparison
UI, page-level PDF citation anchors. Session search in particular is defeated by
its own justification — it only matters at 30+ sessions, which needs retention
that isn't being built yet.

**Adopt when needed, not before:** Alembic. No non-additive schema change is
pending, and the existing hand-written `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
pattern handles additive ones. Adopt it the day a column needs renaming — but
note the debt is growing, not static.

## 9. Standing engineering notes

- **Reuse computed data before adding pipeline stages.** `time_sensitive`, the
  rerank scores, `cost_usd`, and `semantic_guards`' polarity word-set are all
  already computed and under-used. Every recommendation above is either
  already-computed-data reuse or lives in CI — never on the synthesis path.
- **Latency is a hard product constraint.** ~4.5s was bought by deleting a
  planner and a reflect/refine loop. Anything that adds an LLM round trip to
  the critical path needs to justify itself against that.
- **Optional integrations must no-op cleanly.** Sentry, Redis, Langfuse and
  embeddings all follow this. Keep it.
