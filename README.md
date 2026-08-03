# 🔬 goon.ai — AI Research Agent

> An autonomous research agent that decomposes a question, searches the web, reads and ranks sources, and streams back a comprehensive answer with verifiable `[1]`-style citations — a Perplexity-style experience built on an agentic LangGraph pipeline.

## ✨ Features

- **🤖 Agentic pipeline** — triage → parallel search & read → neural re-rank → synthesize a cited answer, orchestrated as a LangGraph state machine
- **🚦 Smart triage/router** — classifies *chat* vs. *research* so a casual "hi" gets an instant conversational reply instead of firing the full (and costly) research run
- **🧠 Two-tier LLM strategy** — a fast model for routing/decomposition, a stronger model for the final synthesized answer
- **🔍 Search + read in one call** — Tavily fetches results *and* page content per sub-query (primary path); Serper + Trafilatura is the automatic fallback, and Serper powers the Images tab
- **⚡ Neural re-ranking** — FlashRank (CPU-only) ranks chunks by relevance before synthesis
- **📝 Trustworthy citations** — a single canonical, relevance-ordered source list drives the prompt, the `[n]` markers, and the UI, so every citation points at exactly the source the model read
- **🌊 Real-time streaming** — SSE token streaming for live answer generation
- **🔎 Trace tab** — every answer ships with a receipt: what was searched, what was ranked, what actually reached the model, and what got cited
- **💰 Answer cache** — a repeat question replays the whole answer (sources, images, trace) for **$0 and zero model calls**; triage labels each answer time-sensitive or evergreen so a stock price and a definition don't share a TTL
- **🧲 Semantic cache** *(optional, off by default)* — on an exact miss, embeds the question and reuses an answer to a differently-worded version of it, behind four safety rails so *"best vector DB"* never gets served *"worst vector DB"*
- **🔐 Auth** — email/password (bcrypt) + Google OAuth (fail-closed token validation), stateless JWT, per-user rate limiting
- **💾 Persistence** — PostgreSQL for sessions/history, Redis (optional) for search, scrape, and answer caching
- **📊 Observability & evals** — Langfuse tracing per pipeline stage, per-turn token/cost accounting, and a scored eval set that runs weekly in CI
- **🚀 Production-ready** — Dockerized services, GitHub Actions CI, pytest suite, deep health checks, and optional Sentry error tracking

## 🏗️ Architecture

```
User Query
  → Answer cache              exact key (query + history + user + model)
      ├─ hit ──────────────→ replay the full SSE sequence · $0 · no model calls
      └─ miss ↓
  → Semantic cache (opt-in)   embed the question, match one that MEANS the same
      ├─ hit (guards pass) ─→ replay + disclose which question was reused
      └─ miss ↓
  → Router (fast LLM)         triage: casual chat vs. research, + time-sensitivity
      ├─ chat ──→ Conversational (fast LLM) → instant reply
      └─ research ↓
  → Researcher (parallel)     decompose into 2–4 sub-queries
                              → Tavily search+read  (Serper + Trafilatura fallback)
  → Re-ranker (FlashRank)     rank chunks; build canonical source list
  → Synthesizer (strong LLM)  stream a cited Markdown answer + follow-ups
                              → store in cache (TTL by time-sensitivity)
```

> **Why no planner/reflector node?** Earlier versions ran a 5-node graph with a
> plan step and a reflect-and-refine loop. Profiling showed they added ~25s of
> latency for marginal answer-quality gains, so the graph was collapsed to the
> lean triage → research → synthesize path above (see the journey below).

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 16, React 19, Vanilla CSS, Motion, Zustand |
| **Backend** | Python 3.12, FastAPI, LangGraph |
| **LLM** | Groq Cloud — `llama-3.1-8b-instant` (route/decompose) + `llama-3.3-70b-versatile` (synthesis) |
| **Search + Read** | Tavily (primary) · Serper (images + fallback) |
| **Extraction** | Trafilatura (fallback scrape path) |
| **Re-ranking** | FlashRank (CPU-only) — `TinyBERT-L-2` for small instances, `MiniLM-L-12` for quality |
| **Embeddings** | Jina (or OpenAI) — **only** for semantic cache matching, never for retrieval; optional |
| **Database** | PostgreSQL 16 |
| **Cache** | Redis 7 (optional) — search, scrape, and full answer replay |
| **Infrastructure** | Docker Compose · Render (backend) + Vercel (frontend) |
| **Observability** | GitHub Actions CI · pytest · Langfuse tracing · Sentry · weekly eval run |

## 🧗 Engineering Journey — Challenges & Solutions

This started as a local-only prototype and evolved into a deployed, multi-user
product over ~3 weeks. The hardest problems weren't writing features — they were
latency, citation trust, concurrency, going public, and then paying for it.

### 1. Latency: a ~35s answer felt broken → got it to ~4.5s

The first working pipeline was *correct* but painfully slow. Three rounds of work
took it from **~35s → ~10s → ~4.5s** (~87% faster):

- **Collapsed the agent graph.** The original 5-node graph (plan → search →
  rerank → synthesize → reflect-and-refine, `max_iterations=2`) spent most of its
  time on a planner and a reflect/refine loop that barely moved answer quality.
  I cut it to a lean **triage → research → synthesize** path and dropped
  iterations to 1. *(`eddf147`, `a34d49b`)*
- **Replaced search-then-scrape with one call.** Serper-search *then*
  Trafilatura-scrape was two round-trips per sub-query and returned empty on
  JS-heavy pages. Switching to **Tavily's single search+read** call — with the
  old Serper+scrape path kept as an automatic fallback — was the biggest single
  win. *(`dc01ae9`, `47eaafb`)*
- **Fixed async bottlenecks.** Added an `asyncio` semaphore to cap concurrent
  scrapes, offloaded CPU-bound chunking to a thread pool so it stops blocking the
  event loop, reused a persistent `httpx` client to avoid repeated TLS
  handshakes, and deduplicated sub-queries to kill redundant pipelines.
  *(`fa5d9d8`, `eb04f62`, `2ebb312`, `42bea3c`)*

### 2. Making citations actually trustworthy

Early answers had `[n]` markers that didn't reliably map to the right source. I
solved it by building **one canonical, relevance-ordered source list** that feeds
the synthesis prompt, the `[n]` markers, *and* the UI from the same array — so a
citation can't drift from the source the model actually read. Then handled the
long tail: citations rendering *inside* table cells, and tooltips getting clipped
or stacking on hover. *(`f9f00b9`, `711ac40`, `aac090e`, `e555ac4`)*

### 3. From "runs on my laptop" to public & multi-user

The prototype assumed a single local user. Going public meant **rewriting auth**
from scratch: stateless JWT, a `useAuth` hook, bcrypt email/password, and Google
OAuth. A security pass made OAuth **fail closed** — verifying `aud`/`iss`/
`email_verified` instead of trusting the token — and added per-user rate
limiting so one user can't exhaust the shared API budget. *(`ec3dab8`,
`2cdebed`, `2042e60`)*

### 4. Concurrency & cold-start race conditions

Under load, the FlashRank model could be initialized by several requests at once.
Added **thread-safe double-checked locking** so the model loads exactly once, and
tuned the scrape pool so a single slow page can't stall the whole batch.
*(`cc918f5`, `fa5d9d8`)*

### 5. Don't run the full pipeline for "hello"

Casual messages were being forced through the entire research pipeline — slow and
wasteful. Added a **router/triage node** that classifies chat vs. research up
front and routes greetings to a fast conversational reply. *(`6353deb`)*

### 6. Deployment hardening

Shipping to Render + Vercel surfaced a fresh class of bugs: the async Postgres
driver (`postgresql+asyncpg://`), CORS for the Vercel origin, the `asyncpg` vs.
internal-URL trap on managed databases, a backend dev-server crash loop, and a
401 that dead-ended the UI instead of recovering. Closed them out alongside CI,
deeper health checks, and custom error pages. *(`e67a9ab`, `8dadcce`,
`69d13a1`, `e67a4be`)*

### 7. Seeing inside the pipeline before optimizing it again

Every earlier optimization was argued from a stopwatch and a hunch. Added
**Langfuse tracing** with a span per stage (triage, search, rerank, synthesis,
cache lookup) plus **per-turn token and cost accounting** surfaced in the `done`
event, so "which stage is slow" and "what did that answer cost" stopped being
guesses. A **Trace tab** puts the same receipt in front of the user: what was
ranked, what actually reached the model, and what got cited. Then a **scored
eval set** (`backend/evals/`) that runs weekly in CI, so a prompt change shows up
as a number rather than a vibe. *(`2fc3496`, `3eb94ef`, `8aa3b0b`)*

### 8. The same question shouldn't cost money twice

Search and scrape were cached, but **both LLM calls still ran on a repeat
question** — the expensive part was the only part not cached. Added an **answer
cache** that stores the complete user-visible output and, on a hit, replays the
whole SSE sequence (sources, images, trace, follow-ups) so the UI is identical
to a live run — just at **$0 with zero model calls**.

The tricky part was staleness. The cache is consulted *before* triage, so at
lookup time nothing knows whether the question is "what is quantum computing" or
"bitcoin price". Fix: triage now also emits a **`time_sensitive` flag**, and that
label is stored *with the answer* — so lifetimes match the content (15 min for
live data, 6 h for evergreen) and the flag defaults to `true` on any triage
failure, meaning a bug can never make a live question look permanent.

### 9. Matching questions by meaning — carefully

An exact-match cache only fires on a word-for-word repeat, and nobody types the
same question twice. A **semantic layer** sits behind it: on a miss, embed the
question and look for a stored one that *means* the same thing.

The danger is the whole feature. Embeddings encode **topic, not direction** —
*"best vector database"* and *"worst vector database"* score ~0.97 against each
other, while the correct answers are opposites. Serving one for the other, with
citations, is worse than being slow. So a high similarity score is necessary but
not sufficient; every candidate must also clear four cheap deterministic rails
(`app/utils/semantic_guards.py`):

| Rail | Catches |
| --- | --- |
| **Numbers** | "Python 3.11 features" vs "3.12" — barely moves a vector |
| **Polarity** | best/worst, pros/cons, is/isn't — flips meaning, not topic |
| **Order** | "is X better than Y" vs "is Y better than X" |
| **Word overlap** | two unrelated questions that happen to score highly |

Plus three structural limits: semantic matching is **first-turn only** (the exact
key covers conversation history because *"how does its pricing work?"* means
different things in different threads), a near-match to a time-sensitive answer
must be **under 2 minutes old**, and the UI **always discloses** which earlier
question was reused. It ships **off by default** — unlike every other optional
integration here, this one can answer a question the user did not literally ask.

## 🚀 Quick Start

### Prerequisites

1. **Docker Desktop** — [Install Docker](https://docs.docker.com/desktop/)
2. **Node.js 20+** — only needed if running the frontend outside Docker
3. **API keys** (all have free tiers):
   - **Groq** — https://console.groq.com (LLM, **required**)
   - **Tavily** — https://tavily.com (primary search+read, recommended)
   - **Serper** — https://serper.dev (images + fallback search, **required**)

### Setup

```bash
# 1. Clone
git clone <your-repo-url>
cd perplexity

# 2. Configure environment
cp backend/.env.example backend/.env     # then fill in GROQ_API_KEY, SERPER_API_KEY, AUTH_SECRET
cp frontend/.env.example frontend/.env

# Generate a strong AUTH_SECRET:
#   python -c "import secrets; print(secrets.token_hex(32))"

# 3. Bring up the whole stack (Postgres + Redis + Backend + Frontend)
docker compose up -d --build
```

> `docker compose` reads variables from a root `.env` file. Set at least
> `GROQ_API_KEY`, `SERPER_API_KEY`, and `AUTH_SECRET` there (or export them)
> before starting.

To run the frontend separately for development:

```bash
cd frontend
npm install
npm run dev
```

### Access

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/health

## 📡 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/auth/register` | — | Register (email/password) |
| `POST` | `/api/auth/login` | — | Log in |
| `POST` | `/api/auth/google` | — | Google OAuth |
| `GET`  | `/api/auth/me` | ✅ | Current user |
| `POST` | `/api/research` | ✅ | Start research (SSE stream) |
| `GET`  | `/api/sessions` | ✅ | List recent sessions |
| `GET`  | `/api/sessions/{id}` | ✅ | Get session details |
| `GET`  | `/api/health` | — | Health check |

### SSE Events

```
event: phase        → {"phase": "planning", "message": "Breaking down..."}
event: sub_queries  → {"queries": ["q1", "q2", "q3"]}
event: sources      → {"sources": [{url, title, domain, favicon, snippet}], "replace": true}
event: images       → {"images": [{url, thumbnail, title, source, domain}]}
event: token        → {"token": "word"}
event: trace        → {what was ranked, what was sent to the model, what was cited}
event: follow_up    → {"suggestions": ["question1", "question2"]}
event: done         → {"session_id": "...", "total_sources": 8, "confidence": 0.89,
                       "latency_ms": 4520, "usage": {...tokens + cost...},
                       "cached": false, "cache_kind": "", "matched_query": "",
                       "similarity": null}
```

A **cache hit replays this same sequence** — sources, images, trace and all — so
a replayed turn is indistinguishable from a live one except for `cached: true`
and an all-zero `usage`. On a *semantic* hit, `cache_kind` is `"semantic"` and
`matched_query` carries the earlier question whose answer was reused; the UI
shows it to the user rather than silently substituting an answer.

## 🔧 Configuration

Backend variables (see `backend/.env.example`):

| Variable | Description | Default / Example |
|----------|-------------|-------------------|
| `GROQ_API_KEY` | Groq Cloud API key (**required**) | `gsk_...` |
| `GROQ_MODEL` | Fast model for planning/reflection | `llama-3.1-8b-instant` |
| `GROQ_SYNTH_MODEL` | Strong model for synthesis | `llama-3.3-70b-versatile` |
| `SERPER_API_KEY` | Serper key — images + fallback search (**required**) | `...` |
| `TAVILY_API_KEY` | Tavily key — primary search+read (recommended) | `tvly-...` |
| `USE_TAVILY_SEARCH` | Use Tavily; set `false` to fall back to Serper+scrape | `true` |
| `AUTH_SECRET` | JWT signing secret (**set a random value**) | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID (optional) | `...apps.googleusercontent.com` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://agent:agent@postgres:5432/research_agent` |
| `REDIS_URL` | Redis connection URL | `redis://redis:6379/0` |
| `RATE_LIMIT_PER_HOUR` | Research queries per user per hour | `30` |
| `ANSWER_CACHE_ENABLED` | Replay a stored answer on a repeat question | `true` |
| `ANSWER_CACHE_TTL` | Lifetime of a **time-sensitive** answer | `900` (15 min) |
| `ANSWER_CACHE_EVERGREEN_TTL` | Lifetime of an **evergreen** answer | `21600` (6 h) |
| `SEMANTIC_CACHE_ENABLED` | Match near-duplicate questions by meaning (**off by default**) | `false` |
| `SEMANTIC_SIMILARITY_THRESHOLD` | Cosine score required to reuse an answer | `0.92` |
| `EMBEDDING_PROVIDER` / `EMBEDDING_API_KEY` | Embeddings for the semantic cache — Groq has none, so this is a separate provider | `jina` / — |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Per-stage tracing (both required, else no-op) | — |
| `SENTRY_DSN` | Error tracking (no-ops when unset) | — |

> **Semantic cache is deliberately opt-in.** It needs *both* the flag and an
> embedding key, and stays inert if either is missing. See
> [`DEPLOYMENT.md`](DEPLOYMENT.md#7-semantic-answer-cache-optional-off-by-default)
> before enabling it in production — and check the threshold against real
> question pairs first; `0.92` is a starting point, not a tuned value.
>
> Cache effectiveness is observable at `/api/health`: `hits`, `misses`,
> `hit_rate`, plus `semantic_hits`, `semantic_rejected` and
> `semantic_rescue_rate` (the share of otherwise-missed questions rescued by
> meaning-based matching).

Frontend variables (see `frontend/.env.example`) — note these are inlined at **build** time:

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Base URL of the backend | `http://localhost:8000` |
| `NEXT_PUBLIC_GOOGLE_CLIENT_ID` | Google OAuth client ID (optional) | — |

## 📁 Project Structure

```
├── backend/
│   ├── app/
│   │   ├── agents/       # LangGraph nodes (router, conversational, researcher, synthesizer) + graph wiring
│   │   ├── services/     # Groq LLM, Tavily + Serper search, Trafilatura scraper, FlashRank,
│   │   │                 #   Redis, answer_cache, embeddings, usage/cost, tracing, auth
│   │   ├── models/       # Pydantic schemas, SQLAlchemy models
│   │   ├── utils/        # Text chunking, citation extraction, semantic cache guards
│   │   └── main.py       # FastAPI app: auth, rate limiting, SSE research endpoint
│   ├── evals/            # Fixed query set + pure scorers + runner (weekly CI run)
│   ├── tests/            # pytest: pipeline, services, cache, guards, auth, rate limit
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── app/          # Next.js pages (home, research, login)
│       ├── components/   # SearchBar, SourceCards, StreamingAnswer, etc.
│       ├── hooks/        # useResearch (SSE), useAuth
│       └── stores/       # Zustand (recent searches)
├── docker-compose.yml    # Postgres + Redis + Backend + Frontend
└── README.md
```

## 📝 License

MIT
