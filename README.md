# 🔬 goon.ai — AI Research Agent

> An autonomous research agent that decomposes a question, searches the web, reads and ranks sources, and streams back a comprehensive answer with verifiable `[1]`-style citations — a Perplexity-style experience built on an agentic LangGraph pipeline.

[![CI](https://github.com/SheinRG/ResearchAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/SheinRG/ResearchAgent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)

**Contents** · [What it does](#-what-it-does) · [Approach](#-approach) · [Features](#-features) · [Architecture](#-architecture) · [Tech stack](#-tech-stack) · [Getting started](#-getting-started) · [Tests](#-running-the-tests) · [Configuration](#-configuration) · [API](#-api-reference) · [Structure](#-project-structure) · [Deployment](#-deployment) · [Journey](#-engineering-journey--challenges--solutions) · [Contributing](#-contributing)

## 🔎 What it does

You ask a real question. Instead of returning ten blue links (a search engine) or
answering from frozen training data (a bare LLM), goon.ai does live
retrieval-augmented generation:

1. **Triages** the message — a casual "hi" gets an instant reply, not a research run.
2. **Decomposes** a research question into 2–4 focused sub-queries.
3. **Searches and reads** the web for all of them in parallel.
4. **Re-ranks** the retrieved chunks by relevance on CPU.
5. **Synthesizes** a cited Markdown answer, streamed token by token.

You can also **attach your own documents** (`.txt`, `.md`, `.pdf`, `.docx`) — they
become first-class sources, ranked alongside web results and treated as the
primary evidence when the question is about them.

Every answer ships with a **Trace tab**: what was searched, what was ranked, what
actually reached the model, and what got cited.

## 🧭 Approach

The design decisions below are the ones that shaped the codebase. They're worth
reading before the feature list, because most of the code exists to serve them.

**1. Grounded or nothing.** The model is instructed to answer *only* from
retrieved sources and to cite them. The point of the whole pipeline is that a
claim in the answer can be traced to a page someone can open.

**2. One canonical source list.** Early on, `[n]` markers didn't reliably map to
the right source. The fix was structural, not a prompt tweak: a single
relevance-ordered array feeds the synthesis prompt, the `[n]` markers, *and* the
UI. A citation can't drift from the source the model read, because there's only
one list.

**3. Show the work.** "Trust me" is not a feature. The Trace tab, per-stage
Langfuse spans, and per-turn token/cost accounting all exist so the pipeline is
inspectable — by the user and by whoever is debugging it.

**4. Cheapest path that can answer correctly.** Requests fall through a ladder:
exact answer cache → optional semantic cache → triage → conversational reply →
full research. Each rung costs more than the one above it, so the expensive rung
only runs when the cheap ones genuinely can't serve the request.

**5. Measure before optimizing — and delete what doesn't earn its latency.** The
graph used to have a planner node and a reflect-and-refine loop. Profiling showed
they cost ~25s for marginal quality gains, so they were cut. See
[the journey](#1-latency-a-35s-answer-felt-broken--got-it-to-45s).

**6. Fail closed.** Google OAuth verifies `aud`/`iss`/`email_verified` rather than
trusting the token. Triage's `time_sensitive` flag defaults to `true` on any
failure, so a bug can never make a live question look permanently cacheable.
Upload MIME is derived from the validated extension, never from the client header.

**7. Optional means inert, not broken.** Redis, Langfuse, Sentry, Google OAuth,
and the semantic cache are all optional. Each one no-ops cleanly when its keys are
absent — the app runs without any of them.

**8. A free tier has to survive contact with the internet.** Anonymous visitors
get a few real queries before the signup wall, and daily/monthly search-credit
ceilings plus a daily cost budget cap the damage. A burst can't become a habit.

## ✨ Features

- **🤖 Agentic pipeline** — triage → parallel search & read → neural re-rank → synthesize a cited answer, orchestrated as a LangGraph state machine
- **🚦 Smart triage/router** — classifies *chat* vs. *research* so a casual "hi" gets an instant conversational reply instead of firing the full (and costly) research run
- **🧠 Two-tier LLM strategy** — a fast model for routing/decomposition, a stronger model for the final synthesized answer
- **🔍 Search + read in one call** — Tavily fetches results *and* page content per sub-query (primary path); Serper + Trafilatura is the automatic fallback, and Serper powers the Images tab
- **📎 Bring your own documents** — attach `.txt`/`.md`/`.pdf`/`.docx` (5 MB max); text is extracted, chunked, ranked with the web results, and cited like any other source. Triage decides whether the web is even needed.
- **⚡ Neural re-ranking** — FlashRank (CPU-only) ranks chunks by relevance before synthesis
- **📝 Trustworthy citations** — a single canonical, relevance-ordered source list drives the prompt, the `[n]` markers, and the UI, so every citation points at exactly the source the model read
- **🌊 Real-time streaming** — SSE token streaming for live answer generation
- **🔎 Trace tab** — every answer ships with a receipt: what was searched, what was ranked, what actually reached the model, and what got cited
- **💰 Answer cache** — a repeat question replays the whole answer (sources, images, trace) for **$0 and zero model calls**; triage labels each answer time-sensitive or evergreen so a stock price and a definition don't share a TTL
- **🧲 Semantic cache** *(optional, off by default)* — on an exact miss, embeds the question and reuses an answer to a differently-worded version of it, behind four safety rails so *"best vector DB"* never gets served *"worst vector DB"*
- **👤 Anonymous demo mode** — logged-out visitors get real queries before the signup wall, protected by per-visitor quota and global spend ceilings
- **🔐 Auth** — email/password (bcrypt) + Google OAuth (fail-closed token validation), JWT access tokens with refresh-token rotation, per-user rate limiting
- **💾 Persistence** — PostgreSQL for sessions/history/notes/uploads, Redis (optional) for search, scrape, and answer caching
- **📤 Export** — save a session as PDF, Markdown, or DOCX
- **📊 Observability & evals** — Langfuse tracing per pipeline stage, per-turn token/cost accounting, and a scored eval set that runs weekly in CI
- **🚀 Production-ready** — Dockerized services, GitHub Actions CI, 355-test pytest suite, deep health checks, and optional Sentry error tracking

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
                              + uploaded documents → chunks (web skipped if not needed)
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
| **Extraction** | Trafilatura (fallback scrape path) · pypdf + python-docx (uploads) |
| **Re-ranking** | FlashRank (CPU-only) — `TinyBERT-L-2` for small instances, `MiniLM-L-12` for quality |
| **Embeddings** | Jina (or OpenAI) — **only** for semantic cache matching, never for retrieval; optional |
| **Database** | PostgreSQL 16 |
| **Cache** | Redis 7 (optional) — search, scrape, and full answer replay |
| **Infrastructure** | Docker Compose · Render (backend) + Vercel (frontend) |
| **Observability** | GitHub Actions CI · pytest · Langfuse tracing · Sentry · weekly eval run |

## 🚀 Getting Started

### 1. Prerequisites

| Requirement | Why | Notes |
|---|---|---|
| **[Docker Desktop](https://docs.docker.com/desktop/)** | Runs the whole stack | The recommended path — no local Python or Postgres needed |
| **[Node.js 20+](https://nodejs.org/)** | Only for running the frontend outside Docker | |
| **[Python 3.12](https://www.python.org/downloads/)** | Only for running the backend outside Docker | 3.12 specifically — some pins don't build on 3.13+ |
| **Git** | Cloning | |

### 2. Get your API keys

All three have free tiers and none require a card.

| Key | Required? | Get it at | Used for |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes** | [console.groq.com](https://console.groq.com) | Every LLM call (triage, decompose, synthesis) |
| `SERPER_API_KEY` | **Yes** | [serper.dev](https://serper.dev) | Images tab + fallback search |
| `TAVILY_API_KEY` | Recommended | [app.tavily.com](https://app.tavily.com) | Primary search+read (1,000 credits/mo). Without it, set `USE_TAVILY_SEARCH=false` to use the Serper+scrape path. |

### 3. Clone and configure

```bash
git clone https://github.com/SheinRG/ResearchAgent.git
cd AI-ResearchAgent
```

Docker Compose reads a **root `.env` file** — that single file configures every
service. Copy the example and fill it in:

```bash
cp .env.example .env          # macOS / Linux / Git Bash
```

```powershell
Copy-Item .env.example .env   # Windows PowerShell
```

Now open `.env` and set at minimum:

```ini
GROQ_API_KEY=gsk_your_key_here
SERPER_API_KEY=your_serper_key_here
TAVILY_API_KEY=tvly_your_key_here     # optional but recommended
AUTH_SECRET=<paste a random 64-char hex string — see below>
```

Generate a strong `AUTH_SECRET`:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

> **Which `.env` file do I edit?** There are three, and they serve different runs:
> - **`.env`** (root) — what `docker compose` reads. **This is the only one you need for the Docker path.**
> - **`backend/.env`** — only read when you run the backend *natively* (`uvicorn` outside Docker).
> - **`frontend/.env`** — only read when you run the frontend *natively* (`npm run dev`).
>
> Copy the per-service examples (`cp backend/.env.example backend/.env`) only if
> you're taking the native path in step 4b.

### 4a. Run with Docker (recommended)

```bash
docker compose up -d --build
```

That brings up four containers: **Postgres**, **Redis**, the **FastAPI backend**,
and the **Next.js frontend**. The first build takes a few minutes (it downloads
the FlashRank model); later starts are fast.

Watch the logs until the backend reports it's listening:

```bash
docker compose logs -f backend
```

To stop everything (data survives in named volumes):

```bash
docker compose down
```

To stop **and wipe the database**:

```bash
docker compose down -v
```

### 4b. Run natively (for development)

Useful if you want a debugger attached or faster frontend iteration. You still
need Postgres and Redis — the easiest way is to run just those in Docker:

```bash
docker compose up -d postgres redis
```

**Backend:**

```bash
cd backend
cp .env.example .env                 # then fill in your keys
# Point DATABASE_URL/REDIS_URL at localhost, not the compose hostnames:
#   DATABASE_URL=postgresql+asyncpg://agent:agent@localhost:5432/research_agent
#   REDIS_URL=redis://localhost:6379/0

python -m venv venv
source venv/bin/activate             # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**

```bash
cd frontend
cp .env.example .env                 # NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev
```

### 5. Verify it's working

| Check | URL | Expect |
|---|---|---|
| Frontend | <http://localhost:3000> | The goon.ai home page |
| Health check | <http://localhost:8000/api/health> | `{"status": "healthy", ...}` |
| Interactive API docs | <http://localhost:8000/docs> | FastAPI Swagger UI |

Then ask it something in the UI — try *"what changed in Python 3.13"*. You should
see sources stream in, then the answer token by token, then a Trace tab.

### 6. Troubleshooting

| Symptom | Cause & fix |
|---|---|
| **Frontend edits don't show up** | The frontend is a *baked* production image with no bind mount, and `NEXT_PUBLIC_*` vars are inlined at **build** time. Run `docker compose build frontend && docker compose up -d frontend`. For fast iteration, use the native path (4b) instead. |
| **`GROQ_API_KEY` errors on startup** | The root `.env` isn't being read. Confirm it sits next to `docker-compose.yml` and is named exactly `.env`, then `docker compose up -d --force-recreate`. |
| **Answers work but the Images tab is empty** | `SERPER_API_KEY` is unset — Serper powers images even when Tavily handles search. |
| **Backend restarts in a loop on Windows/WSL2** | Known inotify issue; compose already sets `WATCHFILES_FORCE_POLLING=true` to work around it. If you changed that, put it back. |
| **CORS errors in the browser console** | The frontend origin isn't in `CORS_ORIGINS`. The compose default covers ports 3000/3001 on localhost. |
| **"Connection refused" to Postgres on the native path** | You left the compose hostnames in `backend/.env`. From outside Docker they must be `localhost`, not `postgres`/`redis`. |
| **Port already in use** | Something else holds 3000/8000/5432/6379. Stop it, or remap the port in `docker-compose.yml`. |

## 🧪 Running the tests

The backend suite is **355 tests** and mocks every external service (LLM, search,
database), so it needs no API keys and no running Postgres or Redis:

```bash
cd backend
python -m pytest -q
```

Validate the eval query set (the free half of the eval harness):

```bash
python -m evals.run_eval --validate
```

Build the frontend the way CI does — this catches type, import, and SSR errors
that dev mode hides:

```bash
cd frontend
npm run build
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs all three on
every push and PR to `main`. The **scored** eval run needs real API keys, so it
runs weekly on its own schedule ([`eval.yml`](.github/workflows/eval.yml)).

## 🔧 Configuration

Backend variables (see [`backend/.env.example`](backend/.env.example)):

| Variable | Description | Default / Example |
|----------|-------------|-------------------|
| `GROQ_API_KEY` | Groq Cloud API key (**required**) | `gsk_...` |
| `GROQ_MODEL` | Fast model for triage/decomposition | `llama-3.1-8b-instant` |
| `GROQ_SYNTH_MODEL` | Strong model for synthesis | `llama-3.3-70b-versatile` |
| `SERPER_API_KEY` | Serper key — images + fallback search (**required**) | `...` |
| `TAVILY_API_KEY` | Tavily key — primary search+read (recommended) | `tvly-...` |
| `USE_TAVILY_SEARCH` | Use Tavily; set `false` to fall back to Serper+scrape | `true` |
| `RERANKER_MODEL` | FlashRank model — `TinyBERT-L-2` is fastest | `ms-marco-MiniLM-L-12-v2` |
| `MAX_CITED_SOURCES` | Distinct sources surfaced to the model + UI | `8` |
| `MAX_ITERATIONS` | Set >1 to re-enable the reflect-and-refine loop | `1` |
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
| `LANGFUSE_CAPTURE_CONTENT` | `false` = ship timings/tokens/cost only, no question or source text | `true` |
| `SENTRY_DSN` | Error tracking (no-ops when unset) | — |

**Anonymous demo mode & spend ceilings** — what a logged-out visitor may do, and
what protects the free tier. Sized against Tavily's 1,000 credits/month, where one
research run bills up to 4 (see `backend/app/services/budget.py`):

| Variable | Description | Default |
|---|---|---|
| `ANONYMOUS_DEMO_ENABLED` | Let logged-out visitors run real queries | `true` |
| `ANON_FREE_QUERIES` | Queries per visitor before the signup wall | `3` |
| `ANON_DAILY_SEARCH_CREDITS` | Daily search-credit ceiling for anonymous traffic | `40` |
| `DAILY_SEARCH_CREDITS` / `MONTHLY_SEARCH_CREDITS` | Global search-credit ceilings | `120` / `600` |
| `DAILY_COST_BUDGET_USD` | Daily LLM spend ceiling | `0.25` |

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

Frontend variables (see [`frontend/.env.example`](frontend/.env.example)) — note
these are inlined at **build** time, so changing one requires a rebuild:

| Variable | Description | Default |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Base URL of the backend (no trailing slash) | `http://localhost:8000` |
| `NEXT_PUBLIC_GOOGLE_CLIENT_ID` | Google OAuth client ID (button hidden if unset) | — |
| `NEXT_PUBLIC_SENTRY_DSN` | Frontend error tracking (optional) | — |

## 📡 API Reference

Full interactive docs run at <http://localhost:8000/docs>.

**Auth** — `/api/auth`

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/auth/register` | — | Register (email/password) |
| `POST` | `/api/auth/login` | — | Log in |
| `POST` | `/api/auth/google` | — | Google OAuth |
| `POST` | `/api/auth/refresh` | cookie | Rotate the refresh token |
| `POST` | `/api/auth/logout` | cookie | Revoke the refresh token |
| `GET`  | `/api/auth/me` | ✅ | Current user |
| `PATCH`| `/api/auth/profile` | ✅ | Update profile |
| `GET`  | `/api/auth/rate-limit` | ✅ | Hourly usage and remaining quota |

**Research & sessions**

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/research` | optional | Run research (SSE stream). Works anonymously while demo quota lasts. |
| `GET`  | `/api/sessions` | ✅ | List recent session threads |
| `GET`  | `/api/sessions/{id}` | — | Get a session's stored turns (the UUID is the capability) |
| `DELETE`| `/api/sessions/{id}` | ✅ | Delete a thread and all its turns |

**Documents, notes & status**

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/upload` | ✅ | Upload `.txt`/`.md`/`.pdf`/`.docx` (5 MB max), extract text |
| `GET`  | `/api/files/{id}` | ✅ | Serve a stored original (owner only) |
| `GET`  | `/api/notes` | ✅ | List notes |
| `POST` | `/api/notes` | ✅ | Create a note |
| `PATCH`| `/api/notes/{id}` | ✅ | Update a note |
| `DELETE`| `/api/notes/{id}` | ✅ | Delete a note |
| `GET`  | `/api/demo/status` | — | Anonymous quota left + whether live research is budget-paused |
| `GET`  | `/api/health` | — | Health check + cache stats |

### SSE Events

`POST /api/research` responds with a `text/event-stream`:

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

## 📁 Project Structure

```
├── backend/
│   ├── app/
│   │   ├── agents/       # LangGraph nodes (router, conversational, researcher,
│   │   │                 #   synthesizer) + graph wiring + shared state
│   │   ├── routers/      # FastAPI routes: auth, research, sessions, upload,
│   │   │                 #   notes, files, demo
│   │   ├── services/     # Groq LLM, Tavily + Serper search, Trafilatura scraper,
│   │   │                 #   FlashRank, Redis, answer_cache, embeddings, budget,
│   │   │                 #   anonymous quota, file_processor, usage/cost, tracing
│   │   ├── models/       # Pydantic schemas, SQLAlchemy models
│   │   ├── utils/        # Text chunking, citation extraction, semantic cache guards
│   │   ├── config.py     # Settings (pydantic-settings)
│   │   └── main.py       # FastAPI app, CORS, health check, router registration
│   ├── evals/            # Fixed query set + pure scorers + runner (weekly CI run)
│   ├── tests/            # pytest: pipeline, services, cache, guards, auth,
│   │                     #   rate limit, budget, uploads, tracing (355 tests)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── app/          # Next.js pages (home, research, login) + globals.css
│       ├── components/   # SearchBar, SourceCards, StreamingAnswer, DocumentViewer, …
│       ├── hooks/        # useResearch (SSE), useAuth
│       ├── lib/          # exportSession (PDF/Markdown/DOCX)
│       └── stores/       # Zustand (research state, notes, toasts)
├── .github/workflows/    # ci.yml, eval.yml (weekly), keepalive.yml
├── docker-compose.yml    # Postgres + Redis + Backend + Frontend
├── render.yaml           # Render blueprint (backend)
├── DEPLOYMENT.md         # Vercel + Render deployment guide
└── ROADMAP.md            # What's next
```

## 🌐 Deployment

The project deploys as **backend on Render** + **frontend on Vercel**, with
Postgres on Supabase. [`render.yaml`](render.yaml) is a ready-to-use blueprint.

Full step-by-step instructions — including the `asyncpg` session-pooler trap,
CORS setup, and Google OAuth origins — are in **[`DEPLOYMENT.md`](DEPLOYMENT.md)**.

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

### 10. Surviving a free tier in public

Opening the app to logged-out visitors meant anyone could spend the API budget.
The answer is layered: each visitor gets a cookie-scoped allowance before the
signup wall, and **daily/monthly search-credit ceilings plus a daily cost budget**
sit behind that. When a ceiling is hit, `/api/demo/status` reports it and the UI
shows a degraded banner instead of failing mid-stream. The per-visitor number is
a conversion lever; the ceilings are what actually protect the bill.

## 🗺️ Roadmap

Planned work and open questions live in [`ROADMAP.md`](ROADMAP.md).

## 🤝 Contributing

Issues and PRs are welcome.

1. Fork and branch off `main`.
2. Make your change, and add tests for it in `backend/tests/`.
3. Make sure CI passes locally: `pytest -q` in `backend/`, `npm run build` in `frontend/`.
4. Open a PR describing what changed and why.

If you're changing a prompt or the pipeline, run the eval set (`python -m
evals.run_eval`) and include the before/after scores — prompt changes should show
up as a number, not a vibe.

## 📝 License

[MIT](LICENSE) © Raghav Gangwa
