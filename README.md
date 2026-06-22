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
- **🔐 Auth** — email/password (bcrypt) + Google OAuth (fail-closed token validation), stateless JWT, per-user rate limiting
- **💾 Persistence** — PostgreSQL for sessions/history, Redis (optional) for search & scrape caching
- **🚀 Production-ready** — Dockerized services, GitHub Actions CI, pytest suite, deep health checks, and optional Sentry error tracking

## 🏗️ Architecture

```
User Query
  → Router (fast LLM)         triage: casual chat vs. research
      ├─ chat ──→ Conversational (fast LLM) → instant reply
      └─ research ↓
  → Researcher (parallel)     decompose into 2–4 sub-queries
                              → Tavily search+read  (Serper + Trafilatura fallback)
  → Re-ranker (FlashRank)     rank chunks; build canonical source list
  → Synthesizer (strong LLM)  stream a cited Markdown answer + follow-ups
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
| **Database** | PostgreSQL 16 |
| **Cache** | Redis 7 (optional) |
| **Infrastructure** | Docker Compose · Render (backend) + Vercel (frontend) |
| **Observability** | GitHub Actions CI · pytest · Sentry |

## 🧗 Engineering Journey — Challenges & Solutions

This started as a local-only prototype and evolved into a deployed, multi-user
product over ~3 weeks and 68 commits. The hardest problems weren't writing
features — they were latency, citation trust, concurrency, and going public.

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
event: token        → {"token": "word"}
event: follow_up    → {"suggestions": ["question1", "question2"]}
event: done         → {"session_id": "...", "total_sources": 8, "confidence": 0.89}
```

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
│   │   ├── services/     # Groq LLM, Tavily + Serper search, Trafilatura scraper, FlashRank, Redis, auth
│   │   ├── models/       # Pydantic schemas, SQLAlchemy models
│   │   ├── utils/        # Text chunking, citation extraction
│   │   └── main.py       # FastAPI app: auth, rate limiting, SSE research endpoint
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
