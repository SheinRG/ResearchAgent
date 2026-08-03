# goon.ai — Senior Engineering Interview Dossier

> A complete technical breakdown of the **goon.ai AI Research Agent**, written so the original developer can defend every design decision to a senior backend engineer. Everything below is grounded in the actual code in this repository, not the (stale) `implementation_plan.md`.

**One-line pitch:** A Perplexity-style research agent that triages a question, decomposes it into sub-queries, searches + reads the web in parallel, neural-reranks the evidence, and streams back a Markdown answer with trustworthy `[n]` citations — orchestrated as a LangGraph state machine on FastAPI, with a Next.js 16 streaming frontend.

---

## 1. Project Overview

### Purpose
goon.ai is an **autonomous, multi-source research assistant**. Instead of returning a list of blue links (search engine) or answering from frozen training data (a bare LLM), it performs *live retrieval-augmented generation*: it reads current web pages (and optionally the user's uploaded documents), grounds its answer strictly in those sources, and attaches inline citations so every factual claim is verifiable.

### Problem solved
- **LLM hallucination & staleness.** A raw LLM confidently invents facts and knows nothing after its training cutoff. goon.ai forces the model to answer *only* from retrieved sources and to cite them, so answers are current and checkable.
- **Manual research is slow.** A human doing this would open 10 tabs, skim each, and synthesize. goon.ai decomposes the question into 2–4 focused sub-queries, fans them out concurrently, and reads/ranks the results automatically.
- **Trust.** Search engines give you links but not an answer; chatbots give you an answer but not its provenance. This bridges the two: a synthesized answer *with* provenance.

### Target users
Knowledge workers, students, developers, and analysts who need a fast, cited, up-to-date answer to a non-trivial question — the same audience as Perplexity. Secondary use case: document Q&A (upload a PDF/DOCX and ask about it), where the pipeline treats the uploaded file as the primary, highest-priority source.

---

## 2. Complete Architecture

### High-level architecture diagram

```
                          ┌──────────────────────────────────────────────┐
                          │                BROWSER (client)               │
                          │  Next.js 16 / React 19 (App Router)           │
                          │  ┌────────────┐  ┌──────────────────────────┐ │
                          │  │ useAuth    │  │ useResearch (SSE reader) │ │
                          │  │ (JWT +     │  │  fetch → ReadableStream  │ │
                          │  │  refresh)  │  │  → parse event/data      │ │
                          │  └────────────┘  └──────────────────────────┘ │
                          │  Zustand store · react-markdown · Motion      │
                          └───────────────┬──────────────────────────────┘
                                          │ HTTPS
                     Bearer JWT (access)  │  HttpOnly cookie (refresh)
                                          ▼
                          ┌──────────────────────────────────────────────┐
                          │              FastAPI backend (ASGI)           │
                          │  CORS · Sentry · Langfuse · lifespan(db/llm)  │
                          │  Routers: auth · research · sessions ·        │
                          │           upload · files · notes              │
                          │  Depends(get_current_user) · check_rate_limit │
                          └───────┬───────────────────────┬──────────────┘
                                  │ SSE StreamingResponse  │
                                  ▼                        ▼
              ┌─────────────────────────────────┐
              │   ANSWER CACHE (before triage)  │  hit → replay full SSE
              │   1. exact key   (always on)    │        sequence, $0,
              │   2. semantic    (opt-in, 1st   │        no model calls
              │      turn only, 4 guards)       │
              └───────────────┬─────────────────┘
                              │ miss
                              ▼
        ┌──────────────────────────────────────┐   ┌───────────────────────┐
        │      LangGraph state machine         │   │   PostgreSQL (asyncpg) │
        │                                      │   │  users · sessions ·    │
        │   router/triage ──chat──► conversational│  research_queries ·    │
        │        │                          │   │   notes · uploaded_files│
        │     research                      │   └───────────────────────┘
        │        ▼                          │   ┌───────────────────────┐
        │   researcher (fan-out asyncio)    │   │   Redis (optional)     │
        │        ▼                          │   │  search/scrape cache · │
        │   reranker (FlashRank, CPU)       │   │  ANSWER cache + vector │
        │        ▼                          │   │   index (semantic) ·   │
        │   synthesizer (stream + followups)│   │  rate-limit counters · │
        │        ▼                          │   │  refresh-token store   │
        │   store answer (TTL by staleness) │   └───────────────────────┘
        └───────┬──────────────┬────────────┘
                │              │
                ▼              ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │  Groq Cloud  │  │  Tavily API  │  │  Serper API  │
        │  8B (triage) │  │ search+read  │  │ images +     │
        │  70B (synth) │  │  (primary)   │  │ fallback srch│
        └──────────────┘  └──────────────┘  └──────────────┘
                                             (+ Trafilatura scrape fallback)
```

### Request flow (a research query)
1. User types a question on `/` (home) → frontend navigates to `/research?q=...`.
2. `useResearch.startResearch()` `POST`s to `/api/research` with a `Bearer` access token, conversation `history`, and any `documents`.
3. FastAPI `research` route authenticates (`Depends(get_current_user)`), enforces the per-user hourly rate limit, loads the user's `preferred_name`, and returns a `StreamingResponse` of `text/event-stream`.
3b. **Answer-cache lookup, before any model runs.** An exact key (`query + history + preferred_name + synth model + cache version`) is checked first; on a miss — and only for a *first turn*, with the feature enabled and an embedding key present — the question is embedded and matched against previously answered ones in the same bucket. A hit **replays the entire recorded SSE sequence** (sources, images, trace, follow-ups) and returns `cached: true` with an all-zero `usage`, having made **zero LLM calls**. Uploads bypass the cache entirely, because document text isn't in the key and a hit could answer from the wrong file.
4. Inside the stream, an `asyncio.Queue` decouples the LangGraph run from the HTTP response: a background task runs `graph.astream(initial_state)` and every node pushes SSE events (`phase`, `sub_queries`, `sources`, `images`, `token`, `follow_up`) onto the queue; the response loop drains the queue and yields `event: … / data: …` frames.
5. LangGraph nodes execute: **router/triage** (one LLM call → chat-vs-research + sub-queries + answer format + **`time_sensitive`**) → **researcher** (parallel Tavily search+read) → **reranker** (FlashRank) → **synthesizer** (streams the cited answer; generates follow-ups concurrently).
6. On completion, a `_final_state` sentinel triggers `_save_session()` (persist the turn to Postgres), a fire-and-forget `store_answer()` (TTL chosen by the `time_sensitive` flag; also indexes the question's vector when one was computed during the missed semantic lookup — so indexing costs nothing extra), and a final `event: done` frame with `session_id`, `total_sources`, `confidence`, `latency_ms`, `usage` (tokens + cost), and the `cached`/`cache_kind` fields.
7. The frontend accumulates tokens into the answer, swaps in the authoritative citation-ordered source list on the `sources … replace:true` event, and freezes the finished turn into the conversation thread.

### Component interactions
- **Frontend ↔ Backend:** REST for auth/sessions/notes/upload; **SSE** (one long POST) for research streaming.
- **Backend ↔ LangGraph:** the router builds a singleton compiled graph; the route passes an `sse_callback` into the graph state so nodes can emit progress without knowing about HTTP.
- **Backend ↔ external APIs:** Groq (LLM), Tavily (search+read), Serper (images + fallback). All wrapped in async `httpx` clients with retries/timeouts and best-effort failure handling.
- **Backend ↔ data stores:** SQLAlchemy async (Postgres) for durable state; Redis for cache, rate-limit counters, and refresh tokens — all *optional* (graceful degradation).

### Data flow
`query → sub_queries → search_results (deduped) → scraped/read content → chunks → ranked_chunks (top-k) → cited_sources + numbered context → streamed answer + citations → persisted ResearchQuery row`. The `ResearchState` TypedDict is the single shared object threaded through every node, with reducers (`_merge_lists`, `operator.add`) for fields that could see concurrent writes.

---

## 3. Tech Stack

| Layer | Technology | Why chosen | Alternatives & trade-offs |
|---|---|---|---|
| Frontend framework | **Next.js 16 (App Router) + React 19** | SSR/streaming-friendly, file-based routing, first-class Vercel deploy, RSC. | Plain React/Vite (no SSR, more setup); Remix. Next 16 has breaking changes vs older versions (repo explicitly flags this in `AGENTS.md`). |
| Styling | **Vanilla CSS (globals.css) + CSS variables + Motion** | Full control, no build-time CSS-in-JS cost, easy theming (light/dark + accent). | Tailwind (faster to write, larger mental model); styled-components (runtime cost). |
| Client state | **Zustand** (+ `persist`) | Tiny, hook-based, no boilerplate; persists `recentSearches`, keeps a transient `sessionsNonce` to trigger sidebar refetches. | Redux (heavy), React Context alone (re-render churn). |
| Markdown | **react-markdown + remark-gfm** | Renders the model's GFM output incl. tables; safe (no `dangerouslySetInnerHTML`). | Manual parsing (error-prone), `marked` + sanitizer. |
| Backend framework | **FastAPI (Python 3.12)** | Native async (critical for I/O-bound fan-out), Pydantic validation, automatic OpenAPI docs, ASGI streaming. | Flask (sync-first), Django (heavy), Node/Express (would fragment the AI stack away from Python's ML ecosystem). |
| Agent orchestration | **LangGraph** | Explicit state-machine graph with typed state + reducers; models the pipeline as nodes/edges and supports `astream`. | LangChain agents (more magic, less control), CrewAI (multi-agent-role focus), hand-rolled async (what this effectively collapsed toward — see §13). |
| LLM | **Groq Cloud** — `llama-3.1-8b-instant` (triage/follow-ups) + `llama-3.3-70b-versatile` (synthesis) | Groq's LPU inference is extremely fast (key to sub-5s latency); two-tier strategy spends the big model only where quality matters. | OpenAI/Anthropic (higher quality, higher latency/cost), local Ollama (free but slow — the *original* plan, abandoned for latency). |
| Search + read | **Tavily (primary), Serper (images + fallback)** | Tavily returns results *and* cleaned page content in **one** call — halves round-trips and handles JS-heavy pages the raw scraper failed on. | Serper+Trafilatura (2 round-trips, empty on JS sites — kept as fallback), Bing/Google CSE, SerpAPI. |
| Extraction | **Trafilatura** | Best-in-class boilerplate removal for the fallback scrape path. | readability-lxml, newspaper3k, BeautifulSoup hand-rolled. |
| Re-ranking | **FlashRank (CPU-only)** — `ms-marco-TinyBERT-L-2-v2` default | Cross-encoder relevance ranking with no GPU; TinyBERT fits a 512MB instance. | Cohere Rerank / bge-reranker (better but network/GPU), embedding cosine (bi-encoder, weaker than cross-encoder). |
| Database | **PostgreSQL 16 (SQLAlchemy async + asyncpg)** | Relational integrity for users/sessions/queries, JSON columns for flexible payloads, mature managed hosting. | MongoDB (looser schema), SQLite (no concurrency at scale). |
| Cache / ephemeral | **Redis 7** (optional) | Sub-ms cache for search/scrape **and whole answers**, a hash-backed vector index for the semantic layer, atomic `INCR` rate limiting, refresh-token store with TTL. | Memcached (no data structures — the semantic index needs hashes), in-proc dict (doesn't survive restart / multi-instance — used only as fallback). |
| Embeddings | **Jina `jina-embeddings-v3` (or OpenAI), 384-dim, optional** | Used **only** to match near-duplicate *questions* for the cache — never for retrieval. Groq serves no embedding models, so this is the one place a second inference provider appears. Jina supports Matryoshka truncation, so 384 dims costs less to send and store. | Local sentence-transformers (adds ~100MB+ to a 512MB container), pgvector/Qdrant (a whole index to run for ≤300 vectors per bucket). |
| Vector search | **Brute-force cosine over a capped in-process bucket** (`math.sumprod` on pre-normalized vectors) | ≤300 vectors scored in ~ms with no extra infrastructure; storing unit vectors turns cosine into a dot product. | FAISS/HNSW/pgvector — correct at millions of vectors, pure overhead at hundreds. Called out as the migration path, not a gap. |
| Auth | **PyJWT (HS256) + bcrypt + Google OAuth (tokeninfo)** | Stateless access tokens, industry-standard password hashing, federated login. | Sessions in DB (stateful), Auth0/Clerk (vendor lock-in, cost). |
| Infra | **Docker Compose (dev) · Render (backend) · Vercel (frontend)** | Reproducible local stack; managed Postgres/Redis on Render; Vercel is the natural Next.js host. | Kubernetes (overkill for one service), single VM (manual ops). |
| Observability | **GitHub Actions CI · pytest · Sentry (optional) · Langfuse tracing (optional) · weekly eval run** | Catch regressions pre-merge; error tracking that no-ops without a DSN; a span per pipeline stage with token/cost attribution so latency and spend are measured, not guessed; a fixed scored query set turns prompt changes into numbers. | Datadog/New Relic (cost, LLM-unaware), OpenTelemetry raw (more wiring, no LLM-native cost view), no CI (risky). |

---

## 4. Folder Structure

```
backend/app/
  agents/     LangGraph nodes + graph wiring — the "brain"
    graph.py          builds & compiles the singleton StateGraph
    state.py          ResearchState TypedDict + reducers + format_history()
    router.py         triage node (route + plan + format, ONE LLM call)
    conversational.py chat-mode direct reply node
    researcher.py     search→read→chunk fan-out + rerank_node
    synthesizer.py    cited-answer streaming + concurrent follow-up gen
  services/   External integrations & stateful singletons
    llm.py            async Groq client (generate / stream / structured / health) + retries
    tavily.py         search+read in one call (primary)
    search.py         Serper web + images (fallback + Images tab)
    scraper.py        Trafilatura scrape (fallback path), pooled httpx, concurrency cap
    reranker.py       FlashRank cross-encoder, thread-safe lazy init
    cache.py          Redis get/set/mget + hash ops + counters, no-cache fallback
    answer_cache.py   whole-answer store/replay: exact key, semantic lookup,
                      TTL-by-staleness, vector index + pruning, hit/miss tallies
    embeddings.py     Jina/OpenAI embeddings — never raises, never blocks long,
                      returns normalized vectors (cosine becomes a dot product)
    usage.py          per-turn token + cost accumulation (contextvar-scoped)
    tracing.py        Langfuse spans per stage; no-ops when unconfigured
    auth.py           JWT, bcrypt, refresh-token rotation, Google token verify
    file_processor.py extract text from txt/md/pdf/docx
  models/     Data contracts
    schemas.py        Pydantic request/response + validators
    database.py       SQLAlchemy async models, engine/session factory, init_db migrations
  routers/    HTTP surface (one module per resource)
    auth.py · research.py · sessions.py · upload.py · files.py · notes.py
  utils/      Pure helpers (fully unit-tested)
    chunker.py        recursive character text splitter
    citations.py      canonical source list + citation extraction
    semantic_guards.py four rails a semantic cache hit must clear (numbers,
                      polarity, comparison order, word overlap) — pure, each
                      returns a *reason string* so rejections are countable
  config.py           pydantic-settings, single source of tunables
  main.py             app factory, CORS, lifespan, /api/health
  evals/              fixed query set · pure scorers · runner · baseline.json
  tests/              pytest: pipeline (pure), services, cache, semantic guards,
                      scorers, auth, rate_limit

frontend/src/
  app/         Next.js App Router pages + error/loading boundaries
    page.js (home) · research/page.js · login/page.js · layout.js
    error.js · global-error.js · not-found.js · loading.js
  components/  UI (SearchBar, ResearchTurn, StreamingAnswer, SourceCards,
               ImageGrid, ResearchTabs, CitationTooltip, DocumentViewer,
               AppLayout, providers, Toast, …)
  hooks/       useResearch (SSE), useAuth (JWT + refresh)
  stores/      Zustand: researchStore, toastStore
  lib/         exportSession, safeUrl (URL hardening)
```

Responsibility split is clean: **routers** are thin HTTP adapters, **services** own external I/O and singletons, **agents** own orchestration logic, **utils** are pure and tested, **models** are the contracts. This is a deliberate hexagonal-ish separation you can point to in an interview.

---

## 5. Backend

### Framework & API architecture
FastAPI on ASGI (uvicorn). The app is assembled by an **application factory** in `main.py` with a `lifespan` context manager that (a) warns if `AUTH_SECRET` is still the insecure default, (b) initializes the DB, and (c) health-checks the LLM at startup. Routers are mounted per-resource. The design is **stateless per-request** (JWT auth, no server session) so it scales horizontally, with all shared state pushed to Postgres/Redis.

### Routes (every endpoint)
| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/auth/register` | — | Create account (email/pw), set refresh cookie, return access JWT |
| POST | `/api/auth/login` | — | Verify bcrypt password, issue tokens |
| POST | `/api/auth/google` | — | Verify Google ID token, link/create account |
| POST | `/api/auth/refresh` | cookie | Rotate refresh token → new access JWT |
| POST | `/api/auth/logout` | cookie | Revoke refresh token, clear cookie |
| GET | `/api/auth/me` | JWT | Current user + preferred_name |
| PATCH | `/api/auth/profile` | JWT | Update preferred_name |
| GET | `/api/auth/rate-limit` | JWT | Current hourly usage/remaining |
| POST | `/api/research` | JWT | **SSE** research stream (the core endpoint) |
| GET | `/api/sessions` | JWT | List user's threads (grouped by session) |
| GET | `/api/sessions/{id}` | — (public by UUID) | Full thread with all turns |
| DELETE | `/api/sessions/{id}` | JWT (owner) | Delete a thread |
| POST | `/api/upload` | JWT | Upload file, extract text, persist bytes |
| GET | `/api/files/{id}` | — (public by UUID) | Serve stored file bytes (viewer) |
| GET/POST/PATCH/DELETE | `/api/notes[/{id}]` | JWT | Notes CRUD |
| GET | `/api/health` | — | Liveness/readiness (LLM+DB+Redis, concurrent) + cache tallies (`hit_rate`, `semantic_rescue_rate`) |

### Controllers / Services / Middleware
There isn't a separate "controller" layer — FastAPI **route functions are the controllers**, and they delegate to the **services** package (the real business logic lives there). Middleware is `CORSMiddleware` plus Sentry's auto-instrumentation; cross-cutting concerns like auth and rate limiting are implemented as **FastAPI dependencies** (`Depends(get_current_user)`) and explicit calls (`await check_rate_limit(user_id)`), which is idiomatic and testable.

### Authentication
Stateless **JWT (HS256)** access tokens (1-hour expiry) carried as `Authorization: Bearer`. Long-lived **refresh tokens** (30 days, opaque `secrets.token_urlsafe`) are stored **hashed** in Redis and delivered as an `HttpOnly`, `Secure` (in prod), `SameSite` cookie scoped to `/api/auth`. Refresh uses **rotation**: the old token is deleted and a new one issued in a Redis pipeline, so a stolen token is detectable. Google OAuth verifies the ID token via Google's `tokeninfo` endpoint and **fails closed** (rejects if no client ID configured, checks `aud`, `iss`, and `email_verified`).

### Validation
**Pydantic v2** everywhere. Request bodies are typed models (`ResearchRequest`, `RegisterRequest`, …) with field constraints (`min_length`, `max_length`, `ge/le`) and custom `field_validator`s: password policy (≥8, 1 uppercase, 1 digit), control-character stripping on any text destined for a prompt (`preferred_name`, document `name`/`text`), and bounded list sizes (`history` ≤20, `documents` ≤5). This is both a correctness and a **security** boundary (prompt-injection hardening starts here).

### Error handling
Layered and defensive:
- **External calls** (Groq, Tavily, Serper, scraper) catch timeouts/HTTP errors and return empty/fallback instead of throwing, so one bad sub-query never kills a run.
- **LLM calls** retry transient failures (429/5xx/timeouts) with exponential backoff; streaming retries only *setup*, never mid-stream (can't safely restart a partial stream).
- **The research route** wraps the whole agent run; failures push an `error` SSE event and a `_final_state`, and a 5-minute watchdog (`asyncio.wait_for`) prevents a hung run from holding the connection forever.
- **DB writes** for session-save are try/wrapped so a persistence failure doesn't break the user's answer.
- **HTTP errors** surface as proper status codes (401/404/409/413/429/503) that the frontend maps to friendly messages.

---

## 6. Frontend

### Component hierarchy
```
RootLayout (layout.js)
 └─ ThemeProvider → AccentProvider → AuthProvider
     └─ AppLayout (sidebar: history, notes, rate-limit bar, profile/appearance menus)
         └─ page:
            ├─ HomePage (/)            greeting + SearchBar + rotating suggestions
            ├─ ResearchPage (/research) Suspense → ResearchContent
            │    ├─ SessionHeader
            │    ├─ ResearchTurn[]  (completed turns)
            │    │    └─ ResearchTabs (Answer|Sources|Images|Trace)
            │    │        ├─ StreamingAnswer (react-markdown + CitationTooltip)
            │    │        ├─ SourceCards / SourcesList
            │    │        └─ ImageGrid
            │    ├─ ResearchTurn (live, streaming)
            │    ├─ SearchBar (follow-up composer)
            │    └─ DocumentViewer (iframe of /api/files/{id})
            └─ LoginPage (/login)
     └─ Toast (portal)
```

### State management
- **Server/auth state:** `AuthProvider` (React Context) holds `user`/`token`, persists to `localStorage`, and exposes `login/register/loginWithGoogle/logout/refreshSession/updateProfile`.
- **Streaming state:** `useResearch` keeps live `phase/sources/images/answer/followUps` in `useState` **and mirrors each in a `useRef`** so the instant the `done` event fires it can hand a complete snapshot to `onComplete` without waiting for a re-render.
- **Global UI state:** Zustand `researchStore` (recent searches persisted; `sessionsNonce` transient to trigger sidebar refetch; `pendingDocuments` to hand files across the home→research navigation) and `toastStore`.

### Routing
Next.js App Router. Home stages a query and pushes `/research?q=…`; the research page is driven entirely by URL params: `?q=` starts a **fresh live run** (Effect B), `?session=` **restores a stored thread** from the DB without re-running (Effect A). Guards with refs (`seedRef`, `loadedSessionRef`) prevent double-firing on re-render/HMR. Shared `?session=` links are public (no login required to view).

### Data fetching
- **REST** via `fetch` for auth/sessions/notes/upload (Bearer header; refresh cookie sent with `credentials:"include"`).
- **SSE** via `fetch` + `response.body.getReader()` + `TextDecoder`, manually parsing `event:`/`data:` frames from a rolling buffer. On `401` it transparently calls `refreshSession()` and retries once; on `429`/`5xx` it surfaces typed error messages. A dev-only simulation streams a fake answer when the backend is offline (strictly gated to `NODE_ENV==='development'`).

### UI architecture
Perplexity-style: a right-aligned chat-bubble question, then tabbed results (**Answer / Sources / Images / Trace**). Answers render as GFM Markdown (tables, lists) with citation markers turned into hover tooltips linking to the exact source. The **Trace** tab is the receipt — what was ranked, what actually reached the model, and what got cited — and a **reused-answer notice** appears above the answer whenever a semantic cache hit served a differently-worded question, quoting the question it reused. That disclosure is deliberately a visible banner and not a tooltip: the user asked one thing and is reading the answer to another. Theme system: `next-themes` (light/dark/system) + a custom accent provider (blue/terracotta/green) via CSS variables. Error/loading are handled by App Router conventions (`error.js`, `global-error.js`, `loading.js`, `not-found.js`).

---

## 7. Database

### Engine & schema
PostgreSQL 16 accessed through **SQLAlchemy 2.0 async** with the `asyncpg` driver. A `_normalize_db_url()` helper rewrites managed-host `postgres://`/`postgresql://` URLs to `postgresql+asyncpg://` so the app boots on Render/Neon/Railway without manual fixing. Engine uses `pool_size=5, max_overflow=10, pool_pre_ping=True`.

### Tables (models)
- **users** — `id` (uuid str PK), `email` (unique, indexed), `name`, `preferred_name`, `password_hash` (nullable for Google-only), `google_id` (unique, indexed, nullable), `picture`, `created_at`.
- **research_sessions** — `id` PK, `user_id` FK→users (indexed, nullable), `created_at`, `updated_at`.
- **research_queries** — `id` PK, `session_id` FK→sessions, `user_id` FK (indexed), `query`, `sub_queries` (JSON), `answer`, `sources` (JSON), `citations` (JSON), `confidence`, `iterations`, `follow_up_suggestions` (JSON), `documents` (JSON), `created_at`.
- **notes** — `id` PK, `user_id` FK (indexed), `text`, timestamps.
- **uploaded_files** — `id` PK, `user_id` FK (indexed), `filename`, `mime`, `size`, `content` (`LargeBinary`), `created_at`.

### Relationships
`User 1─* ResearchSession 1─* ResearchQuery`, `User 1─* Note`, with `cascade="all, delete-orphan"` on the ORM relationships. `UploadedFile` is intentionally standalone (looked up by opaque UUID). A "thread" in the UI is a `session_id` grouping its ordered `research_queries` turns.

### Indexes & query optimization
Indexes on `users.email`, `users.google_id`, and the `user_id` FKs on sessions/queries/notes (the columns actually filtered on). `list_sessions` fetches the user's most recent 200 query rows and groups them in Python into threads — pragmatic for the current scale; at larger scale this becomes a windowed/aggregate SQL query. Reads use `select(...).where(...).order_by(...)`; writes commit inside `async with factory()` blocks. Schema evolution is handled by `init_db()` running `create_all` plus **idempotent `ALTER TABLE … ADD COLUMN IF NOT EXISTS`** for new columns (a lightweight stand-in until Alembic is introduced — a known, called-out limitation).

---

## 8. AI Components

### LLM providers & strategy
**Groq Cloud**, two-tier: `llama-3.1-8b-instant` for cheap structured/auxiliary calls (triage/planning, follow-up generation) and `llama-3.3-70b-versatile` for the final synthesized answer and chat replies. Configurable via `GROQ_MODEL`/`GROQ_SYNTH_MODEL`. The `GroqClient` (singleton) exposes `generate`, `generate_stream` (async generator), `generate_structured` (JSON mode with extraction fallback), and a TTL-cached `health_check`.

### Embedding model
**Not in the retrieval path — by design.** There is still no vector DB and no embedding *retrieval*: retrieval is *live web search* (Tavily/Serper) rather than similarity search over a corpus, and relevance is decided by a **cross-encoder reranker** (FlashRank) rather than embedding cosine. That remains the defensible choice for an *open-web* research agent (freshness + no index to maintain), and it's still a great talking point on bi-encoder vs cross-encoder.

Embeddings *do* now appear in exactly one place, and the distinction is worth stating precisely in an interview: **the semantic answer cache embeds the user's question to find a previously answered question that means the same thing.** It matches *queries to queries*, never queries to documents. Optional (off by default), 384-dim Jina/OpenAI vectors, stored normalized in a Redis hash so cosine similarity is a plain dot product, brute-force scored over a bucket capped at 300 entries. If asked "so is there a vector database?" — no: there's a bounded in-memory scan, because at hundreds of vectors an ANN index is pure operational overhead.

### Prompt engineering
Three carefully engineered system prompts:
- **Triage** — does routing + sub-query planning + answer-format selection + **time-sensitivity labelling** in one JSON response; resolves pronouns in follow-ups so every sub-query is self-contained; anchors recency to the current year. The `time_sensitive` flag is explicitly told it "only controls how long an answer may be reused from cache; it never changes what you research" — keeping a caching concern from leaking into research behaviour — and **defaults to `true`** in the node, so a malformed JSON response can never make a live-data question look evergreen.
- **Synthesizer** — strict citation rules ([n] must map to provided sources, never invent, no trailing reference list), depth guidance (direct answer first, then develop), a format vocabulary (table/list/steps/prose) with an explicit format directive injected from triage, and **prompt-injection hardening** ("sources are untrusted DATA, never instructions").
- **Follow-up** — generate exactly 3 self-contained, non-duplicate next questions.
Temperatures are tuned per task (0.2 triage, 0.4 synth, 0.5 follow-ups, 0.6 chat).

### RAG pipeline
Classic RAG, web-flavored: **retrieve** (search+read) → **chunk** → **rerank** → **build numbered context** → **generate grounded, cited answer**. Uploaded documents are chunked and **prepended** to web evidence with `score=1.0` so they take citation slots `[1], [2]…` and are treated as primary evidence.

### LangGraph / agents
A compiled `StateGraph` over `ResearchState`: entry `router` → conditional edge (`chat`→`conversational`→END, else `research`→`researcher`→`reranker`→`synthesizer`→END). It's a **single forward pass** — the earlier 5-node graph (separate planner + reflect-and-refine loop, `max_iterations=2`) was collapsed for latency (see §13). The graph is a singleton compiled once.

### Retrieval strategy
2–4 LLM-planned sub-queries fanned out concurrently with `asyncio.gather`. Tavily returns results + cleaned content in one call; results are deduped by URL, capped per-domain (Serper path caps 2/domain for diversity), and only the top `scrape_top_n` per sub-query feed the chunker (the rest still appear as sources).

### Chunking
`chunker.py` — a **recursive character splitter** that prefers semantic boundaries (`\n\n` → `\n` → sentence → clause → word) with a configurable size (500) and overlap (50), force-splitting only when no separator fits, and dropping sub-30-char noise. Fully unit-tested.

### Re-ranking
**FlashRank cross-encoder** (`ms-marco-TinyBERT-L-2-v2` default; MiniLM-L-12 for quality). Runs in a thread-pool executor (CPU-bound, keeps the event loop free), thread-safe **double-checked-locked** singleton init, top-k=12, with a graceful fallback that returns chunks at score 0.5 if ranking throws.

### Memory
Conversation memory is **turn history passed per request** (not server-side): the last ~4 turns are compacted by `format_history()` (answers truncated to ~600–800 chars) and injected into triage/synthesizer prompts so follow-ups resolve references and read as a continuation. Personalization (`preferred_name`) is a lightweight long-term memory pulled from Postgres.

### Streaming
End-to-end token streaming: Groq stream → node emits `token` SSE events via the queue → SSE frames → frontend appends to the answer live. Phase/sub-query/sources/images events give a real-time progress UI while the answer is still being written.

### Tool calling & fallback mechanisms
Rather than LLM "function calling," tools are **deterministic pipeline nodes** (search, scrape, rerank) — more predictable and cheaper than letting the model choose tools. Fallbacks are layered: Tavily → (empty) → Serper+Trafilatura; reranker → identity-score fallback; Redis down → no-cache + local rate limiter; embedding provider down/slow/unauthorized → `None` → semantic lookup silently skipped (the cache degrades, the product doesn't); no usable sources → an honest "couldn't find reliable sources" message instead of hallucinating; JSON parse failure → brace-extraction retry then safe default; missing `time_sensitive` → assume it goes stale.

---

## 9. APIs

### `POST /api/research` (the core endpoint)
**Auth:** `Bearer` JWT. **Request:**
```json
{
  "query": "best DSA sheets online",
  "max_iterations": 1,
  "session_id": "uuid-or-null",
  "history": [{"query": "...", "answer": "..."}],
  "documents": [{"name":"f.pdf","text":"...","file_id":"...","mime":"...","size":123}]
}
```
**Response:** `text/event-stream` (SSE). Event sequence:
```
event: phase       data: {"phase":"planning","message":"Breaking down your question..."}
event: sub_queries data: {"queries":["...","..."]}
event: phase       data: {"phase":"searching","message":"Searching 3 sub-questions..."}
event: sources     data: {"sources":[{url,title,domain,favicon,snippet}]}
event: images      data: {"images":[{url,thumbnail,title,source,domain}]}
event: phase       data: {"phase":"reading","message":"Reading and analyzing 6 sources..."}
event: phase       data: {"phase":"writing","message":"Synthesizing your answer..."}
event: sources     data: {"sources":[...],"replace":true}    # authoritative, citation-ordered
event: token       data: {"token":"Modern "}                 # …many…
event: follow_up   data: {"suggestions":["...","...","..."]}
event: done        data: {"session_id":"...","total_sources":8,"confidence":0.89,"latency_ms":4520}
```
**Error cases:** `401` missing/expired token (frontend auto-refreshes + retries once), `429` rate limit exceeded, `503/502` server unavailable, mid-run failure → `event: error`, and a 5-minute timeout watchdog.

### Auth endpoints
- `POST /api/auth/register` — body `{email, password(≥8,1 upper,1 digit), name?}`; `409` if email exists; returns `{token, user}` + sets refresh cookie.
- `POST /api/auth/login` — `{email, password}`; `401` invalid credentials; returns `{token, user}`.
- `POST /api/auth/google` — `{credential}`; `401` invalid Google token; links by `google_id` or `email`.
- `POST /api/auth/refresh` — reads refresh cookie; `401` if invalid/expired (clears cookie); returns new `{token, user}` + rotated cookie.
- `POST /api/auth/logout` — revokes refresh token, clears cookie.
- `GET /api/auth/me` / `PATCH /api/auth/profile` / `GET /api/auth/rate-limit` — JWT-guarded.

### Sessions / upload / files / notes
- `GET /api/sessions?limit=20` — grouped thread list (title, turn_count, timestamps, confidence).
- `GET /api/sessions/{id}` — full thread (public by UUID); `404` if none.
- `DELETE /api/sessions/{id}` — owner-only (ownership checked before delete), `404` otherwise.
- `POST /api/upload` — multipart file; `413` if >5MB, `400` unsupported type, `500` extraction failure; returns extracted text + `file_id`.
- `GET /api/files/{id}` — serves bytes `inline` (public by UUID).
- `GET/POST/PATCH/DELETE /api/notes` — per-user CRUD, ownership enforced, `404` on foreign/missing note.

---

## 10. Security

### JWT
HS256 signed with `AUTH_SECRET` (startup warns if it's still the default; Render `generateValue: true` mints a strong one). Access tokens are short-lived (1h) with `sub/email/name/iat/exp`. Validation catches expired vs invalid separately. Because they're stateless, revocation of *access* tokens isn't possible before expiry — mitigated by the short TTL; *refresh* tokens are revocable (stored in Redis).

### Password hashing
**bcrypt** with per-password salt (`bcrypt.gensalt()`), constant-time `checkpw`. Passwords are never logged or stored in plaintext; Google-only accounts have `password_hash = NULL` and can't be password-logged-in.

### Authorization
Two models: **JWT-guarded owner-scoped** resources (notes, session delete, profile) verify `user_id` ownership before mutating; **public-by-opaque-UUID** resources (session read, file read) treat the unguessable UUID as a capability (sharing feature). This distinction is deliberate and defensible.

### Input validation & prompt-injection hardening
Pydantic constraints on every field; control characters stripped from anything entering a prompt; document text length-capped (16k chars in schema, 12k in extraction). The synthesizer prompt explicitly frames sources as untrusted data and instructs the model to ignore any instructions embedded in them — a concrete prompt-injection defense.

### Rate limiting
Per-user, **fixed-window, 30 queries/hour**. Redis path uses an **atomic `INCR`-then-check** pipeline (sets TTL only on first hit via `nx`) so concurrent requests can't slip past the cap. If Redis is down, a **process-local fixed-window** limiter fails *closed* (bounded abuse per instance) rather than removing the limit — important because each query costs real money on Groq/Tavily/Serper.

### CORS & cookies
`CORSMiddleware` restricts origins to an explicit allow-list (`CORS_ORIGINS`, set to the Vercel URL in prod) with `allow_credentials=True`. Refresh cookie is `HttpOnly` (JS can't read it), `Secure` + `SameSite=None` in production (cross-site Vercel↔Render), scoped to `/api/auth`. Google OAuth **fails closed** on missing config and validates `aud`/`iss`/`email_verified` to prevent audience-confusion account takeover.

---

## 11. Performance

### The headline: ~35s → ~4.5s (warm), ~87% faster
Three rounds of optimization (documented in README + git history):

**Optimizations**
- **Collapsed the graph** from 5 nodes (planner + reflect/refine loop, 2 iterations) to a single triage→research→synthesize pass — removed the two biggest latency sinks for marginal quality loss.
- **Merged routing + planning + format** into one LLM call (triage), removing a sequential round-trip per query.
- **One-call search+read (Tavily)** replaced search-then-scrape (two round-trips, empty on JS pages).
- **Concurrent follow-up generation** overlaps the small aux call with the slow answer stream instead of appending a round-trip.
- **Per-result relevance excerpts** (not full-page raw content) for grounding — ~1–2s vs ~7s cold.
- **Trimmed knobs:** `scrape_top_n=2`, `scrape_timeout=8s`, TinyBERT reranker.

**Caching — three layers, and the interesting one is the third**

1. **Search/scrape cache.** Redis caches search results (`tavily`/`search`) and scraped pages (`scrape`) keyed by MD5 of the query/URL (1h TTL), with **batch `MGET`** for scrape lookups. LLM `health_check` is TTL-cached so health polls don't hammer Groq. This avoids *network* hops only — **both LLM calls still ran** on a repeat question, which is the expensive part.
2. **Exact answer cache.** Stores the complete user-visible output of a run and, on a hit, **replays the whole recorded SSE sequence** (`sources`, `images`, `trace`, `token`s, `follow_up`) so a replayed turn is byte-for-byte the same experience as a live one — just at **$0 with zero model calls**, reported honestly as an all-zero `usage` block rather than an omitted field. Key = `query + history + preferred_name + synth model + cache version`; the version constant lets a prompt change invalidate every entry at once. Uploads bypass entirely (document text isn't in the key, so a hit could answer from the wrong file).
   - **TTL by staleness, not by guess.** The cache is consulted *before* triage runs, so the original design used one deliberately short TTL for everything. Triage now emits `time_sensitive`, and that label is stored *with the answer* — so a price expires in 15 min and a definition lives 6 h. The label describes the answer that was **stored**, which is the only thing knowable at lookup time.
3. **Semantic answer cache** (optional, off by default — see §13.8). On an exact miss, embed the question and find a stored one that *means* the same thing. Candidates are scored best-first and the first one that clears all four guards wins, so one very-close-but-rejected match doesn't block a slightly weaker legitimate one.

**Instrumentation, so the cache is falsifiable**
`/api/health` reports `hits`, `misses`, `hit_rate`, `semantic_hits`, `semantic_rejected`, and `semantic_rescue_rate` (share of otherwise-missed questions rescued by meaning-matching). Without these the cache is unfalsifiable — it either saves a lot of money or none at all, and the logs alone never say which. Cache hits also emit a Langfuse `research` observation with `cached: true`, so cached and live turns sit side by side in the dashboard.

**Concurrency / async**
Fully async I/O. Sub-queries and scrapes run with `asyncio.gather`; scrapes are bounded by a `Semaphore(8)`; the SSE producer/consumer are decoupled by an `asyncio.Queue` so a slow client can't block the graph.

**Async offloading of CPU work**
CPU-bound chunking and FlashRank reranking are pushed to the default thread-pool executor (`loop.run_in_executor`) so they never block the event loop.

**Connection reuse**
Persistent, pooled `httpx.AsyncClient`s for Tavily/Serper/scraper (keep-alive, no repeated TLS handshakes); SQLAlchemy connection pool with `pool_pre_ping`.

**DB optimizations**
Indexed FK/email/google_id columns; `sources[:20]` capped on save; `expire_on_commit=False` avoids reload round-trips after commit.

---

## 12. Deployment

### Docker
Multi-stage where it matters. **Backend** `Dockerfile`: `python:3.12-slim`, builds wheels, runs as a **non-root** `appuser`, bakes a `HEALTHCHECK` hitting `/api/health`, prod `CMD` is uvicorn **without** `--reload`. **Frontend** `Dockerfile`: two-stage (node:20-alpine builder → runner), `NEXT_PUBLIC_*` inlined as **build args** (they're compiled into the client bundle), runs as non-root `node`.

### Docker Compose (dev)
Brings up Postgres 16 + Redis 7 (both with healthchecks and named volumes) + backend + frontend. Backend depends on healthy Postgres/Redis, mounts source for hot-reload (`--reload`, with `WATCHFILES_FORCE_POLLING=true` to dodge a Windows/WSL2 inotify crash), and reads secrets from a root `.env`.

### Environment variables
Centralized in `pydantic-settings` (`config.py`) with sensible defaults: Groq keys/models/timeouts/retries, Tavily/Serper keys + toggles, Redis/Postgres URLs, `AUTH_SECRET`, Google client id, agent knobs (`max_sub_queries`, `scrape_top_n`, `rerank_top_k`, chunk sizes), rate limit, CORS origins, Sentry, Langfuse, and the cache tunables (`ANSWER_CACHE_TTL` / `ANSWER_CACHE_EVERGREEN_TTL`, `SEMANTIC_CACHE_ENABLED`, `SEMANTIC_SIMILARITY_THRESHOLD`, `EMBEDDING_*`). `.env.example` files document them, and every optional integration **fails inert** — a blank key means the feature is skipped, never that the app breaks. The one integration that is off even when configurable is the semantic cache, which needs *both* its flag and an embedding key; `DEPLOYMENT.md` explains why.

### Hosting
**Backend → Render** via `render.yaml` blueprint (web service + managed Postgres + managed Redis; `AUTH_SECRET` auto-generated, DB/Redis URLs auto-wired, health check path set, TinyBERT reranker to fit the 512MB Starter plan). **Frontend → Vercel** (root dir `frontend`, `NEXT_PUBLIC_API_URL` set at build). CORS connects the two.

### Production considerations
Non-root containers, health-gated readiness (returns 503 when Postgres is down), graceful shutdown (closes scraper/tavily/redis/db clients in `lifespan`), optional Sentry, CI gate before deploy, secrets never committed (gitignored `.env`), refresh cookies `Secure`/`SameSite=None` in prod, and documented cold-start behavior on free tiers.

---

## 13. Challenges

### 1. Latency — the defining challenge
The first correct pipeline took ~35s and *felt broken*. Root causes: a separate planner LLM call, a reflect-and-refine loop running 2 iterations, and search-then-scrape doing two round-trips per sub-query (often returning empty on JS-heavy pages). **Solution:** collapse the graph to one forward pass, merge routing+planning+format into a single call, switch to Tavily's one-call search+read (Serper+scrape kept as fallback), overlap follow-up generation with the answer stream, and fix async bottlenecks (semaphore-bounded scrapes, thread-pool chunking, pooled httpx, deduped sub-queries). **Result:** ~4.5s warm. **Lesson:** measure before optimizing; the "smart" agent loop was the cost, not the search.

### 2. Trustworthy citations
Early answers had `[n]` markers that didn't reliably point at the right source. **Solution:** `build_cited_context()` produces **one canonical, relevance-ordered source list** that simultaneously drives the synthesis prompt numbering, the `[n]` marker resolution (`extract_citations`), and the UI (`sources … replace:true`) — so a citation physically cannot drift from the source the model read. Then the long tail: citations inside Markdown table cells, tooltip clipping/stacking on hover. **Lesson:** make one array the single source of truth and derive everything from it.

### 3. From "runs on my laptop" to public & multi-user
The prototype assumed one local user. Going public meant **rewriting auth from scratch**: stateless JWT + refresh rotation, bcrypt, Google OAuth. A security pass made OAuth **fail closed** (verify `aud`/`iss`/`email_verified`) and added per-user rate limiting so one user can't drain the shared API budget. **Lesson:** auth and abuse-control are load-bearing, not afterthoughts.

### 4. Concurrency & cold-start races
Under load, several concurrent requests could each try to initialize the FlashRank model. **Solution:** thread-safe **double-checked locking** so the model loads exactly once; tuned the scrape pool so a single slow page can't stall the batch. **Lesson:** lazy singletons need locks under async concurrency.

### 5. Don't run the full pipeline for "hello"
Casual messages were being forced through the entire (slow, costly) research pipeline. **Solution:** a triage/router node classifies chat vs research up front and routes greetings to a fast conversational reply. **Lesson:** cheap classification up front saves the expensive path.

### 6. Deployment hardening
Shipping to Render+Vercel surfaced a fresh bug class: the async Postgres driver (`postgresql+asyncpg://` normalization), CORS for the Vercel origin, a dev-server crash loop (inotify → polling), and a `401` that dead-ended the UI (fixed with silent refresh+retry). Closed out alongside CI, deeper health checks, and custom error pages.

### 7. Optimizing blind — no per-stage latency or cost attribution
Every earlier optimization was argued from a stopwatch and a hunch: "the reflect loop *feels* like the problem." That worked at 35s and would not have worked at 4.5s. **Solution:** Langfuse tracing with a span per stage (triage, search, rerank, synthesis, cache lookups) and a contextvar-scoped **usage accumulator** that totals tokens and dollar cost per turn and ships them in the `done` event. Then the same receipt was exposed to the *user* as a **Trace tab** — what was ranked, what actually reached the model, what got cited — and a **weekly eval run** (`backend/evals/`) so a prompt change registers as a number, not a vibe. **Lesson:** you can optimize the first 80% on intuition; past that you need attribution. Note the honest scope of the evals: they measure *faithfulness and structural integrity*, not truth — an answer can be perfectly faithful to a source that is wrong.

### 8. Caching the expensive part — and the trap in doing it semantically
Search and scrape were cached, but a repeat question still paid for **both LLM calls**. Adding an exact answer cache was easy; two things about it were not.

**(a) Staleness without understanding.** The cache must be checked *before* triage (checking it after would mean paying for triage on every hit), so at lookup time nothing knows if the question is "what is quantum computing" or "bitcoin price today". The first version used one short TTL for everything, which is the correct conservative move but caps the hit rate. **Solution:** triage emits `time_sensitive` and it's stored *alongside the answer* — the flag describes the entry that exists, which is knowable at lookup time even though the incoming question's nature isn't. It defaults to `true` on any parse failure, so the failure mode is a wasted cache slot, never a stale price.

**(b) Semantic matching can serve a question the user didn't ask.** This is the real engineering story. Embeddings encode **topic, not direction**: *"best vector database"* and *"worst vector database"* score ~0.97 against each other while the correct answers are opposites. A similarity threshold alone will confidently serve one for the other, **with citations** — for a research tool that's worse than being slow. **Solution:** treat a high score as necessary but not sufficient, and gate every candidate behind four cheap deterministic rails (`utils/semantic_guards.py`): differing **numbers** ("Python 3.11" vs "3.12" barely moves a vector), differing **polarity** words (best/worst, pros/cons, isn't), reversed **`X vs Y` order**, and a floor on **shared content words**. Each returns a *reason string*, so rejections are logged and counted instead of vanishing.

Three structural limits fall out of the same reasoning: semantic matching is **first-turn only** (the exact key includes conversation history because *"how does its pricing work?"* means different things in different threads, and meaning-matching can't preserve that); a near-match to a time-sensitive answer must be **under 2 minutes old** (staleness compounds with the fact that it's not even the same question); and the bucket key hashes `user_name + model + cache version` so cosine similarity is only ever asked to judge *the question itself* — a user with a preferred name can't be served an answer addressed to someone else.

Finally, it ships **off by default**, unlike Sentry/Langfuse/Tavily-fallback. That asymmetry is the point: the other optional integrations degrade the *system* when misconfigured; this one degrades the *answer*, silently. And the UI always discloses which earlier question was reused — a fast answer and a quietly substituted one should not look the same.

**Lesson:** when a heuristic can be confidently wrong, the guard rails matter more than the model. The similarity math is 6 lines; the guards and their tests are ~330.

---

## 14. Scalability

### Current limitations
- **Single-instance assumptions in the fallback paths** (process-local rate limiter and in-memory FlashRank singleton) mean rate limiting is only globally correct with Redis, and each instance loads its own reranker.
- **`list_sessions` grabs 200 rows and groups in Python** — fine now, O(n) memory per call at scale.
- **Uploaded file bytes live in Postgres `LargeBinary`** — simple but not how you'd store blobs at scale.
- **Schema migrations are ad-hoc `ADD COLUMN IF NOT EXISTS`** — no Alembic yet.
- **External API rate/credit limits** (Tavily free ≈1k credits/month) are the real ceiling on a free tier.
- **The semantic index is brute-force over a capped bucket** (300 vectors, scored in-process on every lookup). Correct and cheap at this size, but it's an O(n) scan per lookup and it caps how much history can be matched — the ANN migration point is called out below, not glossed over.
- **Cache buckets are keyed per `(user_name, model)`**, so a popular question is re-answered once per distinct preferred name. Deliberate — it's what makes personalization leakage impossible — but it costs hit rate, and a shared bucket for non-personalized answers is the obvious next step.
- **The `time_sensitive` label is an LLM judgement**, not a measurement. It fails safe (defaults to `true`), but a mislabelled evergreen answer is a 6-hour stale window.

### How to scale to 100k users
1. **Stateless backend → horizontal autoscale** behind a load balancer (already stateless thanks to JWT); move *all* rate limiting to Redis (it's already the primary path).
2. **Managed Postgres with read replicas + connection pooler (PgBouncer)**; convert session listing to a windowed SQL query with pagination and proper composite indexes (`user_id, created_at`).
3. **Move file blobs to object storage (S3/R2)**, store only a key in Postgres.
4. **Cache layer up**: shared Redis cluster. Whole-answer caching now exists (exact + optional semantic); scaling it means splitting personalized from non-personalized buckets so popular questions are answered once globally, and moving the semantic index from an in-process brute-force scan to a real ANN index (Redis Search / pgvector / Qdrant) once buckets exceed a few thousand vectors.
5. **Isolate the reranker** as its own CPU service (or a hosted rerank API) so app instances stay light and the model loads once per pool.
6. **Queue + workers** for research runs (e.g. a task queue) if you want to decouple request lifetime from compute and add backpressure.
7. **Observability**: Sentry + tracing (`sentry_traces_sample_rate`) + structured metrics on per-stage latency and API spend.
8. **Alembic** for real migrations; blue/green deploys.

### Bottlenecks
In order: (1) **external API latency/credits** (Tavily/Groq) — the dominant cost and speed factor; (2) **the synthesis LLM stream** — inherently the longest single step; (3) **reranker CPU** on tiny instances; (4) **DB connection pool** under burst without a pooler. Each has a documented mitigation above.

---

## 15. Interview Preparation

### A. 50 Beginner Questions
1. What does this project do in one sentence?
2. What is RAG and where does it appear here?
3. Why use FastAPI instead of Flask or Django?
4. What is an ASGI server and why does it matter for streaming?
5. What is Server-Sent Events (SSE) and why use it over WebSockets here?
6. What is the difference between the access token and the refresh token?
7. Why is the access token short-lived (1 hour)?
8. Where are passwords stored and in what form?
9. What is bcrypt and why not SHA-256 for passwords?
10. What is a JWT and what's inside yours (`sub`, `email`, `iat`, `exp`)?
11. What does `Depends(get_current_user)` do in FastAPI?
12. What is Pydantic and what does it validate here?
13. What is the password policy on registration?
14. What is LangGraph at a high level?
15. What are the nodes in your research graph?
16. What is the difference between "chat" mode and "research" mode?
17. What is a sub-query and why decompose the question?
18. What does the reranker do?
19. What is chunking and why is text split into chunks?
20. What is a citation marker `[1]` and how is it resolved to a source?
21. Which LLM provider do you use and why Groq?
22. Why two different models (8B and 70B)?
23. What is Tavily and what does it return in one call?
24. What is Serper used for in this app?
25. What is Trafilatura and when does it run?
26. What database do you use and why Postgres?
27. What is Redis used for here?
28. Is Redis required for the app to work?
29. What happens if Redis is down?
30. What frontend framework and version do you use?
31. What is the App Router in Next.js?
32. What is Zustand and what do you store in it?
33. How does the frontend receive streamed tokens?
34. What is `useResearch` responsible for?
35. What is `useAuth` responsible for?
36. Where is the JWT stored on the client?
37. What are the three result tabs on a research answer?
38. What file types can be uploaded and what's the size limit?
39. How does the app handle an uploaded document differently from web results?
40. What is CORS and why do you configure it?
41. What is rate limiting and what is your limit?
42. What is the `/api/health` endpoint for?
43. What is Docker and why is the app containerized?
44. What does docker-compose bring up locally?
45. Where is the backend deployed and where is the frontend deployed?
46. What are environment variables used for here?
47. Why must `NEXT_PUBLIC_*` vars be set at build time?
48. What is the purpose of the `phase` SSE events?
49. What happens when research finds no usable sources?
50. What does the `confidence` score represent?

### B. 50 Intermediate Questions
1. Walk me through the full lifecycle of a `POST /api/research` request.
2. Why is the SSE producer decoupled from the consumer with an `asyncio.Queue`?
3. How does a node emit progress to the client without knowing about HTTP?
4. Why did you merge routing, planning, and format selection into one LLM call?
5. How does triage resolve pronouns in follow-up questions?
6. Why must every sub-query be self-contained?
7. How do you guarantee `[1]` in the answer maps to source 1 in the UI?
8. Explain `build_cited_context()` and why it's the single source of truth.
9. How does `extract_citations` avoid resolving an out-of-range `[99]`?
10. Why prepend uploaded-document chunks with `score=1.0`?
11. How does the researcher fan out sub-queries, and why not a LangGraph `Send` fan-out?
12. What reducers does `ResearchState` use and why (`_merge_lists`, `add`)?
13. Why run chunking and reranking in a thread-pool executor?
14. Explain the FlashRank double-checked locking and the race it prevents.
15. Why is the reranker a cross-encoder and not embedding cosine similarity?
16. Why is there no vector database in this RAG system?
17. How does the Tavily→Serper fallback trigger?
18. Why use per-result excerpts instead of full page content by default?
19. How does refresh-token rotation detect a stolen token?
20. Why are refresh tokens stored hashed in Redis instead of raw?
21. Why is the refresh cookie `HttpOnly` and scoped to `/api/auth`?
22. Walk through the atomic Redis rate-limit (`INCR` + `expire nx`) — why atomic?
23. Why does the local rate-limit fallback fail closed instead of open?
24. How does the frontend transparently recover from a 401 mid-research?
25. How do Effect A and Effect B on the research page avoid double-firing?
26. Why mirror streaming state in refs as well as `useState`?
27. How does `sessionsNonce` cause the sidebar to refresh?
28. How are documents handed from the home page to the research page?
29. Why is `get_settings()` wrapped in `lru_cache`?
30. How does `_normalize_db_url` make the app portable across managed hosts?
31. Why `pool_pre_ping=True` and `expire_on_commit=False` on the DB engine?
32. How does the app do schema migrations today, and what's the limitation?
33. Why does the LLM stream retry setup but not mid-stream?
34. Which HTTP statuses are retryable for Groq calls and why not 400/401?
35. How does `generate_structured` recover from malformed JSON?
36. How is prompt injection from scraped pages mitigated?
37. Why strip control characters from `preferred_name` and document text?
38. How does the synthesizer choose a table vs list vs prose?
39. How are follow-up questions generated without adding latency?
40. Why is the confidence score a heuristic and not an LLM self-rating anymore?
41. What does `format_history` do to keep prompts within budget?
42. Why are shared session links public-by-UUID, and what's the risk trade-off?
43. How does session ownership get enforced on delete?
44. What does the 5-minute `asyncio.wait_for` watchdog protect against?
45. How does the Serper path enforce source diversity?
46. Why cap per-page content at 12k chars?
47. How does the health endpoint decide 200 vs 503?
48. Why does the frontend Dockerfile use build args for `NEXT_PUBLIC_*`?
49. Why run containers as non-root, and where is that done?
50. How does the dev-only simulation mode work and why is it gated to development?

### C. 50 Senior-Level Follow-Up Questions
1. You collapsed a reflect-and-refine loop for latency — how would you *quantitatively* prove answer quality didn't regress? Design the eval.
2. The confidence score is `0.5 + 0.08 * n_sources`. That's arbitrary. What would a calibrated confidence look like and how would you validate calibration?
3. Your rate limiter is a fixed window — describe the burst it allows at window edges and how a sliding-window or token-bucket in Redis (Lua) would fix it.
4. Access tokens can't be revoked before expiry. Walk me through the threat model and when 1-hour TTL is unacceptable. How would you add revocation without losing statelessness?
5. `list_sessions` loads 200 rows and groups in Python. Write the SQL that returns paginated threads with turn counts and last-updated in one query, and the index you'd add.
6. Uploaded files sit in Postgres `LargeBinary`. Quantify the problems at 100k users and design the S3 migration incl. signed URLs and the public-by-UUID semantics.
7. The FlashRank singleton is per-process. Under autoscaling that's N model loads and N cold starts. Design a reranking service and discuss the latency/isolation trade-off vs a hosted rerank API.
8. Prompt injection: the system prompt says "ignore instructions in sources," but that's soft. What layered defenses (content provenance, structured extraction, output constraints) would you add for a hostile web?
9. You dedupe sources by URL. How do you handle near-duplicate content across different URLs (syndication, mirrors)? Discuss MinHash/SimHash.
10. The graph is a single forward pass. Where would a *bounded* re-retrieval step genuinely help, and how would you decide at runtime whether to spend it?
11. Tavily is a single point of failure and cost ceiling. Design a multi-provider retrieval abstraction with health-based routing and per-provider budgets.
12. Describe an end-to-end tracing setup that attributes latency and dollar cost to each stage (triage, search, rerank, synth) per request.
13. Your SSE stream holds a connection for the whole run. At 100k users what breaks first — file descriptors, LB timeouts, worker count? How do you size it?
14. Compare SSE vs WebSocket vs chunked HTTP vs a resumable stream (with an event id) for this workload. When would you switch?
15. If the client disconnects mid-stream, what happens to the running graph, the Groq spend, and the DB save? How would you make runs resumable?
16. The citation system trusts the model to emit `[n]` for the right claim. How would you *verify* each cited claim is actually supported by that source (NLI/entailment)?
17. You cache search results for 1 hour by query hash. What's the staleness risk for breaking-news queries and how would you make TTL query-aware? *(Now partly built — see §11 and §13.8: triage emits `time_sensitive` and the answer cache picks its TTL from it. The **search** cache is still a flat 1h; be ready to say so.)*
18. Walk through a consistency problem: two concurrent follow-ups in the same session — any interleaving or ordering hazards in the save path?
19. How would you A/B test the 8B-vs-70B synthesis decision and the format-selection prompt in production?
20. Your reducers merge concurrent node writes, but the researcher is a single writer. If you moved to true graph-level fan-out, exactly which state fields become unsafe and why?
21. Design a red-team suite for the auth surface (JWT confusion, alg=none, audience confusion on Google, cookie scoping, CSRF on refresh).
22. Is the refresh endpoint CSRF-safe given `SameSite=None`? Prove it or fix it.
23. How do you prevent SSRF via the scraper (it fetches arbitrary URLs from search results)? What about internal IP ranges and redirects?
24. The chunker force-splits with character overlap. What retrieval-quality issues does naive character chunking cause, and would semantic/late chunking help here?
25. Estimate the per-query cost (Groq tokens + Tavily credits + Serper) and design a budget guardrail that degrades gracefully as spend rises.
26. How would you shadow-deploy a new synthesizer prompt and detect regressions automatically before flipping traffic?
27. The health check returns 503 only when Postgres is down. Should Groq/Tavily outages affect readiness? Argue both sides.
28. Describe how you'd make the whole research run idempotent and safe to retry (client double-submit, LB retry, network flake).
29. What's your data-retention and privacy story for stored queries, answers, and uploaded files? GDPR delete path?
30. The synthesizer max is 2000 tokens. How do you handle genuinely long-form requests without blowing latency or truncating mid-table?
31. How would you introduce streaming *structured* output (e.g. a table) so the UI can render rows incrementally without waiting for the full stream?
32. Design caching for the *answer* (not just search) — keying, invalidation, personalization leakage risks. *(Now built — answer with the real design in §11: the key, the `_CACHE_VERSION` invalidation lever, the `user_name`-in-the-bucket-key answer to leakage, and why uploads bypass entirely.)*
33. Under a thundering herd on a popular query, how do you avoid N identical Tavily calls? (single-flight / request coalescing)
34. The reranker fallback silently returns score 0.5 for all chunks. What's the downstream failure mode and how would you detect it in prod?
35. How would you evaluate retrieval quality (recall@k of the right sources) independent of the LLM?
36. Walk me through migrating from ad-hoc `ADD COLUMN` to Alembic with zero-downtime on a live DB.
37. If you had to remove LangGraph entirely, what would you lose and what would you gain? Justify keeping or dropping it at this scale.
38. Your JWT secret is a single symmetric key (HS256). When would you move to asymmetric (RS256/ES256) and JWKS rotation?
39. The frontend keeps the token in `localStorage`. Discuss the XSS exposure vs the HttpOnly-cookie alternative and why you chose this.
40. Design multi-region deployment: where do Postgres, Redis, and the reranker live, and how do you keep p95 latency down globally?
41. How would you implement per-organization tenancy and quota without rewriting the auth layer?
42. The pipeline swallows many exceptions to "never break the run." How do you avoid silently shipping degraded answers? What SLOs and alerts?
43. Propose a schema and pipeline change to support "deep research" (10+ sources, multi-step) *without* regressing the fast path's 4.5s.
44. How do you fingerprint and block abusive automated clients that stay under the per-user hourly limit (many accounts)?
45. What's your rollback story if a bad prompt ships and answer quality tanks in production?
46. Explain how you'd add response-level guardrails (PII redaction, unsafe-content filtering) in the stream without adding a blocking round-trip.
47. The `sub_queries` are LLM-generated and unbounded in cost. How do you cap and budget the fan-out adaptively based on query complexity?
48. How would you measure and reduce hallucination rate specifically (claims not entailed by sources) as a tracked production metric?
49. Design the observability dashboard you'd want on day one of a public launch — top 8 metrics and why.
50. If p95 latency doubled overnight with no deploy, what's your triage runbook across Groq, Tavily, Redis, Postgres, and the reranker?

### D. Caching, Semantic Matching & Observability (the newest work)

These are the questions an interviewer will actually dig into, because this is the part of the system where a wrong decision produces a *confidently wrong answer* rather than an error.

**Fundamentals**
1. What exactly is stored on a cache hit — the answer text, or more? Why does replaying only the text break the UI?
2. Why is the cache consulted *before* triage rather than after?
3. What is `_CACHE_VERSION` for, and when must you bump it?
4. Why do uploaded documents bypass the answer cache entirely?
5. What does `cached: true` with an all-zero `usage` block communicate, and why not just omit the field?
6. What's the difference between an "exact" and a "semantic" cache hit from the user's point of view?
7. Why does the semantic cache ship *off* by default when Sentry and Langfuse ship *on*?
8. Why is there no vector database, given that there are now embeddings?

**Design reasoning**
9. Triage runs *after* the cache lookup, so how can the cache possibly know whether an answer is time-sensitive? (The flag describes the *stored* entry, not the incoming question.)
10. Why does `time_sensitive` default to `true` on a triage parse failure? Walk through the failure mode of defaulting the other way.
11. The prompt tells the model the flag "never changes what you research." Why does that sentence exist?
12. Why are `user_name` and the synthesis model in the *bucket* key rather than compared by similarity?
13. Why is semantic matching restricted to first turns? Give the concrete failure case.
14. Why must a semantic hit on a time-sensitive answer be far fresher (2 min) than an exact hit (15 min)?
15. Why are candidates sorted best-first and evaluated until one *passes*, instead of taking the top match and rejecting it outright?
16. Why store normalized vectors? What does that buy on the hot path?
17. Why round stored vectors to 5 decimal places?
18. `embed()` never raises and never retries. Justify both — especially "no retries" on a cache path.
19. Why does the vector index live in a Redis *hash* rather than one key per vector?
20. Explain the index TTL: why is the bucket's TTL refreshed on every write, and why is it the *longest* answer TTL rather than the shortest?
21. When a vector's answer has expired but the vector remains, what happens? Why is it cleaned up lazily rather than on a schedule?
22. Why is the index capped at 300 entries, and why prune oldest-first rather than least-recently-used?

**The guards (expect the most pressure here)**
23. Give me a pair of questions with ~0.97 cosine similarity and opposite correct answers.
24. Walk through all four rails and what each one catches.
25. Why is the token-overlap floor `0.3` and not `0.5`? (Hint: "what changed in Python 3.12" vs "what is new in Python 3.12" — a genuine paraphrase — scores 0.33.)
26. Why do the guards return a *reason string* instead of a boolean?
27. `numbers_in` treats any digit difference as a rejection. What false negatives does that cause, and why is that trade acceptable here?
28. "before"/"after" and "can"/"cannot" are in the polarity set. Is that over-broad? Defend it.
29. The guards are deterministic string matching, not a model. Why not use a small LLM to judge equivalence?
30. What's the residual failure mode — a pair that passes all four rails but should not be reused?
31. How would you *measure* the semantic cache's false-positive rate in production rather than reasoning about it?
32. `semantic_rejected` counts rejections. What would a *rising* rejection count tell you, and what would a rising rescue rate with a flat rejection count tell you?
33. If you had to raise the hit rate without touching the threshold, what would you change first?
34. How would you A/B test a threshold change safely when the failure is "a plausible but wrong answer"?
35. `math.sumprod` over mismatched-length vectors returns 0 instead of raising. Why is that the right call?
36. Suppose you switch embedding models. What breaks, and what in the design already handles it?

**Observability & evals**
37. What does a Langfuse span buy you that a log line doesn't?
38. Why is the usage accumulator a `ContextVar` rather than passed through the state?
39. Why do *cache hits* still emit a `research` observation?
40. The evals measure faithfulness and structure, not truth. Explain that distinction and why it's an honest scope rather than a cop-out.
41. Why are the eval *scorers* unit-tested on every PR while the live run happens weekly?
42. The Trace tab shows users what was ranked and sent to the model. What's the argument for exposing that, and what's the argument against?

---

## 16. Resume Validation

For each likely résumé claim, here's the supporting code and how to defend it — plus honesty flags.

**Claim: "Built a full-stack AI research agent (Perplexity-style) with cited, streaming answers."**
- *Supported by:* the whole repo — `agents/` pipeline, `research.py` SSE endpoint, `StreamingAnswer`/`ResearchTurn` frontend, `citations.py`.
- *Defend:* walk the request lifecycle in §2/§9; demo the SSE event sequence; show `build_cited_context` as the citation-integrity mechanism.
- *Flag:* it's "Perplexity-*style*," not Perplexity-scale — be precise about that.

**Claim: "Reduced research latency ~87% (≈35s → ≈4.5s)."**
- *Supported by:* the graph collapse (`graph.py` single pass, `max_iterations=1`), Tavily one-call path (`tavily.py`, `researcher.py`), concurrent follow-ups (`synthesizer.py`), thread-pool offloading, pooled httpx, semaphore-bounded scrapes.
- *Defend:* name each change and its mechanism; the README "Engineering Journey" cites specific commits.
- *Flag:* "~4.5s **warm**" (cache hits) — say warm/cold honestly; cold and free-tier cold-starts are slower. Don't claim a rigorous benchmark harness — it's measured, not a formal eval.

**Claim: "Designed a LangGraph multi-node agent (triage → research → rerank → synthesize)."**
- *Supported by:* `graph.py`, `router.py`, `researcher.py`, `synthesizer.py`, `conversational.py`, `state.py`.
- *Defend:* explain conditional routing, typed state + reducers, `astream`.
- *Flag:* it's a **single forward pass**, not a cyclic multi-agent system — the planner/reflector were *removed*. Frame that as a deliberate latency decision (a strength), not as a running reflection loop.

**Claim: "Engineered trustworthy inline citations mapping claims to exact sources."**
- *Supported by:* `citations.py` (`build_cited_context`, `extract_citations`), the `sources … replace:true` SSE contract, `CitationTooltip`, and the `test_pipeline.py` suite.
- *Defend:* one canonical relevance-ordered array drives prompt numbering, marker resolution, and UI — they can't drift. Show the unit tests for out-of-range/dedup.
- *Flag:* the system doesn't *verify* entailment (that the source truly supports the claim) — it guarantees *numbering integrity*, not factual entailment. Be honest if pressed (see senior Q16).

**Claim: "Implemented secure auth: JWT, bcrypt, Google OAuth, refresh-token rotation."**
- *Supported by:* `services/auth.py`, `routers/auth.py`, `useAuth.js`.
- *Defend:* short access token + rotating hashed refresh token in Redis; OAuth fails closed with `aud`/`iss`/`email_verified` checks; bcrypt with salt.
- *Flag:* tokens live in `localStorage` (XSS trade-off) and HS256 single-key — know the trade-offs (senior Q38/Q39).

**Claim: "Neural re-ranking with a cross-encoder for retrieval relevance."**
- *Supported by:* `services/reranker.py` (FlashRank), `config.py` model choice.
- *Defend:* cross-encoder vs bi-encoder distinction; CPU-only for cost; thread-safe singleton + executor offload.
- *Flag:* it's `TinyBERT-L-2` by default (speed over max quality) — say so.

**Claim: "Production-ready: Dockerized, CI, health checks, graceful degradation, Sentry."**
- *Supported by:* `Dockerfile`s (non-root, healthcheck), `docker-compose.yml`, `render.yaml`, `.github/workflows/ci.yml`, `/api/health`, Redis-optional fallbacks, Sentry no-op-without-DSN.
- *Defend:* show the lifespan startup/shutdown, the 503-on-DB-down readiness logic, and the fallback ladders.
- *Flag:* migrations are ad-hoc (no Alembic yet), single backend instance in practice — mention as known next steps, which reads as maturity.

**Claim: "Cut repeat-query cost to zero with a full-answer cache, with staleness bounded by an LLM-emitted freshness label."**
- *Supported by:* `services/answer_cache.py` (`build_answer_key`, `ttl_for`, `iter_replay_events`, `cached_state_for_save`), the cache branch in `routers/research.py`, `time_sensitive` in `agents/router.py` + `agents/state.py`, `cache_stats` on `/api/health`, `tests/test_semantic_cache.py`.
- *Defend:* a hit replays the *entire* SSE sequence (sources, images, trace, follow-ups), so the UI is identical — at zero LLM calls, reported as an all-zero `usage` block rather than a hidden field. Explain why the lookup must precede triage, and how storing the freshness label *with the answer* solves the "we don't know what we're about to be asked" problem.
- *Flag:* the hit rate is bucket-scoped per `(user_name, model)`, so it's lower than a global cache would be — a deliberate personalization-leakage trade. And `time_sensitive` is an LLM judgement, not a measurement.

**Claim: "Built a semantic cache that matches near-duplicate questions by meaning, with deterministic safety rails against embedding failure modes."**
- *Supported by:* `services/embeddings.py`, `utils/semantic_guards.py`, `find_similar_answer` / `_index_embedding` / `_prune_index` in `answer_cache.py`, `tests/test_semantic_guards.py` (the guard suite), the disclosure UI in `ResearchTurn.js`, and the enablement docs in `DEPLOYMENT.md`.
- *Defend:* lead with the failure mode, not the feature — embeddings encode topic, not direction, so *"best X"* and *"worst X"* score ~0.97 and a threshold alone will serve one for the other **with citations**. Then the four rails, the first-turn-only restriction, the 2-minute freshness bound on time-sensitive matches, and the fact that it ships off by default and always discloses the reused question. The similarity math is ~6 lines; the guards and their tests are ~330 — that ratio *is* the point.
- *Flag:* the `0.92` threshold is a **starting point, not a tuned value** — say so before you're asked. There is no measured production false-positive rate yet (see §15.D, Q31). Brute-force scan over ≤300 vectors, not an ANN index.

**Claim: "Instrumented the pipeline with per-stage tracing, per-turn cost accounting, and a scored eval set in CI."**
- *Supported by:* `services/tracing.py` (Langfuse spans per stage incl. both cache lookups), `services/usage.py` (ContextVar accumulator → `usage` in the `done` event), `backend/evals/` (`queries.yaml`, pure `scorers.py`, `run_eval.py`, `baseline.json`), `.github/workflows/eval.yml`, and the Trace tab in the frontend.
- *Defend:* the honest framing — the first 80% of the latency win was won on intuition, and this exists because the next 20% can't be. Note the split between cheap deterministic scorers (tested every PR, no API keys) and the expensive live run (weekly).
- *Flag:* the evals measure **faithfulness and structural integrity, not truth** — an answer can be perfectly faithful to a source that's wrong. Say this unprompted; claiming "answer-quality evals" without it is the overclaim.

**Claim: "Per-user rate limiting to control shared API spend."**
- *Supported by:* `check_rate_limit` (atomic Redis `INCR`), the process-local fail-closed fallback, `/api/auth/rate-limit`, and the sidebar usage bar.
- *Defend:* explain atomicity and the fail-closed design; each query costs real money.
- *Flag:* fixed-window (edge bursts) and per-account (multi-account abuse) — know the mitigations (senior Q3/Q44).

**Claim: "Document Q&A over uploaded PDFs/DOCX."**
- *Supported by:* `file_processor.py`, `upload.py`, `files.py`, `_build_doc_sources`, doc-only path in `researcher.py`, `DocumentViewer`.
- *Defend:* docs become score-1.0 prepended sources; triage's `needs_web` decides web augmentation; viewer re-renders from stored bytes.
- *Flag:* text is capped (12–16k chars) — not full long-document RAG; be clear it's bounded.

**Hard-to-defend claims to AVOID or reframe:**
- ❌ "Multi-agent system with reflection/self-critique" → the reflector was removed; say "collapsed a reflect loop for an 87% latency win."
- ⚠️ "Vector database / semantic search" → still **no vector DB, and no embedding retrieval**. Embeddings now exist in exactly one place: matching a *question to a previously asked question* for the cache, brute-force scored over a capped bucket. Reframe as "web-RAG without an index, plus embedding-based cache matching" — and be ready to state the query-to-query vs query-to-document distinction crisply, because it's the first thing a sharp interviewer will probe.
- ⚠️ "Rigorous automated answer-quality evaluation" → there **is** now a scored eval set running weekly in CI, but it measures **faithfulness and structural integrity, not truth**. Claim "an automated faithfulness/structure eval with a tracked baseline," never "answer-quality evals" unqualified.
- ❌ "Semantic caching improved hit rate by X%" → no measured production number exists; the feature is off by default and the threshold is untuned. Talk about the *design and its failure modes*, not a metric you don't have.
- ❌ "Horizontally scaled / handles X thousand users" → it's deployed and multi-user-capable but not load-proven; talk about the *path* to scale (§14), not a demonstrated number.

---

*Prepared from a full read of the repository. Trust the code over `implementation_plan.md`, which describes an earlier Ollama/DuckDuckGo/reflect-loop design that was superseded.*
