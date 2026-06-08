# 🔬 AI Research Agent

> A Perplexity AI clone — autonomous research agent that searches the web, reads sources, and delivers comprehensive cited answers. **100% free, 100% local, zero API keys.**

## ✨ Features

- **🤖 Agentic Research Loop** — Decomposes questions → parallel web search → scrape & rank → synthesize cited answers → reflect & refine
- **🔍 Web Search** — DuckDuckGo search (no API key needed)
- **📖 Smart Extraction** — Trafilatura for gold-standard content extraction
- **⚡ Re-ranking** — FlashRank CPU-only neural re-ranker
- **📝 Cited Answers** — Markdown answers with clickable [1], [2] citation badges
- **🎯 Self-Reflection** — Confidence scoring + gap analysis with automatic refinement loops
- **🌊 Real-time Streaming** — SSE-based token streaming for live answer generation
- **🎨 Premium UI** — Glassmorphism, animations, dark/light mode
- **💾 Session History** — PostgreSQL persistence + Redis caching
- **🏠 100% Local** — Ollama LLM runs on your machine, no data leaves your network

## 🏗️ Architecture

```
User Query → Planner (LLM) → 2-4 Sub-queries
  → Parallel Researchers (Search → Scrape → Chunk → Rerank)
    → Synthesizer (LLM, streaming) → Cited Answer
      → Reflector (LLM) → Loop or Finalize
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 15, Vanilla CSS, Motion, Zustand |
| **Backend** | Python 3.12, FastAPI, LangGraph |
| **LLM** | Ollama (Llama 3.1 8B) |
| **Search** | DuckDuckGo (no API key) |
| **Extraction** | Trafilatura |
| **Re-ranking** | FlashRank (CPU-only) |
| **Database** | PostgreSQL 16 |
| **Cache** | Redis 7 |
| **Infrastructure** | Docker Compose |

## 🚀 Quick Start

### Prerequisites

1. **Docker Desktop** — [Install Docker](https://docs.docker.com/desktop/)
2. **Ollama** — [Install Ollama](https://ollama.com)
3. **Node.js 20+** — [Install Node.js](https://nodejs.org)

### Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd ai-research-agent

# 2. Pull the Ollama model
ollama pull llama3.1:8b

# 3. Verify Ollama is running
curl http://localhost:11434/api/tags

# 4. Start infrastructure (Redis + Postgres + Backend)
docker compose up -d

# 5. Install frontend dependencies
cd frontend
npm install

# 6. Start the frontend
npm run dev
```

### Access

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/health

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/research` | Start research (SSE stream) |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/sessions` | List recent sessions |
| `GET` | `/api/sessions/{id}` | Get session details |

### SSE Events

```
event: phase        → {"phase": "planning", "message": "Breaking down..."}
event: sub_queries  → {"queries": ["q1", "q2", "q3"]}
event: sources      → {"sources": [{url, title, domain, favicon, snippet}]}
event: token        → {"token": "word"}
event: follow_up    → {"suggestions": ["question1", "question2"]}
event: done         → {"session_id": "...", "confidence": 0.89}
```

## 🔧 Configuration

Copy `.env.example` to `.env` and adjust the variables:

| Variable | Description | Default / Example |
|----------|-------------|-------------------|
| `GROQ_API_KEY` | API key for Groq Cloud LLM | `your-groq-api-key-here` |
| `GROQ_MODEL` | Groq LLM model identifier | `llama-3.1-8b-instant` |
| `SERPER_API_KEY` | API key for Serper search API | `your-serper-api-key-here` |
| `AUTH_SECRET` | Secret key used for session authentication | `change-me-in-production-use-a-random-string` |
| `GOOGLE_CLIENT_ID` | Client ID for Google OAuth integration | `your-google-client-id-here` |
| `OLLAMA_MODEL` | Ollama model override (for local model setups) | `llama3.2:3b` |


## 📁 Project Structure

```
├── backend/
│   ├── app/
│   │   ├── agents/       # LangGraph nodes (planner, researcher, synthesizer, reflector)
│   │   ├── services/     # Ollama, DuckDuckGo, Trafilatura, FlashRank, Redis
│   │   ├── models/       # Pydantic schemas, SQLAlchemy models
│   │   ├── utils/        # Text chunking, citation extraction
│   │   └── main.py       # FastAPI app with SSE endpoints
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── app/          # Next.js pages (home, research)
│       ├── components/   # SearchBar, SourceCards, StreamingAnswer, etc.
│       ├── hooks/        # useResearch (SSE hook)
│       └── stores/       # Zustand (recent searches)
├── docker-compose.yml    # Redis + Postgres + Backend
└── README.md
```

## 📝 License

MIT
