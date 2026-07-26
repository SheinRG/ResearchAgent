# Deploying goon.ai (Vercel + Render)

Frontend → **Vercel**. Backend + Redis → **Render** (via `render.yaml`).
Postgres → **Supabase** (Render's free Postgres expires after 30 days).

The two need each other's URLs, so deploy in this order: **backend first**, then
frontend, then come back and set the backend's CORS to the frontend URL.

---

## 1. Database on Supabase

Do this first — the backend needs the connection string.

1. Create a free project at https://supabase.com (500 MB, no expiry).
2. **Connect** → **Session pooler** → copy the URI. It looks like:
   ```
   postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
   ```
3. Substitute your database password for `[YOUR-PASSWORD]`.

Pick the region closest to your Render region to keep query latency down.

> **Use the session pooler, not the other two options.** The direct connection
> (`db.<ref>.supabase.co`) is IPv6-only and unreachable from Render, which fails
> as a confusing connection timeout. The transaction pooler (port `6543`) is
> incompatible with asyncpg's prepared-statement cache. The session pooler on
> port `5432` is IPv4 and works as-is — the app strips `sslmode` from the URL
> and enables TLS for asyncpg automatically.

No schema setup is needed; the backend creates its tables on boot.

---

## 2. Backend on Render

1. Push this repo to GitHub (already done).
2. Render dashboard → **New + → Blueprint** → select this repo. Render reads
   `render.yaml` and creates: `goon-backend` (web) and `goon-redis` (Redis).
3. When prompted, fill the secrets:
   - `GROQ_API_KEY` — from https://console.groq.com
   - `SERPER_API_KEY` — from https://serper.dev (powers the Images tab + fallback search)
   - `TAVILY_API_KEY` — from https://app.tavily.com (primary search+read; the
     speed/quality win depends on this being set. Free tier ≈ 1,000 credits/month,
     ~4 per query. If unset, the app falls back to Serper search + scraping.)
   - `GOOGLE_CLIENT_ID` — optional (leave blank to hide the Google button)
   - `CORS_ORIGINS` — leave as `[]` for now; you'll set it in step 4.
   - `DATABASE_URL` — the Supabase session-pooler URI from step 1.
   - `AUTH_SECRET` is generated automatically; `REDIS_URL` is wired automatically.
4. Click **Apply**. First build is slow (installs FlashRank/onnxruntime).
5. When live, note the URL: `https://goon-backend.onrender.com`. Check
   `https://goon-backend.onrender.com/api/health` → should return `healthy`.

> **Memory note:** the blueprint uses the lightweight reranker
> (`ms-marco-TinyBERT-L-2-v2`) so it fits the 512MB *Starter* plan. If you move
> to a ≥2GB plan, set `RERANKER_MODEL=ms-marco-MiniLM-L-12-v2` for better ranking.

> **Keeping the database awake:** Supabase pauses free projects after 7 days
> with no activity. `.github/workflows/keepalive.yml` pings `/api/health` (which
> runs a real query) every 3 days to prevent this. Set the repo variable
> `BACKEND_HEALTH_URL` to `https://goon-backend.onrender.com/api/health` under
> **Settings → Secrets and variables → Actions → Variables**, or the job skips
> with a warning. If the project does pause, no data is lost — restore it from
> the Supabase dashboard.

---

## 3. Frontend on Vercel

1. Vercel → **Add New → Project** → import this repo.
2. **Root Directory:** set to `frontend` (important — the Next.js app isn't at
   the repo root).
3. Framework preset: **Next.js** (auto-detected). Leave build/output defaults.
4. **Environment Variables** (these are inlined at build time):
   - `NEXT_PUBLIC_API_URL` = `https://goon-backend.onrender.com` (your step‑1 URL,
     no trailing slash)
   - `NEXT_PUBLIC_GOOGLE_CLIENT_ID` = your Google client ID (optional)
5. **Deploy.** Note the URL: `https://your-app.vercel.app`.

---

## 4. Connect them (CORS)

1. Back in Render → `goon-backend` → **Environment** → set:
   ```
   CORS_ORIGINS=["https://your-app.vercel.app"]
   ```
   (JSON array. Add more origins comma-separated inside the brackets if needed.)
2. Save — Render redeploys. Done: open the Vercel URL and run a query.

---

## 5. Google OAuth (only if using it)

In [Google Cloud Console](https://console.cloud.google.com) → your OAuth client:
- **Authorized JavaScript origins:** add `https://your-app.vercel.app`
- The same client ID must be set as `GOOGLE_CLIENT_ID` (backend) and
  `NEXT_PUBLIC_GOOGLE_CLIENT_ID` (frontend).

Email/password works without any of this.

---

## Smoke test

```bash
curl https://goon-backend.onrender.com/api/health
# {"status":"healthy","llm":"connected","model":"llama-3.1-8b-instant"}
```

Then on the live site: register (password needs 8+ chars, 1 uppercase, 1 number)
and run a research query.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Frontend loads but every query fails / CORS error in console | `CORS_ORIGINS` missing the exact Vercel URL, or `NEXT_PUBLIC_API_URL` wrong. Both must match, no trailing slash. |
| Queries hit `http://localhost:8000` in prod | `NEXT_PUBLIC_API_URL` wasn't set at **build** time — set it in Vercel and redeploy. |
| Backend boot crash: "dialect requires an async driver" | Should be auto-handled; ensure `DATABASE_URL` is a normal `postgres(ql)://` URL. |
| DB connection times out on Render, works locally | You used Supabase's **direct** connection (`db.<ref>.supabase.co`), which is IPv6-only. Switch to the session pooler URI (`...pooler.supabase.com:5432`). |
| `connect() got an unexpected keyword argument 'sslmode'` | An older build; `_normalize_db_url` strips libpq-only params. Redeploy from `main`. |
| Errors mentioning prepared statements / `DuplicatePreparedStatementError` | You used the transaction pooler on port `6543`. Use port `5432` (session mode). |
| App suddenly 503s with `"database": "disconnected"` after a quiet week | Supabase paused the free project. Restore it in the dashboard and set `BACKEND_HEALTH_URL` so the keepalive workflow runs. |
| Backend OOM / restart loop on first query | Reranker model too big for the plan — keep `RERANKER_MODEL=ms-marco-TinyBERT-L-2-v2` or bump the plan. |
| Google button missing | `NEXT_PUBLIC_GOOGLE_CLIENT_ID` unset (expected if you're not using Google). |
| First request after idle is slow | Render free/Starter spins down or is cold; the first hit warms it. |
