# Deploying goon.ai (Vercel + Render)

Frontend → **Vercel**. Backend + Postgres + Redis → **Render** (via `render.yaml`).

The two need each other's URLs, so deploy in this order: **backend first**, then
frontend, then come back and set the backend's CORS to the frontend URL.

---

## 1. Backend on Render

1. Push this repo to GitHub (already done).
2. Render dashboard → **New + → Blueprint** → select this repo. Render reads
   `render.yaml` and creates: `goon-backend` (web), `goon-db` (Postgres),
   `goon-redis` (Redis).
3. When prompted, fill the secrets:
   - `GROQ_API_KEY` — from https://console.groq.com
   - `SERPER_API_KEY` — from https://serper.dev
   - `GOOGLE_CLIENT_ID` — optional (leave blank to hide the Google button)
   - `CORS_ORIGINS` — leave as `[]` for now; you'll set it in step 3.
   - `AUTH_SECRET` is generated automatically. `DATABASE_URL` / `REDIS_URL` are
     wired automatically.
4. Click **Apply**. First build is slow (installs FlashRank/onnxruntime).
5. When live, note the URL: `https://goon-backend.onrender.com`. Check
   `https://goon-backend.onrender.com/api/health` → should return `healthy`.

> **Memory note:** the blueprint uses the lightweight reranker
> (`ms-marco-TinyBERT-L-2-v2`) so it fits the 512MB *Starter* plan. If you move
> to a ≥2GB plan, set `RERANKER_MODEL=ms-marco-MiniLM-L-12-v2` for better ranking.

> **Free Postgres note:** Render's free Postgres is time-limited. For something
> longer-lived, create a free DB on [Neon](https://neon.tech) and paste its
> connection string into `DATABASE_URL` instead (the app normalizes the driver
> automatically). Use the **internal** Render DB URL (no `sslmode`) for asyncpg.

---

## 2. Frontend on Vercel

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

## 3. Connect them (CORS)

1. Back in Render → `goon-backend` → **Environment** → set:
   ```
   CORS_ORIGINS=["https://your-app.vercel.app"]
   ```
   (JSON array. Add more origins comma-separated inside the brackets if needed.)
2. Save — Render redeploys. Done: open the Vercel URL and run a query.

---

## 4. Google OAuth (only if using it)

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
| Backend OOM / restart loop on first query | Reranker model too big for the plan — keep `RERANKER_MODEL=ms-marco-TinyBERT-L-2-v2` or bump the plan. |
| Google button missing | `NEXT_PUBLIC_GOOGLE_CLIENT_ID` unset (expected if you're not using Google). |
| First request after idle is slow | Render free/Starter spins down or is cold; the first hit warms it. |
