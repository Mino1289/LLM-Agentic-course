# Finance RAG Hub-and-Spoke — Next.js UI

Production UI for the Finance RAG multi-agent system. Talks to the FastAPI backend via `/api/*` (proxied in dev).

## Run locally

Terminal 1 — API (from repo root):

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --reload-dir api --reload-dir src --port 8000
```

Terminal 2 — frontend:

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) (French) or [http://localhost:3000/en](http://localhost:3000/en).

The Next.js dev server proxies `/api/*` to `http://127.0.0.1:8000` (see `next.config.ts`).

## Alternate ports

If ports 8000 or 3000 are busy, use custom ports and configure `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8080
API_PROXY_TARGET=http://127.0.0.1:8080
PORT=3030
```

```bash
# API on 8080 (repo root)
uvicorn api.main:app --reload --reload-dir api --reload-dir src --port 8080

# UI on 3030
npm run dev -- -p 3030
```

Add to root `.env` for CORS:

```env
CORS_ORIGINS=http://localhost:3030,http://127.0.0.1:3030
```

Restart both servers after changing env files.

## Stack

- Next.js 16 (App Router), Tailwind CSS 4, next-intl (fr/en)
- FastAPI backend (`api/`) with SSE streaming from `HubAndSpokeGraph`

## API endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/health` | Readiness |
| `GET /api/config` | LLM models + default agent settings |
| `GET /api/tools` | Tool definitions |
| `POST /api/conversations` | New chat |
| `GET /api/conversations` | Sidebar history |
| `POST /api/chat/stream` | SSE chat stream |
| `POST /api/chat/resume` | Trade approve/cancel |
| `GET /api/reports/{filename}` | Download generated reports |

## Docker

From repo root:

```bash
docker compose run --rm bootstrap   # index RAG data (first time)
docker compose up finance-rag-api finance-rag-web
```

- Web: http://localhost:3000
- API: http://localhost:8000

## Scripts

```bash
npm run dev      # Development
npm run build    # Production build
npm run start    # Production server
npm run lint     # ESLint
```
