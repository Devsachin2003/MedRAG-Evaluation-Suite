# MedRAG Evaluation Suite

A FastAPI and React application for evaluating retrieval-augmented answers against a shared medical-document corpus. Upload PDF, TXT, or Markdown documents, retrieve evidence using Chroma vector search plus BM25, generate grounded answers with Groq, and score a PPCM benchmark with Ragas.

This is a research evaluation tool. The bundled three-case benchmark is a demonstration, not clinical validation. See [production operations](docs/production.md) for the deployment boundary and outstanding release gates.

## Architecture

```text
Browser -> authenticated Nginx -> FastAPI
                                 |-> bounded document parsing -> Chroma
                                 |-> vector + BM25 retrieval -> Groq -> SSE
                                 |-> benchmark -> Ragas -> atomic CSV reports
```

The backend uses one process and one shared corpus. Ingestion is serialized and content-hash deduplicated. Blocking parsing, retrieval, and evaluation run off the HTTP event loop. The API limits request size and concurrent pipeline work. Production authentication uses browser Basic Auth at Nginx and a separate bearer credential between Nginx and the private backend.

## Local development

Use Python 3.11 and Node 22, or Docker Compose v2. Set your Groq key in a root `.env` for Compose; for native Python use `backend/.env` or exported environment variables. Never commit live secrets.

```bash
cp .env.example .env
# Set GROQ_API_KEY in .env
docker compose up --build
```

Open http://localhost:5173. The local API is at http://127.0.0.1:8000, with interactive docs at `/docs`. Initial startup downloads the embedding model and fails if warmup cannot complete. Local ports bind to loopback.

For native development:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

## API

When `MEDRAG_API_KEY` is configured, send `Authorization: Bearer <key>` to protected backend routes. The production browser receives no backend key; Nginx injects it after authenticating the user.

```bash
curl -F 'file=@guideline.pdf' http://127.0.0.1:8000/api/ingest/upload
curl -H 'Content-Type: application/json' \
  -d '{"query":"What limitations does the guideline describe?"}' \
  http://127.0.0.1:8000/chat
curl -N -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Summarize the uploaded evidence."}]}' \
  http://127.0.0.1:8000/api/chat/stream
curl -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8000/api/evaluation/run
```

- `/live`: process liveness; `/health`: vector-store readiness; `/version`: release version.
- `/ingest`: PDF-only compatibility endpoint. `/api/ingest/upload`: PDF/TXT/MD.
- `/chat`: bounded query, answer, and evidence sources. No evidence yields an explicit insufficient-evidence answer without calling Groq.
- `/api/chat/stream`: emits `sources`, `token`, then `done`; a failed stream emits `error` instead of `done`. It uses the last user question, not conversational memory.
- `/api/evaluation/run`: one evaluation at a time; optional `output_csv` must be a safe `.csv` basename. Each report gets a unique suffix. The response contains scores, valid-score counts, model/version/corpus hashes, and a report basename.

Validation failures return 422, invalid documents 400, oversized requests 413, unsupported uploads 415, authentication failures 401, overlapping evaluations 409, and capacity exhaustion 503. Streaming errors occur inside the SSE response after headers have been sent. Unexpected exceptions return generic errors with a request ID; application logs omit query text and document snippets.

## Evaluation integrity

Metrics are context precision, faithfulness, and answer relevancy. Scores outside `[0, 1]` and non-finite scores become unavailable (`null` in JSON, `N/A` in the UI). Averages exclude unavailable scores and the response reports valid-score counts. A missing metric column or row-count mismatch fails evaluation. CSV writes use atomic replacement, and spreadsheet formula prefixes are escaped in exports.

Run the same harness from the CLI:

```bash
python evaluate_rag.py --api-url http://127.0.0.1:8000 --output-csv report.csv
```

The CLI path is operator-controlled; only HTTP output paths are restricted. Scores depend on the selected model and corpus. Replace the demonstration cases with independently reviewed references before using scores as release criteria. Retrieved text is sent to Groq for generation and evaluation.

## Tests and CI

```bash
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
python -m pytest backend/tests -q
ruff check backend --select E9,F63,F7,F82
pip check
pip-audit --progress-spinner off
cd frontend
npm ci
npm run build
npm run lint
npm audit
```

Tests use temporary Chroma databases and deterministic embeddings; no Groq credentials or embedding downloads are needed. They cover API validation/authentication, request limits, safe errors, SSE failures, ingestion deduplication/concurrency, and evaluation output integrity. They do not establish live-model answer quality.

GitHub Actions runs Python tests, frontend build/lint, dependency audits, and container builds. The audit gate deliberately fails for unresolved upstream advisories. Python versions are constrained by `backend/constraints.txt`; npm uses the committed lockfile. Update and re-audit both before dependency changes.

## Production deployment

See [the runbook](docs/production.md) for setup, migration, backups, recovery, and release blockers. The standalone `docker-compose.prod.yml` builds a non-root backend and a static Nginx frontend, persists `/data` and the embedding cache, and exposes only `127.0.0.1:8080`. Put a TLS ingress in front before remote access.

```bash
mkdir -p deploy
htpasswd -cB deploy/medrag.htpasswd operator
export MEDRAG_API_KEY="$(openssl rand -hex 32)"
export GROQ_API_KEY="your-provider-key"
docker compose -f docker-compose.prod.yml up --build -d
docker compose -f docker-compose.prod.yml ps
```

Use a secret manager or protected environment file to preserve deployment credentials between invocations. Docker Compose stores data in named volumes; `down` preserves them, while `down -v` deletes them.

## Configuration

| Variable | Default / purpose |
| --- | --- |
| `APP_ENV` | `development`; `production` validates credentials at startup |
| `MEDRAG_API_KEY` | Optional locally; production requires 32-128 URL-safe characters |
| `GROQ_API_KEY`, `GROQ_MODEL` | Provider credential and model; default `llama-3.1-8b-instant` |
| `MEDRAG_DATA_DIR` | Native `backend/data`; container `/data` |
| `RETAIN_UPLOADS` | `false`; opt in to storing original uploaded bytes |
| `MAX_UPLOAD_BYTES` | 10 MiB; bounded reads and total-body middleware |
| `MAX_TEXT_CHARS`, `MAX_PDF_PAGES` | 2,000,000 extracted characters / 500 pages |
| `MAX_CORPUS_CHUNKS` | 20,000; bounds in-memory BM25 corpus loading |
| `MAX_CONCURRENT_REQUESTS` | 4 pipeline requests per worker |
| `LLM_TIMEOUT_SECONDS` | 60 per provider request; one retry, 2048 output tokens |
| `CORS_ORIGINS` | Local Vite origins; comma-separated explicit origins |

Raw upload retention being disabled does not remove extracted document text from Chroma or reports. Treat both stores as sensitive. Authentication is shared-corpus access control, not per-user document isolation.
