# Production operations and release gates

## Supported deployment boundary

One backend process, one embedded Chroma store, one shared corpus. Do not add Uvicorn workers or backend replicas: ingestion serialization and evaluation admission are process-local. BM25 loads the corpus per query and is capped at 20,000 chunks. Move to an external retrieval service and durable job queue before scaling beyond this boundary.

The bundled production proxy protects the entire UI with Basic Auth, rate-limits API requests, injects the backend credential, and streams SSE without buffering. It binds to loopback; terminate TLS with your ingress before exposing it remotely. Never expose the backend port directly or place the shared API key in a Vite build variable. Only authenticated, trusted operators should upload documents. PDF parsing is bounded by file size, page count, and extracted text, but is not sandboxed against parser-level CPU exhaustion.

## Release blockers discovered during hardening

The installed dependency audit still reports advisories without listed fixes for Chroma, DiskCache, and Ragas. CI deliberately keeps `pip-audit` as a failing release gate until these are remediated or formally assessed against the deployment. Do not interpret passing functional tests as a security approval. Details and exact current package versions are in `backend/constraints.txt`; rerun the audit before release.

- Chroma: PYSEC-2026-311, PYSEC-2026-3813, PYSEC-2026-3814, PYSEC-2026-3815.
- DiskCache: PYSEC-2026-2447.
- Ragas: PYSEC-2026-3046.

Docker was unavailable in the implementation environment. Container definitions were statically validated; the GitHub container build job and a staging startup/health/streaming test must pass before release. Live Groq evaluation and provider quotas also require a staging run with operator-supplied credentials.

The three PPCM test cases include broad expected-answer descriptions. They are a demo benchmark, not a clinically adjudicated gold standard. No clinical correctness, regulatory compliance, multi-tenant isolation, or high-availability claim is made.

## Storage and migration

The canonical data directory contains `chroma/`, optional `uploads/`, and `evaluations/`. Native runs default to `backend/data`; containers use `/data`. The original Compose mounts used `/app/backend/...`, while the application wrote under `/app/...`. Inspect the old container's actual directories before replacing it. Copy out and back up existing data first.

For an old native installation, stop all writers, back up `backend/chroma_data` and `backend/uploads`, then migrate their contents into `backend/data/chroma` and `backend/data/uploads`. Chroma has moved from 0.5 to 1.x: test migration on a disposable copy and validate retrieval and chunk counts before switching. Keep the untouched backup; do not downgrade a migrated store in place. Re-ingestion from original documents is the fallback if store migration fails.

The image runs as UID 10001. Bind mounts restored from another machine must be writable by that UID. Named volumes are initialized from image directory ownership.

## Backup and restore

Stop writes during backup. From the repository root with production environment variables loaded:

```bash
mkdir -p backups
docker compose -f docker-compose.prod.yml stop backend
docker compose -f docker-compose.prod.yml run --rm --no-deps --user root \
  -v "$PWD/backups:/backup" --entrypoint tar backend -czf /backup/medrag-backup.tgz -C /data .
docker compose -f docker-compose.prod.yml start backend
```

Encrypt and copy the archive to your backup storage with a documented retention policy. Test restoration to a fresh volume in a separate Compose project. With that project's backend stopped, mount its fresh data volume and extract the archive into `/data`, set ownership to UID 10001, then start it and compare `/health` counts and retrieval results. Never restore over a running database. Preserve the previous image and matching data snapshot for rollback.

## Operations

- `/live` checks the process; `/health` checks the vector store. Startup completes only after the embedding model warms up. Pre-populate the model cache for restricted-network deployments.
- Watch 5xx/503 rates, failed SSE events, evaluation duration, disk usage, memory, provider token consumption, and report growth. `X-Request-ID` identifies unexpected API failures.
- A provider timeout or partial stream yields an explicit failure; the UI does not present a truncated answer as complete. Retrying may incur another provider charge.
- Evaluation is synchronous to the HTTP request, and only one runs at a time. Reports survive restarts; in-flight jobs do not. If workloads exceed proxy timeouts, implement a durable queue and polling API.
- Stop the service before pruning the corpus. There is no per-document deletion API; retention is currently an operator responsibility. Raw originals are disabled by default, but Chroma and CSVs contain extracted text.
- Rotate browser passwords with `htpasswd`; rotate the shared backend key by recreating both services with the new environment. Protect Compose environment files and Docker daemon access.
- Keep dependency audits enabled. Do not add broad advisory suppressions to obtain a green build.
