"""Bounded HTTP interface to the shared-corpus MedRAG pipeline."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app import settings
from app.limits import BodyLimitMiddleware
from app.rag_core import (
    collection_count,
    generate_grounded_response,
    ingest_document,
    persist_upload_copy,
    retrieve_context,
    stream_grounded_response,
    warmup_embedding_model,
)

logger = logging.getLogger(__name__)


def run_medrag_evaluation(**kwargs):
    from app.eval_core import run_medrag_evaluation as evaluate

    return evaluate(**kwargs)


def query_local_pipeline(question):
    from app.eval_core import query_local_pipeline as query

    return query(question)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=8000)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class LegacyChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(max_length=16000)


class LegacyChatRequest(BaseModel):
    messages: list[LegacyChatMessage] = Field(min_length=1, max_length=100)


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_csv: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,80}\.csv$"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_production()
    app.state.active = 0
    app.state.evaluation_running = False
    await warmup_embedding_model()
    yield


app = FastAPI(title="MedRAG API", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.add_middleware(BodyLimitMiddleware)


@app.middleware("http")
async def protect_requests(request: Request, call_next):
    request_id = uuid4().hex
    if (
        request.url.path not in {"/health", "/live", "/version"}
        and request.method != "OPTIONS"
    ):
        if settings.API_KEY and not hmac.compare_digest(
            request.headers.get("authorization", "").encode(),
            f"Bearer {settings.API_KEY}".encode(),
        ):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error("Request failed id=%s type=%s", request_id, type(exc).__name__)
        response = JSONResponse({"detail": "Internal server error"}, status_code=500)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    return JSONResponse(
        {
            "detail": [
                {"loc": e["loc"], "msg": e["msg"], "type": e["type"]}
                for e in exc.errors()
            ]
        },
        status_code=422,
    )


@asynccontextmanager
async def capacity():
    active = getattr(app.state, "active", 0)
    if active >= settings.MAX_CONCURRENT_REQUESTS:
        raise HTTPException(
            503, "Server busy; retry later", headers={"Retry-After": "5"}
        )
    app.state.active = active + 1
    try:
        yield
    finally:
        app.state.active -= 1


@app.get("/live")
def live():
    return {"status": "ok"}


@app.get("/health")
def health():
    try:
        return {"status": "ok", "collection_count": collection_count()}
    except Exception:
        raise HTTPException(503, "Vector store unavailable") from None


@app.get("/version")
def version():
    return {"version": os.getenv("MEDRAG_VERSION", app.version)}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    return await _ingest_uploaded_file(file, pdf_only=True)


@app.post("/api/ingest/upload")
async def ingest_compat(file: UploadFile = File(...)):
    return await _ingest_uploaded_file(file, pdf_only=False)


async def _ingest_uploaded_file(file: UploadFile, *, pdf_only: bool):
    name = file.filename or ""
    try:
        if not name or "/" in name or "\\" in name or len(name) > 200:
            raise HTTPException(400, "Invalid filename")
        if not name.lower().endswith(
            (".pdf",) if pdf_only else (".pdf", ".txt", ".md")
        ):
            raise HTTPException(415, "Unsupported file type")
        raw = await file.read(settings.MAX_UPLOAD_BYTES + 1)
        if len(raw) > settings.MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large")
        if not raw:
            raise HTTPException(400, "Empty file")
        async with capacity():
            result = await asyncio.to_thread(ingest_document, raw, name)
            stored = await asyncio.to_thread(persist_upload_copy, raw, name)
        return {**result, "stored_file": stored}
    except ValueError:
        raise HTTPException(400, "Invalid document or no extractable text") from None
    finally:
        await file.close()


@app.post("/chat")
async def chat(body: ChatRequest):
    async with capacity():
        contexts = await asyncio.to_thread(retrieve_context, body.query, k=6)
        response = await asyncio.to_thread(
            generate_grounded_response, body.query, contexts
        )
    return {"response": response, "sources": contexts}


@app.post("/api/evaluation/run")
async def run_evaluation(body: EvaluationRequest | None = None):
    if getattr(app.state, "evaluation_running", False):
        raise HTTPException(409, "Evaluation already running")
    async with capacity():
        app.state.evaluation_running = True
        try:
            settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            stem = body.output_csv[:-4] if body and body.output_csv else "evaluation"
            output = settings.OUTPUT_DIR / f"{stem}-{uuid4().hex}.csv"
            result = await asyncio.to_thread(
                run_medrag_evaluation,
                question_runner=query_local_pipeline,
                output_csv=output,
            )
            result["output_csv"] = output.name
            return result
        finally:
            app.state.evaluation_running = False


@app.post("/api/chat/stream")
async def chat_stream_compat(body: LegacyChatRequest, request: Request):
    user_parts = [m.content for m in body.messages if m.role == "user"]
    query = user_parts[-1].strip() if user_parts else ""
    if not query or len(query) > 8000:
        raise HTTPException(400, "A user query of 1-8000 characters is required")

    async def event_stream():
        try:
            async with capacity():
                contexts = await asyncio.to_thread(retrieve_context, query, k=6)
                yield f"data: {json.dumps({'type': 'sources', 'sources': contexts})}\n\n"
                async for piece in stream_grounded_response(query, contexts):
                    if await request.is_disconnected():
                        return
                    yield f"data: {json.dumps({'type': 'token', 'token': piece})}\n\n"
                yield 'data: {"type": "done"}\n\n'
        except Exception as exc:
            logger.error("Stream failed type=%s", type(exc).__name__)
            yield 'data: {"type": "error", "message": "Generation failed; retry later"}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )
