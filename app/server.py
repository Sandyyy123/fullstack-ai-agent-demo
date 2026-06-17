"""FastAPI backend: typed endpoints for chat, RAG ingest, streaming, and health.

Run:  uvicorn app.server:app --reload   (or: python main.py)
Open: http://localhost:8000  for the bundled frontend.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .agent import run_agent
from .llm_router import detect_mode, estimate_tokens, generate
from .rag import build_seeded_store
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    RetrievedDoc,
)

app = FastAPI(title="Full-Stack AI Agent Demo", version="1.0.0")

# CORS so a separately-hosted frontend (Next.js dev server) can call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STORE = build_seeded_store()
FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok", llm_mode=detect_mode(), documents_indexed=len(STORE)
    )


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    STORE.upsert(req.doc_id, req.text)
    return {"ok": True, "documents_indexed": len(STORE)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    mode = detect_mode()
    if req.use_agent:
        answer, retrieved, tool_calls = run_agent(req.message, STORE, top_k=req.top_k)
    else:
        hits = STORE.search(req.message, top_k=req.top_k)
        retrieved = [
            RetrievedDoc(doc_id=d, score=round(s, 4), text=t) for d, s, t in hits
        ]
        answer, mode = generate(req.message, [r.text for r in retrieved])
        tool_calls = []
    return ChatResponse(
        answer=answer,
        mode=mode,
        retrieved=retrieved,
        tool_calls=tool_calls,
        tokens_estimated=estimate_tokens(req.message + answer),
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Server-Sent Events: stream the answer token-by-token (chunked here).

    A real provider streams deltas; this demonstrates the SSE contract the
    frontend consumes so streaming UX works the same against a live model.
    """
    answer, retrieved, tool_calls = run_agent(req.message, STORE, top_k=req.top_k)

    async def event_gen():
        meta = {
            "retrieved": [r.model_dump() for r in retrieved],
            "tool_calls": [t.model_dump() for t in tool_calls],
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
        for word in answer.split(" "):
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"
            await asyncio.sleep(0.02)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/")
def index():
    if FRONTEND.exists():
        return FileResponse(str(FRONTEND))
    raise HTTPException(status_code=404, detail="frontend not found")
