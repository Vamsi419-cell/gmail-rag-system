"""
routers/chat.py — Chat endpoints.

POST /chat/stream   — Server-Sent Events streaming (ChatGPT-style)
POST /chat/ask      — Non-streaming JSON response (easier for testing)
"""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from schemas import ChatRequest, ChatResponse, SourceChunk
from services import embeddings as emb
from services import rag

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _latest_user_message(req: ChatRequest) -> str:
    """Extract the most recent user turn from the messages list."""
    for msg in reversed(req.messages):
        if msg.role == "user":
            return msg.content
    raise ValueError("No user message found in request.")


def _guard_index():
    if not emb.is_index_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "The FAISS index is empty. "
                "Trigger a sync first via POST /emails/sync/initial"
            ),
        )


def _guard_model():
    if not rag.model_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "The generator model is still loading. "
                "Please wait a moment and try again."
            ),
        )


# ────────────────────────────────────────────────────────────────────────────
# Streaming endpoint  (SSE)
# ────────────────────────────────────────────────────────────────────────────

@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """
    Returns a text/event-stream response.

    Event shape (each line is `data: <json>\\n\\n`):
      {"event": "token",   "data": "<text fragment>"}
      {"event": "sources", "data": [{"email_id": ..., "subject": ..., ...}]}
      {"event": "done",    "data": null}
      {"event": "error",   "data": "<message>"}

    Streamlit usage:
        import requests, json
        with requests.post(url, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line.startswith(b"data:"):
                    obj = json.loads(line[5:])
                    if obj["event"] == "token":
                        print(obj["data"], end="", flush=True)
    """
    _guard_index()
    _guard_model()

    try:
        question = _latest_user_message(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    async def event_generator() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in rag.stream_rag_response(question, top_k=req.top_k):
                yield chunk
        except Exception as exc:
            logger.exception("Streaming generation failed")
            error_evt = json.dumps({"event": "error", "data": str(exc)})
            yield f"data: {error_evt}\n\n".encode()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",       # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


# ────────────────────────────────────────────────────────────────────────────
# Non-streaming endpoint  (convenient for curl / unit tests)
# ────────────────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=ChatResponse)
async def chat_ask(req: ChatRequest):
    """
    Waits for the full answer and returns it as JSON.
    Sources are included in the response body.
    """
    _guard_index()
    _guard_model()

    try:
        question = _latest_user_message(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    tokens = []
    sources = []

    async for raw in rag.stream_rag_response(question, top_k=req.top_k):
        try:
            obj = json.loads(raw.decode().removeprefix("data: "))
            if obj["event"] == "token":
                tokens.append(obj["data"])
            elif obj["event"] == "sources":
                sources = [SourceChunk(**s) for s in obj["data"]]
        except Exception:
            pass

    return ChatResponse(answer="".join(tokens), sources=sources)
