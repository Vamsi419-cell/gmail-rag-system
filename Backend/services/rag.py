"""
services/rag.py — Retrieval-Augmented Generation pipeline.

Two backends are supported (toggled via settings.USE_OPENAI_COMPAT):

1. Local FLAN-T5 (default)
   Uses transformers.TextIteratorStreamer so tokens are yielded as they
   are produced by the model rather than after full generation.
   ⚠ Streaming requires num_beams=1; we compensate with temperature sampling.

2. OpenAI-compatible endpoint (LM Studio, vLLM, Groq, Ollama, etc.)
   A single env-var swap drops in any server that speaks the OpenAI
   chat-completions API with streaming (stream=True).

Both paths yield SSE-formatted byte strings:
    data: {"event": "token",   "data": "<text>"}\n\n
    data: {"event": "sources", "data": [...]}\n\n
    data: {"event": "done",    "data": null}\n\n
"""
from __future__ import annotations

import json
import logging
import threading
from typing import AsyncGenerator, List, Optional

import asyncio
import torch

from config import settings
from schemas import SourceChunk
from services.embeddings import search

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Local model singletons (FLAN-T5)
# ────────────────────────────────────────────────────────────────────────────

_tokenizer = None
_gen_model  = None


def _load_local_model():
    global _tokenizer, _gen_model

    if _gen_model is not None:
        return

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading generator '%s' on %s …", settings.GENERATOR_MODEL, device)

    _tokenizer = AutoTokenizer.from_pretrained(settings.GENERATOR_MODEL)
    _gen_model  = AutoModelForSeq2SeqLM.from_pretrained(
        settings.GENERATOR_MODEL,
        use_safetensors=False,          # avoids HF xet downloader that stalls
    )
    _gen_model  = _gen_model.to(device)
    _gen_model.eval()

    logger.info("Generator ready.")


_model_loading = False
_model_load_error: Optional[str] = None


def preload_model():
    """
    Kick off model loading in a background thread so the server
    can start immediately (auth, sync, health all work while the
    model downloads).  Only /chat needs the model.
    """
    global _model_loading

    if settings.USE_OPENAI_COMPAT:
        return                       # no local model needed

    _model_loading = True

    def _bg_load():
        global _model_loading, _model_load_error
        try:
            _load_local_model()
        except Exception as exc:
            _model_load_error = str(exc)
            logger.exception("Failed to load generator model")
        finally:
            _model_loading = False

    t = threading.Thread(target=_bg_load, daemon=True)
    t.start()


def model_ready() -> bool:
    """True once the local generator has finished loading."""
    if settings.USE_OPENAI_COMPAT:
        return True
    return _gen_model is not None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a helpful Gmail assistant. "
    "Answer questions using ONLY the email context provided. "
    "If the answer is not in the emails say exactly: "
    "\"I don't have enough information in your emails to answer that.\""
)

# Minimum cosine similarity to keep a retrieved chunk (filters noise)
_MIN_RELEVANCE_SCORE = 0.20

# Context char budget — keep small for flan-t5-small (512 token input limit)
_CONTEXT_CHAR_BUDGET = 800


def _build_prompt(question: str, context: str) -> str:
    """Build a T5-friendly prompt. The QUESTION must come FIRST because
    T5 models have a 512-token input limit — if context is long, anything
    at the end gets truncated and the model never sees the question."""
    return (
        f"Question: {question}\n\n"
        f"Summarize the relevant emails below to answer the question above.\n\n"
        f"Emails:\n{context}\n\n"
        f"Answer:"
    )


# ────────────────────────────────────────────────────────────────────────────
# SSE helpers
# ────────────────────────────────────────────────────────────────────────────

def _sse(event: str, data) -> bytes:
    payload = json.dumps({"event": event, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n".encode()


# ────────────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────────────

async def stream_rag_response(
    question: str,
    top_k: int = 5,
) -> AsyncGenerator[bytes, None]:
    """
    Full RAG pipeline yielding SSE byte chunks.

    Callers:  async for chunk in stream_rag_response(q): ...
    """
    # ── 1. Retrieve (run in thread to avoid blocking the event loop) ─────
    hits = await asyncio.to_thread(search, question, top_k)

    # Filter out low-relevance chunks
    hits = [(text, meta, score) for text, meta, score in hits
            if score >= _MIN_RELEVANCE_SCORE]

    if not hits:
        yield _sse("token", "I don't have enough information in your emails to answer that.")
        yield _sse("sources", [])
        yield _sse("done", None)
        return

    # Build context (respect character budget)
    context_parts: List[str] = []
    char_count = 0
    for chunk_text, meta, _ in hits:
        # Include subject for context
        part = f"Subject: {meta.get('subject', 'N/A')}\n{chunk_text}"
        if char_count + len(part) > _CONTEXT_CHAR_BUDGET:
            break
        context_parts.append(part)
        char_count += len(part)
    context = "\n---\n".join(context_parts)

    # Build source list for the UI
    sources = [
        SourceChunk(
            email_id=meta["id"],
            subject=meta["subject"],
            snippet=chunk[:120],
            score=round(score, 4),
        ).model_dump()
        for chunk, meta, score in hits
    ]

    # ── 2. Generate ──────────────────────────────────────────────────────────
    if settings.USE_OPENAI_COMPAT:
        async for chunk in _openai_stream(question, context):
            yield chunk
    else:
        async for chunk in _local_stream(question, context):
            yield chunk

    # ── 3. Sources + done sentinel ───────────────────────────────────────────
    yield _sse("sources", sources)
    yield _sse("done", None)


# ────────────────────────────────────────────────────────────────────────────
# Backend A — Local FLAN-T5 with TextIteratorStreamer
# ────────────────────────────────────────────────────────────────────────────

async def _local_stream(
    question: str,
    context: str,
) -> AsyncGenerator[bytes, None]:
    from transformers import TextIteratorStreamer  # type: ignore

    # Ensure model is loaded (offload to thread so we don't block the loop)
    await asyncio.to_thread(_load_local_model)
    device = _device()

    prompt = _build_prompt(question, context)

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)

    streamer = TextIteratorStreamer(
        _tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
        timeout=120,
    )

    # For small models: lower temperature + higher repetition penalty
    # prevents copying raw email text and produces focused summaries
    temp = min(settings.TEMPERATURE, 0.4)
    rep_pen = max(settings.REPETITION_PENALTY, 2.5)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=settings.MAX_NEW_TOKENS,
        num_beams=1,                        # beam search blocks the streamer
        do_sample=True,
        temperature=temp,
        repetition_penalty=rep_pen,
    )

    # Run generation in a background thread so we can yield from async context
    thread = threading.Thread(target=_gen_model.generate, kwargs=generation_kwargs)
    thread.start()

    for token_text in streamer:
        if token_text:
            yield _sse("token", token_text)
        # yield control back to the event loop so other requests can proceed
        await asyncio.sleep(0)

    thread.join()


# ────────────────────────────────────────────────────────────────────────────
# Backend B — OpenAI-compatible streaming (drop-in upgrade path)
# ────────────────────────────────────────────────────────────────────────────

async def _openai_stream(
    question: str,
    context: str,
) -> AsyncGenerator[bytes, None]:
    try:
        import httpx  # type: ignore
    except ImportError:
        raise RuntimeError("httpx is required for OpenAI-compat backend: pip install httpx")

    import asyncio

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Email context:\n{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    payload = {
        "model": settings.OPENAI_MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": settings.MAX_NEW_TOKENS,
        "temperature": settings.TEMPERATURE,
    }

    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{settings.OPENAI_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield _sse("token", delta)
                        await asyncio.sleep(0)
                except (json.JSONDecodeError, KeyError):
                    continue


# ────────────────────────────────────────────────────────────────────────────
# Non-streaming convenience (used by tests / scripts)
# ────────────────────────────────────────────────────────────────────────────

async def ask(question: str, top_k: int = 5) -> str:
    tokens: List[str] = []
    async for raw in stream_rag_response(question, top_k=top_k):
        try:
            data = json.loads(raw.decode().removeprefix("data: "))
            if data["event"] == "token":
                tokens.append(data["data"])
        except Exception:
            pass
    return "".join(tokens)
