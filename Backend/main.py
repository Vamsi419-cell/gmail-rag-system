"""
main.py — FastAPI application entry point.

Start with:
    cd Backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Interactive API docs:  http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import auth, chat, emails
from services import database, embeddings
from services import rag

# ────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Lifespan  (replaces deprecated @app.on_event)
# ────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:  ensure DB tables exist, load the sentence-transformer encoder
              and any previously persisted FAISS index from disk.
    Shutdown: flush the FAISS index to disk so it survives restarts.
    """
    logger.info("═══ Gmail RAG API — starting up ═══")

    # 1. Database
    database.ensure_tables()
    logger.info("SQLite database ready at %s", settings.DB_PATH)

    # 2. Sentence-Transformer encoder  (downloads model on first run)
    embeddings.load_encoder()

    # 3. FAISS index  (no-op if no persisted index exists yet)
    embeddings.load_index()
    logger.info(
        "FAISS index: %d vectors (ready=%s)",
        embeddings.index_size(),
        embeddings.is_index_ready(),
    )

    # 4. Generator model  (pre-load so first chat query is fast)
    rag.preload_model()

    yield   # ← app is running

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("Saving FAISS index to disk …")
    embeddings.save_index()
    logger.info("═══ Gmail RAG API — shutdown complete ═══")


# ────────────────────────────────────────────────────────────────────────────
# App factory
# ────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Gmail RAG API",
    description=(
        "Retrieval-Augmented Generation over your Gmail inbox.  "
        "Authenticate → Sync emails → Chat."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow Streamlit to call us) ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(emails.router)
app.include_router(chat.router)


# ────────────────────────────────────────────────────────────────────────────
# Health check
# ────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health():
    return {
        "status":           "ok",
        "emails_in_db":     database.count_emails(),
        "chunks_in_index":  embeddings.index_size(),
        "index_ready":      embeddings.is_index_ready(),
        "model_ready":      rag.model_ready(),
    }


# ────────────────────────────────────────────────────────────────────────────
# Dev runner
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
