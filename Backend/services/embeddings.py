"""
services/embeddings.py — Sentence-Transformer encoder + FAISS index.

Responsibilities
─────────────────
• Load / persist the FAISS index and parallel metadata list to disk
• Encode text chunks via SentenceTransformer (GPU if available)
• Add new chunks incrementally (no full rebuild needed)
• Similarity search → returns (chunk_text, metadata, score) triples
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import faiss                                          # type: ignore
import numpy as np
import torch
from sentence_transformers import SentenceTransformer  # type: ignore

from config import settings

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Module-level singletons (loaded once at application start-up)
# ────────────────────────────────────────────────────────────────────────────

_encoder: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_chunks: List[str] = []          # parallel list of raw text chunks
_metadata: List[dict] = []       # parallel list of {"id": ..., "subject": ...}
_dimension: Optional[int] = None


# ────────────────────────────────────────────────────────────────────────────
# Init
# ────────────────────────────────────────────────────────────────────────────

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        logger.info("Loading encoder '%s' on %s …", settings.EMBEDDING_MODEL, _device())
        _encoder = SentenceTransformer(settings.EMBEDDING_MODEL, device=_device())
        logger.info("Encoder ready.")
    return _encoder


def load_index() -> None:
    """
    Load a persisted FAISS index + metadata from disk (if they exist),
    otherwise initialise empty structures.  Must be called at startup.
    """
    global _index, _chunks, _metadata, _dimension

    idx_path = settings.FAISS_INDEX_PATH
    meta_path = settings.METADATA_PATH

    if idx_path.exists() and meta_path.exists():
        logger.info("Loading existing FAISS index from %s …", idx_path)
        _index = faiss.read_index(str(idx_path))
        with open(str(meta_path), "rb") as f:
            saved = pickle.load(f)
        _chunks   = saved["chunks"]
        _metadata = saved["metadata"]
        _dimension = _index.d
        logger.info("Index loaded — %d vectors.", _index.ntotal)
    else:
        logger.info("No persisted index found; starting fresh.")
        _index     = None
        _chunks    = []
        _metadata  = []
        _dimension = None


def save_index() -> None:
    """Flush the in-memory FAISS index and metadata to disk."""
    if _index is None:
        return

    idx_path  = settings.FAISS_INDEX_PATH
    meta_path = settings.METADATA_PATH
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(_index, str(idx_path))
    with open(str(meta_path), "wb") as f:
        pickle.dump({"chunks": _chunks, "metadata": _metadata}, f)

    logger.info("Index saved — %d vectors.", _index.ntotal)


# ────────────────────────────────────────────────────────────────────────────
# Chunking
# ────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> List[str]:
    words = text.split()
    step  = settings.CHUNK_SIZE - settings.CHUNK_OVERLAP
    return [
        " ".join(words[i : i + settings.CHUNK_SIZE])
        for i in range(0, len(words), step)
        if words[i : i + settings.CHUNK_SIZE]   # skip empty tail
    ]


# ────────────────────────────────────────────────────────────────────────────
# Index building / updating
# ────────────────────────────────────────────────────────────────────────────

def _encode(texts: List[str]) -> np.ndarray:
    enc = load_encoder()
    vecs = enc.encode(
        texts,
        batch_size=64,
        show_progress_bar=len(texts) > 200,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype("float32")


def build_index_from_emails(
    emails: List[Tuple[str, str, str]]   # (id, subject, body)
) -> int:
    """
    Rebuild the FAISS index completely from the supplied emails.
    Returns total number of vectors in the index.
    """
    global _index, _chunks, _metadata, _dimension

    all_chunks:    List[str]  = []
    all_metadata:  List[dict] = []

    for email_id, subject, body in emails:
        full_text = f"Subject: {subject}\nBody: {body}"
        for chunk in chunk_text(full_text):
            all_chunks.append(chunk)
            all_metadata.append({"id": email_id, "subject": subject})

    if not all_chunks:
        logger.warning("No chunks to index.")
        return 0

    logger.info("Encoding %d chunks …", len(all_chunks))
    vecs = _encode(all_chunks)

    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner-product on normalised vectors == cosine
    index.add(vecs)

    _index     = index
    _chunks    = all_chunks
    _metadata  = all_metadata
    _dimension = dim

    save_index()
    logger.info("Index built — %d vectors.", _index.ntotal)
    return _index.ntotal


def add_emails_to_index(
    emails: List[Tuple[str, str, str]]
) -> int:
    """
    Incrementally add new emails to an existing index.
    If the index doesn't exist yet, builds it from scratch.
    Returns number of NEW vectors added.
    """
    global _index, _chunks, _metadata, _dimension

    if _index is None:
        return build_index_from_emails(emails)

    new_chunks:   List[str]  = []
    new_metadata: List[dict] = []

    for email_id, subject, body in emails:
        full_text = f"Subject: {subject}\nBody: {body}"
        for chunk in chunk_text(full_text):
            new_chunks.append(chunk)
            new_metadata.append({"id": email_id, "subject": subject})

    if not new_chunks:
        return 0

    vecs = _encode(new_chunks)
    _index.add(vecs)
    _chunks.extend(new_chunks)
    _metadata.extend(new_metadata)

    save_index()
    logger.info("Added %d new vectors. Total: %d.", len(new_chunks), _index.ntotal)
    return len(new_chunks)


# ────────────────────────────────────────────────────────────────────────────
# Retrieval
# ────────────────────────────────────────────────────────────────────────────

def search(
    query: str,
    top_k: int = 5,
) -> List[Tuple[str, dict, float]]:
    """
    Search the index for the most relevant chunks.
    Returns list of (chunk_text, metadata_dict, cosine_score).
    """
    if _index is None or _index.ntotal == 0:
        return []

    enc = load_encoder()
    q_vec = enc.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    k = min(top_k, _index.ntotal)
    scores, indices = _index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append((_chunks[idx], _metadata[idx], float(score)))

    return results


# ────────────────────────────────────────────────────────────────────────────
# Status helpers
# ────────────────────────────────────────────────────────────────────────────

def index_size() -> int:
    return _index.ntotal if _index else 0


def is_index_ready() -> bool:
    return _index is not None and _index.ntotal > 0
