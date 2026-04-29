import sqlite3
import re
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from datetime import datetime

from config import DB_PATH, get_user_paths

# =========================
# CACHED MODEL (loaded once)
# =========================
_embed_model = None


def _get_embed_model():
    """Get or create the shared SentenceTransformer instance."""
    global _embed_model
    if _embed_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("✅ Embedding model loaded (process.py)")
    return _embed_model


# =========================
# LOAD EMAILS (PER USER)
# =========================
def load_emails(user_id):
    """Load all emails for a specific user from the database."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, subject, body, sender, labels, timestamp FROM emails WHERE user_id = ?",
            (user_id,)
        )
        rows = cursor.fetchall()
    return rows


# =========================
# TEXT PROCESSING
# =========================
def clean_email(text):
    """Remove URLs, forwarded headers, and excess whitespace."""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'---------- Forwarded message ---------.*', '', text, flags=re.DOTALL)
    # More specific reply header pattern: "On <date>, <name> <email> wrote:"
    text = re.sub(r'On\s+\w{3},\s+\w{3}\s+\d{1,2},\s+\d{4}.*?wrote:', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_text(text, chunk_size=300, overlap=50):
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# =========================
# PREPARE CHUNKS (PER USER)
# =========================
def prepare_chunks(user_id):
    """Load emails for a user and split them into chunks with metadata."""
    emails = load_emails(user_id)
    if not emails:
        print(f"No emails found in DB for user {user_id}")
        return [], []

    chunks, metadata = [], []
    for email_id, subject, raw_body, sender, labels, timestamp in emails:
        full_text = (
            f"From: {sender}\n"
            f"Date: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')}\n"
            f"Subject: {subject}\n"
            f"Body: {clean_email(raw_body)}"
        )
        for idx, chunk in enumerate(chunk_text(full_text)):
            chunks.append(chunk)
            metadata.append({
                "id": email_id,
                "subject": subject,
                "sender": sender,
                "labels": labels,
                "timestamp": timestamp,
                "chunk_index": idx,
            })

    print(f"Total chunks for user {user_id}: {len(chunks)}")
    return chunks, metadata


# =========================
# EMBEDDINGS & FAISS
# =========================
def create_embeddings(chunks):
    """Generate normalized sentence embeddings for a list of text chunks."""
    model = _get_embed_model()
    return np.array(
        model.encode(chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    ).astype("float32")


def build_faiss_index(embeddings):
    """Build a FAISS inner-product index from embeddings."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# =========================
# SAVE (PER USER)
# =========================
def save_all(index, chunks, metadata, user_id):
    """Save FAISS index, chunks, and metadata to user-specific paths."""
    paths = get_user_paths(user_id)

    faiss.write_index(index, str(paths["index_path"]))

    with open(paths["chunks_path"], "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    with open(paths["metadata_path"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    print(f"Saved index, chunks, and metadata for user {user_id}")


# =========================
# CONVENIENCE: BUILD ALL
# =========================
def build_user_index(user_id):
    """
    Full pipeline: load emails → chunk → embed → build FAISS → save.
    Returns a status dict for the Flask frontend.
    """
    chunks, metadata = prepare_chunks(user_id)

    if not chunks:
        return {
            "status": "error",
            "message": f"No emails found for user {user_id}. Fetch emails first.",
        }

    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)
    save_all(index, chunks, metadata, user_id)

    msg = f"Index built — {len(chunks)} chunks, {index.ntotal} vectors"
    print(f"✅ {msg}")
    return {"status": "ok", "message": msg, "chunks": len(chunks)}


# =========================
# CLI ENTRY POINT
# =========================
if __name__ == "__main__":
    import sys
    uid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    result = build_user_index(uid)
    print(result)