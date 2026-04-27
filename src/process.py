import sqlite3
import re
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from datetime import datetime
from pathlib import Path
from config import DB_PATH, CHUNKS_PATH, METADATA_PATH, INDEX_PATH

CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_emails():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT id, subject, body, sender, labels, timestamp FROM emails")
    rows = cursor.fetchall()
    conn.close()
    return rows

def clean_email(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'---------- Forwarded message ---------.*', '', text, flags=re.DOTALL)
    text = re.sub(r'On .* wrote:', '', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def prepare_chunks():
    emails = load_emails()
    if not emails:
        print("No emails found in DB")
        return [], []

    chunks, metadata = [], []
    for email_id, subject, raw_body, sender, labels, timestamp in emails:
        full_text = f"From: {sender}\nDate: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')}\nSubject: {subject}\nBody: {clean_email(raw_body)}"
        for idx, chunk in enumerate(chunk_text(full_text)):
            chunks.append(chunk)
            metadata.append({"id": email_id, "subject": subject, "sender": sender, "labels": labels, "timestamp": timestamp, "chunk_index": idx})

    print(f"Total chunks: {len(chunks)}")
    return chunks, metadata

def create_embeddings(chunks):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return np.array(model.encode(chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True)).astype("float32")

def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def save_all(index, chunks, metadata):
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print("Saved index, chunks, and metadata")

if __name__ == "__main__":
    chunks, metadata = prepare_chunks()
    if not chunks:
        exit()
    save_all(build_faiss_index(create_embeddings(chunks)), chunks, metadata)
    print("Done!")