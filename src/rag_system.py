import faiss
import json
import os
import torch
import re
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from dotenv import load_dotenv

from config import INDEX_PATH, CHUNKS_PATH, METADATA_PATH, BASE_DIR

load_dotenv(BASE_DIR / ".env")

# =========================
# GLOBALS
# =========================
_model = None
_cross_encoder = None
_index = None
_chunks = None
_metadata = None
_client = None


# =========================
# LOAD SYSTEM
# =========================
def _ensure_loaded():
    global _model, _cross_encoder, _index, _chunks, _metadata, _client

    if _model:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    _index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        _chunks = json.load(f)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    _client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print("✅ RAG Ready")


# =========================
# FILTER PARSER
# =========================
def extract_filters(query):
    q = query.lower()
    filters = {}

    sender = re.search(r'from\s+([\w\.-]+)', q)
    if sender:
        filters["sender"] = sender.group(1)

    time_match = re.search(r'(\d+)\s+(day|week|month)s?', q)
    if time_match:
        num = int(time_match.group(1))
        unit = time_match.group(2)

        delta = timedelta(
            days=num if unit == "day" else
            7*num if unit == "week" else
            30*num
        )

        filters["after"] = int((datetime.now() - delta).timestamp())

    elif "last week" in q:
        filters["after"] = int((datetime.now() - timedelta(days=7)).timestamp())

    return filters


# =========================
# KEYWORD SCORE
# =========================


def keyword_score(query, text):
    query_words = re.findall(r'\w+', query.lower())
    text = text.lower()

    score = 0
    for word in query_words:
        if word in text:
            score += 1

    return score


# =========================
# CROSS ENCODER
# =========================
def rerank(query, candidates):
    pairs = [(query, c[1]) for c in candidates]
    scores = _cross_encoder.predict(pairs)

    reranked = [
        (float(scores[i]), candidates[i][1], candidates[i][2])
        for i in range(len(scores))
    ]

    return sorted(reranked, key=lambda x: x[0], reverse=True)


# =========================
# BUILD CLEAN CONTEXT
# =========================
def build_context(top_candidates):
    context_blocks = []

    for i, (score, chunk, meta) in enumerate(top_candidates):

        date = datetime.fromtimestamp(meta["timestamp"]).strftime("%Y-%m-%d")

        block = f"""
Email {i+1}
From: {meta['sender']}
Date: {date}
Subject: {meta['subject']}

Content:
{chunk}
"""
        context_blocks.append(block.strip())

    return "\n\n---\n\n".join(context_blocks)


# =========================
# MAIN FUNCTION
# =========================
def ask_my_emails(question, top_k=4, min_score=0.25):
    _ensure_loaded()

    print(f"\n🤔 {question}")

    filters = extract_filters(question)

    query_vec = _model.encode([question], normalize_embeddings=True).astype("float32")

    distances, indices = _index.search(query_vec, top_k * 5)

    candidates = []

    for i, idx in enumerate(indices[0]):
        score = float(distances[0][i])

        if score < min_score:
            continue

        chunk = _chunks[idx]
        meta = _metadata[idx]

        # FILTERING
        if "sender" in filters and filters["sender"] not in meta["sender"].lower():
            continue

        if "after" in filters and meta["timestamp"] < filters["after"]:
            continue

        combined = score + 0.1 * keyword_score(question, chunk)

        candidates.append((combined, chunk, meta))

    if not candidates:
        return "No relevant emails found."

    # RERANK
    reranked = rerank(question, candidates)

    top_candidates = reranked[:top_k]

    # CONTEXT
    context = build_context(top_candidates)

    # =========================
    # 🔥 ADVANCED PROMPT
    # =========================
    prompt = f"""
You are an intelligent email assistant.

You are given email records with metadata:
- Sender
- Date
- Subject
- Content

INSTRUCTIONS:
- Answer ONLY using the given emails
- If multiple emails exist, choose the most relevant
- Prefer recent emails if time is relevant
- Be precise and concise
- If answer not found, say "Not found in emails"

EMAIL DATA:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = _client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_completion_tokens=512,
    )

    answer = response.choices[0].message.content

    print("\n" + "="*50)
    print(answer)
    print("="*50)

    return answer


# =========================
# TEST LOOP
# =========================
if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        if q.lower() == "exit":
            break
        ask_my_emails(q)
        
        