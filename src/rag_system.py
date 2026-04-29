import faiss
import json
import os
import torch
import re
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from dotenv import load_dotenv

from config import BASE_DIR, get_user_paths

load_dotenv(BASE_DIR / ".env")

# =========================
# SHARED MODELS (loaded once)
# =========================
_model = None
_cross_encoder = None
_client = None


def _ensure_models_loaded():
    """Load the embedding model, cross-encoder, and Groq client once."""
    global _model, _cross_encoder, _client

    if _model is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    _client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print("✅ RAG models loaded")


# =========================
# PER-USER DATA CACHE
# =========================
_user_cache = {}  # { user_id: {"index": ..., "chunks": ..., "metadata": ...} }


def _load_user_data(user_id):
    """Load a user's FAISS index, chunks, and metadata. Uses in-memory cache."""
    if user_id in _user_cache:
        return _user_cache[user_id]

    paths = get_user_paths(user_id)

    index_path = paths["index_path"]
    chunks_path = paths["chunks_path"]
    metadata_path = paths["metadata_path"]

    if not index_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found for user {user_id}. "
            "Please fetch emails and build the index first."
        )

    index = faiss.read_index(str(index_path))

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    data = {"index": index, "chunks": chunks, "metadata": metadata}
    _user_cache[user_id] = data

    # Simple LRU: keep at most 10 users in cache
    if len(_user_cache) > 10:
        oldest = next(iter(_user_cache))
        del _user_cache[oldest]

    print(f"✅ Loaded data for user {user_id} ({index.ntotal} vectors)")
    return data


def invalidate_user_cache(user_id):
    """Remove a user's data from cache (call after rebuilding index)."""
    _user_cache.pop(user_id, None)


# =========================
# FILTER PARSER
# =========================
def extract_filters(query):
    """Extract sender and time-range filters from natural language query."""
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
            7 * num if unit == "week" else
            30 * num
        )
        filters["after"] = int((datetime.now() - delta).timestamp())

    elif "last week" in q:
        filters["after"] = int((datetime.now() - timedelta(days=7)).timestamp())
    elif "last month" in q:
        filters["after"] = int((datetime.now() - timedelta(days=30)).timestamp())
    elif "today" in q:
        filters["after"] = int(datetime.now().replace(hour=0, minute=0, second=0).timestamp())
    elif "yesterday" in q:
        filters["after"] = int((datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0).timestamp())

    return filters


# =========================
# KEYWORD SCORE
# =========================
_STOP_WORDS = frozenset({
    "i", "me", "my", "the", "a", "an", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "to",
    "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
    "about", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "it", "its", "and", "but", "or", "not", "no", "so",
    "if", "then", "than", "too", "very", "just", "how", "when", "where",
    "why", "all", "any", "each", "every", "some", "such",
})


def keyword_score(query, text):
    """Word-boundary keyword overlap score, ignoring stop words."""
    query_words = set(re.findall(r'\w+', query.lower())) - _STOP_WORDS
    text_words = set(re.findall(r'\w+', text.lower()))

    if not query_words:
        return 0

    return len(query_words & text_words)


# =========================
# CROSS ENCODER RERANK
# =========================
def rerank(query, candidates):
    """Rerank candidates using a cross-encoder model."""
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
    """Format top email chunks into a structured context string for the LLM."""
    context_blocks = []

    for i, (score, chunk, meta) in enumerate(top_candidates):
        date = datetime.fromtimestamp(meta["timestamp"]).strftime("%Y-%m-%d")

        block = f"""Email {i+1}
From: {meta['sender']}
Date: {date}
Subject: {meta['subject']}

Content:
{chunk}"""
        context_blocks.append(block.strip())

    return "\n\n---\n\n".join(context_blocks)


# =========================
# QUERY REWRITER (for follow-ups)
# =========================
def _rewrite_followup(question, chat_history):
    """
    If the question looks like a follow-up, use the LLM to rewrite it
    into a standalone question using recent chat context.
    """
    # Only rewrite if there's history and the question seems like a follow-up
    followup_signals = [
        len(question.split()) < 8,
        any(w in question.lower() for w in ["it", "that", "this", "them", "those", "their", "its", "he", "she", "they", "more", "also", "same", "above"]),
    ]

    if not any(followup_signals):
        return question

    # Build recent history string (last 3 exchanges max)
    recent = chat_history[-6:]  # 3 pairs of user/assistant
    history_str = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:200]}"
        for m in recent
    )

    rewrite_prompt = f"""Given this conversation history and a follow-up question, rewrite the follow-up into a standalone question that can be understood without the conversation history. Only output the rewritten question, nothing else.

Conversation:
{history_str}

Follow-up: {question}

Standalone question:"""

    try:
        response = _client.chat.completions.create(
            messages=[{"role": "user", "content": rewrite_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_completion_tokens=100,
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten:
            print(f"🔄 Rewritten: '{question}' → '{rewritten}'")
            return rewritten
    except Exception as e:
        print(f"⚠️ Rewrite failed: {e}")

    return question


# =========================
# MAIN FUNCTION
# =========================
def ask_my_emails(user_id, question, chat_history=None, top_k=4, min_score=0.25):
    """
    Run the full RAG pipeline for a specific user:
    rewrite follow-ups → embed query → FAISS search → filter → rerank → LLM answer.
    """
    _ensure_models_loaded()

    # Rewrite follow-up questions using chat history
    search_query = question
    if chat_history:
        search_query = _rewrite_followup(question, chat_history)

    user_data = _load_user_data(user_id)
    index = user_data["index"]
    chunks = user_data["chunks"]
    metadata = user_data["metadata"]

    print(f"\n🤔 [{user_id}] {question}")

    filters = extract_filters(search_query)

    query_vec = _model.encode([search_query], normalize_embeddings=True).astype("float32")

    # Retrieve more candidates, filter down
    search_k = min(top_k * 5, index.ntotal)
    distances, indices = index.search(query_vec, search_k)

    candidates = []

    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        score = float(distances[0][i])

        if score < min_score:
            continue

        chunk = chunks[idx]
        meta = metadata[idx]

        # FILTERING
        if "sender" in filters and filters["sender"] not in meta["sender"].lower():
            continue

        if "after" in filters and meta["timestamp"] < filters["after"]:
            continue

        combined = score + 0.1 * keyword_score(search_query, chunk)

        candidates.append((combined, chunk, meta))

    if not candidates:
        return "No relevant emails found."

    # RERANK
    reranked = rerank(search_query, candidates)

    top_candidates = reranked[:top_k]

    # CONTEXT
    context = build_context(top_candidates)

    # =========================
    # LLM PROMPT (with chat history)
    # =========================
    history_block = ""
    if chat_history:
        recent = chat_history[-4:]  # last 2 exchanges
        history_block = "CONVERSATION HISTORY:\n" + "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:300]}"
            for m in recent
        ) + "\n\n"

    prompt = f"""You are an intelligent email assistant.

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
- Use conversation history to understand follow-up questions

{history_block}EMAIL DATA:
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

    print("\n" + "=" * 50)
    print(answer)
    print("=" * 50)

    return answer


# =========================
# CLI TEST LOOP
# =========================
if __name__ == "__main__":
    import sys
    uid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    while True:
        q = input("\nAsk: ")
        if q.lower() == "exit":
            break
        ask_my_emails(uid, q)