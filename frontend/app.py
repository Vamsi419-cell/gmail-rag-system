"""
frontend/app.py — Streamlit UI for the Gmail RAG Assistant.

Connects to the FastAPI backend at BACKEND_URL (default localhost:8000).
Streams tokens in real-time using the /chat/stream SSE endpoint.
"""
from __future__ import annotations

import json
import time
from typing import Generator

import requests
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────

BACKEND_URL = "http://localhost:8000"


def _api(path: str) -> str:
    return f"{BACKEND_URL}{path}"


# ────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Gmail Assistant",
    page_icon="📬",
    layout="centered",
)

st.markdown("""
<style>
/* ── Layout ──────────────────────────────────────────────────────────── */
.block-container { padding-top: 1.2rem; max-width: 820px; }

/* ── Header card ─────────────────────────────────────────────────────── */
.header-card {
    padding: 16px 20px;
    border-radius: 12px;
    background: linear-gradient(135deg, #f8f9fb 0%, #eef2ff 100%);
    border: 1px solid #e0e4ef;
    margin-bottom: 8px;
}
.header-title { font-size: 22px; font-weight: 700; color: #1a1a2e; }
.header-sub   { color: #6b7280; font-size: 13px; margin-top: 2px; }

/* ── Status badges ───────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
}
.badge-green  { background: #d1fae5; color: #065f46; }
.badge-yellow { background: #fef3c7; color: #92400e; }
.badge-red    { background: #fee2e2; color: #991b1b; }

/* ── Source chips ────────────────────────────────────────────────────── */
.source-chip {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 11px;
    color: #475569;
    margin: 2px 3px;
}

/* ── Footer ──────────────────────────────────────────────────────────── */
.footer { text-align: center; color: #9ca3af; font-size: 11px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Session state bootstrap
# ────────────────────────────────────────────────────────────────────────────

if "messages"      not in st.session_state: st.session_state.messages      = []
if "sources_map"   not in st.session_state: st.session_state.sources_map   = {}   # msg_idx → sources
if "syncing"       not in st.session_state: st.session_state.syncing       = False


# ────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ────────────────────────────────────────────────────────────────────────────

def _get_health() -> dict:
    try:
        r = requests.get(_api("/health"), timeout=3)
        return r.json()
    except Exception:
        return {}


def _get_auth_status() -> dict:
    try:
        r = requests.get(_api("/auth/status"), timeout=3)
        return r.json()
    except Exception:
        return {"authenticated": False}


def _get_sync_status() -> dict:
    try:
        r = requests.get(_api("/emails/sync/status"), timeout=3)
        return r.json()
    except Exception:
        return {}


def _trigger_login():
    try:
        r = requests.post(_api("/auth/login"), timeout=120)
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _trigger_initial_sync():
    try:
        r = requests.post(_api("/emails/sync/initial"), timeout=60)
        return r.json()
    except Exception as e:
        return {"message": str(e)}


def _trigger_delta_sync():
    try:
        r = requests.post(_api("/emails/sync/delta"), timeout=60)
        return r.json()
    except Exception as e:
        return {"message": str(e)}


def _stream_answer(question: str) -> Generator[tuple, None, None]:
    """
    Yields (token_text, sources_or_None) tuples.
    sources is a list of dicts, yielded exactly once at stream end.
    """
    payload = {
        "messages": [{"role": "user", "content": question}],
        "top_k": 5,
        "stream": True,
    }
    try:
        with requests.post(
            _api("/chat/stream"),
            json=payload,
            stream=True,
            timeout=(10, 300),   # (connect_timeout, read_timeout)
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                try:
                    obj = json.loads(line[5:].strip())
                    event = obj.get("event")
                    if event == "token":
                        yield obj["data"], None
                    elif event == "sources":
                        yield "", obj["data"]
                    elif event == "error":
                        yield f"\n\n⚠️ Error: {obj['data']}", None
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        yield f"\n\n⚠️ Could not reach backend: {exc}", None


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Gmail RAG Controls")

    # ── Auth status ──────────────────────────────────────────────────────────
    auth = _get_auth_status()
    if auth.get("authenticated"):
        st.markdown(
            f'<span class="badge badge-green">✓ Connected</span> '
            f'<span style="font-size:12px;color:#374151">{auth.get("email","")}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="badge badge-red">✗ Not connected</span>', unsafe_allow_html=True)
        if st.button("🔐 Connect Gmail", use_container_width=True):
            with st.spinner("Opening browser for OAuth…"):
                result = _trigger_login()
            if result.get("status") == "ok":
                st.success(result["message"])
                st.rerun()
            else:
                st.error(result.get("message", "Login failed"))

    st.divider()

    # ── Index status ─────────────────────────────────────────────────────────
    health = _get_health()
    emails_count = health.get("emails_in_db", "—")
    chunks_count = health.get("chunks_in_index", "—")
    index_ready  = health.get("index_ready", False)

    col1, col2 = st.columns(2)
    col1.metric("Emails", emails_count)
    col2.metric("Vectors", chunks_count)

    if index_ready:
        st.markdown('<span class="badge badge-green">Index ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-yellow">Index empty</span>', unsafe_allow_html=True)

    st.divider()

    # ── Sync controls ────────────────────────────────────────────────────────
    st.markdown("**Sync Emails**")

    sync_s = _get_sync_status()
    sync_status = sync_s.get("status", "idle")

    if sync_status == "running":
        st.info(f"⏳ {sync_s.get('message', 'Running…')}")
        if st.button("🔄 Refresh status"):
            st.rerun()
    else:
        if sync_status == "error":
            st.error(sync_s.get("message", ""))

        c1, c2 = st.columns(2)
        if c1.button("🚀 Full sync", use_container_width=True):
            _trigger_initial_sync()
            st.toast("Full sync started!")
            st.rerun()

        if c2.button("⚡ Delta sync", use_container_width=True):
            _trigger_delta_sync()
            st.toast("Delta sync started!")
            st.rerun()

    st.divider()

    # ── Clear chat ───────────────────────────────────────────────────────────
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages    = []
        st.session_state.sources_map = {}
        st.rerun()

    st.markdown(
        '<div class="footer">Gmail RAG • FastAPI + FAISS + FLAN-T5</div>',
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-card">
  <div class="header-title">📬 Gmail Assistant</div>
  <div class="header-sub">Ask anything about your emails — answers are generated from your inbox</div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Chat history
# ────────────────────────────────────────────────────────────────────────────

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Show sources beneath assistant messages
        if message["role"] == "assistant":
            srcs = st.session_state.sources_map.get(idx, [])
            if srcs:
                chips = " ".join(
                    f'<span class="source-chip">📧 {s["subject"][:50]}</span>'
                    for s in srcs
                )
                st.markdown(
                    f'<div style="margin-top:6px">{chips}</div>',
                    unsafe_allow_html=True,
                )


# ────────────────────────────────────────────────────────────────────────────
# Chat input
# ────────────────────────────────────────────────────────────────────────────

health_ok = _get_health().get("index_ready", False)
placeholder = (
    "Ask something about your emails…"
    if health_ok
    else "⚠ Sync emails first, then ask questions here"
)

query = st.chat_input(placeholder)

if query:
    # ── Show user bubble ─────────────────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # ── Stream assistant response ────────────────────────────────────────────
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()

        full_answer = ""
        final_sources = []

        for token, sources in _stream_answer(query):
            if sources is not None:
                final_sources = sources
            if token:
                full_answer += token
                answer_placeholder.markdown(full_answer + "▌")  # cursor effect

        answer_placeholder.markdown(full_answer)   # final, no cursor

        # Render source chips
        if final_sources:
            chips = " ".join(
                f'<span class="source-chip">📧 {s["subject"][:50]}</span>'
                for s in final_sources
            )
            sources_placeholder.markdown(
                f'<div style="margin-top:6px">{chips}</div>',
                unsafe_allow_html=True,
            )

    # ── Persist to session state ─────────────────────────────────────────────
    msg_idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    st.session_state.sources_map[msg_idx] = final_sources
