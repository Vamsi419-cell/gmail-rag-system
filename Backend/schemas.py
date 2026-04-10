"""
schemas.py — All Pydantic request / response models for the API.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ────────────────────────────────────────────────
# Generic
# ────────────────────────────────────────────────

class StatusResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# ────────────────────────────────────────────────
# Auth
# ────────────────────────────────────────────────

class AuthStatusResponse(BaseModel):
    authenticated: bool
    email: Optional[str] = None


# ────────────────────────────────────────────────
# Email / Sync
# ────────────────────────────────────────────────

class SyncStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class SyncStatusResponse(BaseModel):
    status: SyncStatus
    emails_in_db: int
    chunks_in_index: int
    last_sync_timestamp: Optional[int] = None
    message: str = ""


class TriggerSyncResponse(BaseModel):
    message: str
    task_id: str


# ────────────────────────────────────────────────
# Chat
# ────────────────────────────────────────────────

class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = True


class SourceChunk(BaseModel):
    email_id: str
    subject: str
    snippet: str           # first 120 chars of the chunk
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


# ────────────────────────────────────────────────
# SSE streaming event shapes  (serialised to JSON)
# ────────────────────────────────────────────────

class StreamEventType(str, Enum):
    TOKEN = "token"
    SOURCES = "sources"
    DONE = "done"
    ERROR = "error"


class StreamEvent(BaseModel):
    event: StreamEventType
    data: Any
