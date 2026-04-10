"""
services/sync.py — Email ingestion / sync state machine.

Designed to run as FastAPI BackgroundTask so the HTTP response returns
immediately while the heavy lifting happens asynchronously.

State is kept in memory (sufficient for a single-process deployment).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from config import settings
from schemas import SyncStatus
from services import database, embeddings, gmail

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Module-level state
# ────────────────────────────────────────────────────────────────────────────

class _SyncState:
    def __init__(self):
        self.status:  SyncStatus  = SyncStatus.IDLE
        self.message: str         = "Waiting for first sync."
        self.task_id: str         = ""
        self.lock: asyncio.Lock   = asyncio.Lock()


_state = _SyncState()


def get_status() -> dict:
    return {
        "status":               _state.status,
        "message":              _state.message,
        "emails_in_db":         database.count_emails(),
        "chunks_in_index":      embeddings.index_size(),
        "last_sync_timestamp":  _read_last_sync_ts(),
    }


# ────────────────────────────────────────────────────────────────────────────
# Last-sync clock helpers
# ────────────────────────────────────────────────────────────────────────────

def _read_last_sync_ts() -> Optional[int]:
    sync_file = settings.SYNC_FILE
    if not sync_file.exists():
        return None
    try:
        with open(str(sync_file)) as f:
            return json.load(f).get("last_sync_time")
    except Exception:
        return None


def _write_last_sync_ts(ts: int) -> None:
    settings.SYNC_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(str(settings.SYNC_FILE), "w") as f:
        json.dump({"last_sync_time": ts}, f)


# ────────────────────────────────────────────────────────────────────────────
# Background tasks
# ────────────────────────────────────────────────────────────────────────────

async def run_initial_sync(task_id: str) -> None:
    """
    Full backfill: fetch up to INITIAL_FETCH_LIMIT emails, store them,
    then (re)build the FAISS index from scratch.
    """
    async with _state.lock:
        if _state.status == SyncStatus.RUNNING:
            logger.warning("Sync already running; skipping duplicate trigger.")
            return
        _state.status  = SyncStatus.RUNNING
        _state.task_id = task_id
        _state.message = "Starting initial email backfill …"

    try:
        # ── fetch ────────────────────────────────────────────────────────────
        _state.message = f"Fetching up to {settings.INITIAL_FETCH_LIMIT} emails …"
        loop = asyncio.get_event_loop()

        emails = await loop.run_in_executor(
            None,
            gmail.fetch_emails_initial,
            settings.INITIAL_FETCH_LIMIT,
        )

        # ── store ────────────────────────────────────────────────────────────
        _state.message = f"Saving {len(emails)} emails to database …"
        inserted = await loop.run_in_executor(None, database.bulk_upsert_emails, emails)

        # ── index ─────────────────────────────────────────────────────────── 
        _state.message = "Building FAISS index …"
        all_emails = await loop.run_in_executor(None, database.get_all_emails)
        email_triples = [(r[0], r[1], r[2]) for r in all_emails]
        await loop.run_in_executor(None, embeddings.build_index_from_emails, email_triples)

        # ── finalise ─────────────────────────────────────────────────────────
        _write_last_sync_ts(int(time.time()))
        _state.status  = SyncStatus.DONE
        _state.message = (
            f"Initial sync complete. "
            f"{inserted} new emails stored. "
            f"{embeddings.index_size()} vectors indexed."
        )
        logger.info(_state.message)

    except Exception as exc:
        _state.status  = SyncStatus.ERROR
        _state.message = f"Sync failed: {exc}"
        logger.exception("Initial sync error")


async def run_delta_sync(task_id: str) -> None:
    """
    Incremental sync: fetch emails newer than last_sync_timestamp and
    add them to the existing index without a full rebuild.
    """
    async with _state.lock:
        if _state.status == SyncStatus.RUNNING:
            logger.warning("Sync already running; skipping.")
            return
        _state.status  = SyncStatus.RUNNING
        _state.task_id = task_id
        _state.message = "Starting delta sync …"

    try:
        last_ts = _read_last_sync_ts()
        if last_ts is None:
            # No baseline → fall back to full sync
            await run_initial_sync(task_id)
            return

        loop = asyncio.get_event_loop()

        _state.message = f"Fetching emails after {last_ts} …"
        emails = await loop.run_in_executor(None, gmail.fetch_emails_delta, last_ts)

        if not emails:
            _write_last_sync_ts(int(time.time()))
            _state.status  = SyncStatus.DONE
            _state.message = "Already up to date. No new emails."
            return

        inserted = await loop.run_in_executor(None, database.bulk_upsert_emails, emails)

        _state.message = f"Indexing {inserted} new emails …"
        new_triples = [(id_, subj, body) for id_, subj, body in emails]
        await loop.run_in_executor(None, embeddings.add_emails_to_index, new_triples)

        _write_last_sync_ts(int(time.time()))
        _state.status  = SyncStatus.DONE
        _state.message = (
            f"Delta sync complete. "
            f"{inserted} new emails. "
            f"Index now has {embeddings.index_size()} vectors."
        )
        logger.info(_state.message)

    except Exception as exc:
        _state.status  = SyncStatus.ERROR
        _state.message = f"Delta sync failed: {exc}"
        logger.exception("Delta sync error")
