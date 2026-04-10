"""
routers/emails.py — Email sync management endpoints.

POST /emails/sync/initial   — full backfill (use once on first run)
POST /emails/sync/delta     — incremental update (call on a schedule)
GET  /emails/sync/status    — poll for progress
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from schemas import SyncStatus, SyncStatusResponse, TriggerSyncResponse
from services import sync as sync_svc

router = APIRouter(prefix="/emails", tags=["Emails"])


@router.post("/sync/initial", response_model=TriggerSyncResponse)
async def trigger_initial_sync(background_tasks: BackgroundTasks):
    """
    Kicks off a full email backfill + FAISS index build in the background.
    Returns immediately — poll /emails/sync/status to track progress.
    """
    status = sync_svc.get_status()
    if status["status"] == SyncStatus.RUNNING:
        raise HTTPException(status_code=409, detail="A sync is already running.")

    task_id = str(uuid.uuid4())
    background_tasks.add_task(sync_svc.run_initial_sync, task_id)

    return TriggerSyncResponse(
        message="Initial sync started in the background.",
        task_id=task_id,
    )


@router.post("/sync/delta", response_model=TriggerSyncResponse)
async def trigger_delta_sync(background_tasks: BackgroundTasks):
    """
    Incrementally fetches emails received since the last sync.
    If no baseline exists it automatically falls back to a full backfill.
    """
    status = sync_svc.get_status()
    if status["status"] == SyncStatus.RUNNING:
        raise HTTPException(status_code=409, detail="A sync is already running.")

    task_id = str(uuid.uuid4())
    background_tasks.add_task(sync_svc.run_delta_sync, task_id)

    return TriggerSyncResponse(
        message="Delta sync started in the background.",
        task_id=task_id,
    )


@router.get("/sync/status", response_model=SyncStatusResponse)
async def sync_status():
    """
    Returns the current sync state plus DB / index statistics.
    Poll this every few seconds from the frontend while a sync is running.
    """
    s = sync_svc.get_status()
    return SyncStatusResponse(**s)
