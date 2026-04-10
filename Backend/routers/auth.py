"""
routers/auth.py — Gmail OAuth endpoints.

GET  /auth/status  → check whether token.json is valid
POST /auth/login   → trigger the OAuth browser flow (blocks until the user
                     completes it in their browser, so call from a script or
                     the frontend opens a pop-up window)
POST /auth/logout  → delete token.json
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from config import settings
from schemas import AuthStatusResponse, StatusResponse
from services import gmail

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Auth"])


@router.get("/status", response_model=AuthStatusResponse)
async def auth_status():
    """Returns whether the app has a valid Gmail token."""
    authed, email = gmail.is_authenticated()
    return AuthStatusResponse(authenticated=authed, email=email)


@router.post("/login", response_model=StatusResponse)
async def login():
    """
    Launches the OAuth consent flow in the user's default browser.
    The call blocks until the user grants permission (or times out).
    Suitable for a desktop / localhost deployment.
    """
    if not settings.CREDENTIALS_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"credentials.json not found at {settings.CREDENTIALS_PATH}. "
                "Download it from the Google Cloud Console and place it in the "
                "notebook/ directory."
            ),
        )

    try:
        email = gmail.run_oauth_flow()
        return StatusResponse(
            status="ok",
            message=f"Authenticated as {email}",
            data={"email": email},
        )
    except Exception as exc:
        logger.exception("OAuth flow failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/logout", response_model=StatusResponse)
async def logout():
    """Revokes the local token (the user will need to log in again)."""
    token_path = settings.TOKEN_PATH
    if token_path.exists():
        token_path.unlink()
        return StatusResponse(status="ok", message="Logged out. token.json deleted.")
    return StatusResponse(status="ok", message="Already logged out.")
