"""
services/gmail.py — Gmail OAuth flow and email fetching logic.

Keeps all Google API concerns in one place.
"""
from __future__ import annotations

import base64
import logging
import re
from typing import List, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Lazy imports so the app starts even if google libs aren't installed yet
# ────────────────────────────────────────────────────────────────────────────

def _get_google_libs():
    from google.auth.transport.requests import Request          # type: ignore
    from google.oauth2.credentials import Credentials           # type: ignore
    from google_auth_oauthlib.flow import InstalledAppFlow      # type: ignore
    from googleapiclient.discovery import build                 # type: ignore
    return Request, Credentials, InstalledAppFlow, build


# ────────────────────────────────────────────────────────────────────────────
# Auth helpers
# ────────────────────────────────────────────────────────────────────────────

def get_credentials():
    """
    Load saved creds from token.json, refresh if expired.
    Returns None if the user has never authorised.
    """
    Request, Credentials, InstalledAppFlow, _ = _get_google_libs()

    token_path = settings.TOKEN_PATH
    if not token_path.exists():
        return None

    creds = Credentials.from_authorized_user_file(str(token_path), settings.GMAIL_SCOPES)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_credentials(creds)
        except Exception as exc:
            logger.error("Token refresh failed: %s", exc)
            return None

    return creds if (creds and creds.valid) else None


def _save_credentials(creds) -> None:
    settings.TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(settings.TOKEN_PATH), "w") as f:
        f.write(creds.to_json())


def run_oauth_flow() -> str:
    """
    Starts the local OAuth server flow.
    Returns the authorised user's email address.
    Raises RuntimeError if credentials.json is missing.
    """
    _, _, InstalledAppFlow, build = _get_google_libs()

    if not settings.CREDENTIALS_PATH.exists():
        raise RuntimeError(
            f"credentials.json not found at {settings.CREDENTIALS_PATH}. "
            "Download it from Google Cloud Console → APIs & Services → Credentials."
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        str(settings.CREDENTIALS_PATH), settings.GMAIL_SCOPES
    )
    creds = flow.run_local_server(port=settings.OAUTH_REDIRECT_PORT, open_browser=True)
    _save_credentials(creds)

    # Retrieve email for confirmation
    service = build("gmail", "v1", credentials=creds)
    profile = service.users().getProfile(userId="me").execute()
    return profile.get("emailAddress", "unknown")


def is_authenticated() -> Tuple[bool, Optional[str]]:
    """Returns (is_authed, email_or_None)."""
    creds = get_credentials()
    if not creds:
        return False, None
    try:
        _, _, _, build = _get_google_libs()
        service = build("gmail", "v1", credentials=creds)
        profile = service.users().getProfile(userId="me").execute()
        return True, profile.get("emailAddress")
    except Exception:
        return False, None


# ────────────────────────────────────────────────────────────────────────────
# Email fetching
# ────────────────────────────────────────────────────────────────────────────

def _build_service():
    creds = get_credentials()
    if not creds:
        raise RuntimeError("Not authenticated. Call /auth/login first.")
    _, _, _, build = _get_google_libs()
    return build("gmail", "v1", credentials=creds)


def _extract_plain_text(payload: dict) -> str:
    """Recursively dig through MIME parts to find text/plain body."""
    mime = payload.get("mimeType", "")

    # Direct body data
    body_data = payload.get("body", {}).get("data")
    if body_data and mime == "text/plain":
        return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")

    # Recurse into parts
    for part in payload.get("parts", []):
        result = _extract_plain_text(part)
        if result:
            return result

    return ""


def _clean_body(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"---------- Forwarded message ---------.*", "", text, flags=re.DOTALL)
    text = re.sub(r"On .{1,100} wrote:.*", "", text, flags=re.DOTALL)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_emails_initial(limit: int = 100) -> List[Tuple[str, str, str]]:
    """
    Pull the most recent `limit` emails.
    Returns list of (id, subject, clean_body).
    """
    service = _build_service()
    results = service.users().messages().list(userId="me", maxResults=limit).execute()
    messages = results.get("messages", [])

    emails: List[Tuple[str, str, str]] = []
    for msg in messages:
        try:
            emails.append(_fetch_single(service, msg["id"]))
        except Exception as exc:
            logger.warning("Skipping message %s: %s", msg["id"], exc)

    return emails


def fetch_emails_delta(after_timestamp: int) -> List[Tuple[str, str, str]]:
    """
    Pull only emails received after `after_timestamp` (Unix epoch).
    Returns list of (id, subject, clean_body).
    """
    service = _build_service()
    results = service.users().messages().list(
        userId="me", q=f"after:{after_timestamp}"
    ).execute()
    messages = results.get("messages", [])

    emails: List[Tuple[str, str, str]] = []
    for msg in messages:
        try:
            emails.append(_fetch_single(service, msg["id"]))
        except Exception as exc:
            logger.warning("Skipping message %s: %s", msg["id"], exc)

    return emails


def _fetch_single(service, msg_id: str) -> Tuple[str, str, str]:
    data = service.users().messages().get(
        userId="me", id=msg_id, format="full"
    ).execute()
    payload = data["payload"]
    headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
    subject = headers.get("Subject", "No Subject")
    body = _clean_body(_extract_plain_text(payload))
    return msg_id, subject, body
