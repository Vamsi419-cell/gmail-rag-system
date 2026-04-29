import os
import sys
import base64
import json
import sqlite3
import re
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for emoji/unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from config import DB_PATH, DATA_DIR, get_user_paths

# =========================
# CONFIG
# =========================
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
DEFAULT_FETCH_LIMIT = 500


# =========================
# DB SETUP
# =========================
def init_db():
    """Create users and emails tables. Migrates old schema if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                google_id TEXT UNIQUE NOT NULL,
                email TEXT,
                name TEXT,
                access_token TEXT,
                refresh_token TEXT,
                token_expiry TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Check if old emails table exists without user_id column
        cursor.execute("PRAGMA table_info(emails)")
        columns = [row[1] for row in cursor.fetchall()]

        if columns and "user_id" not in columns:
            print("⚠️ Migrating old emails table → adding user_id support")
            cursor.execute("DROP TABLE emails")

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emails (
                id TEXT,
                user_id INTEGER,
                subject TEXT,
                body TEXT,
                sender TEXT,
                labels TEXT,
                timestamp INTEGER,
                PRIMARY KEY (id, user_id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Index for fast per-user queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_emails_user_id ON emails(user_id)
        ''')

        conn.commit()


# =========================
# USER MANAGEMENT
# =========================
def upsert_user(google_id, email, name, access_token, refresh_token, token_expiry):
    """
    Create or update a user record. Returns the integer user_id.
    On conflict (same google_id), update tokens and profile info.
    """
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO users (google_id, email, name, access_token, refresh_token, token_expiry)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(google_id) DO UPDATE SET
                email = excluded.email,
                name = excluded.name,
                access_token = excluded.access_token,
                refresh_token = COALESCE(excluded.refresh_token, users.refresh_token),
                token_expiry = excluded.token_expiry
        ''', (google_id, email, name, access_token, refresh_token, token_expiry))

        conn.commit()

        cursor.execute('SELECT id FROM users WHERE google_id = ?', (google_id,))
        user_id = cursor.fetchone()[0]

    return user_id


def get_user_tokens(user_id):
    """Retrieve stored OAuth tokens for a user."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT access_token, refresh_token, token_expiry FROM users WHERE id = ?',
            (user_id,)
        )
        row = cursor.fetchone()

    if row:
        return {"access_token": row[0], "refresh_token": row[1], "token_expiry": row[2]}
    return None


def update_user_tokens(user_id, access_token, refresh_token, token_expiry):
    """Update stored OAuth tokens after a refresh."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET access_token = ?, refresh_token = COALESCE(?, refresh_token), token_expiry = ?
            WHERE id = ?
        ''', (access_token, refresh_token, token_expiry, user_id))
        conn.commit()


def get_user_email_count(user_id):
    """Return the number of emails stored for a user."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM emails WHERE user_id = ?', (user_id,))
        return cursor.fetchone()[0]


# =========================
# AUTHENTICATION
# =========================
def authenticate_gmail(user_id):
    """
    Build a Gmail API service using stored OAuth tokens.
    Automatically refreshes expired tokens.
    """
    tokens = get_user_tokens(user_id)
    if not tokens:
        raise ValueError(f"No tokens found for user {user_id}. User must re-authenticate.")

    creds = Credentials(
        token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=SCOPES,
    )

    # Auto-refresh if expired
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        update_user_tokens(
            user_id,
            creds.token,
            creds.refresh_token,
            creds.expiry.isoformat() if creds.expiry else None,
        )

    return build('gmail', 'v1', credentials=creds)


# =========================
# EMAIL PARSING
# =========================
def get_clean_text(payload):
    """Extract plain text from email payload (recursive)."""

    if 'data' in payload.get('body', {}):
        return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='replace')

    if 'parts' in payload:
        for part in payload['parts']:
            mime = part.get('mimeType', '')

            if mime == 'text/plain' and 'data' in part.get('body', {}):
                return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')

            if mime.startswith('multipart/'):
                result = get_clean_text(part)
                if result != "NO_TEXT":
                    return result

        # fallback to HTML
        for part in payload['parts']:
            if part.get('mimeType') == 'text/html' and 'data' in part.get('body', {}):
                html = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')
                return re.sub(r'<[^>]+>', ' ', html)

    return "NO_TEXT"


# =========================
# FETCH EMAILS (WITH PAGINATION)
# =========================
def fetch_messages(service, query=None, max_results=100):
    """
    List message IDs from the user's inbox.
    Handles Gmail pagination via nextPageToken to fetch all requested messages.
    """
    all_messages = []
    page_token = None

    while len(all_messages) < max_results:
        batch_size = min(max_results - len(all_messages), 100)  # Gmail max per page is 100

        kwargs = {
            "userId": "me",
            "maxResults": batch_size,
        }
        if query:
            kwargs["q"] = query
        if page_token:
            kwargs["pageToken"] = page_token

        results = service.users().messages().list(**kwargs).execute()

        messages = results.get("messages", [])
        all_messages.extend(messages)

        page_token = results.get("nextPageToken")
        if not page_token:
            break

    return all_messages[:max_results]


def process_and_store(service, messages, user_id):
    """Fetch full email data and store in SQLite with user_id."""
    saved_count = 0
    skipped_count = 0
    failed_count = 0
    latest_time = 0

    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()

        for message in messages:
            msg_id = message['id']

            try:
                msg_data = service.users().messages().get(
                    userId='me',
                    id=msg_id,
                    format='full'
                ).execute()

                payload = msg_data['payload']
                headers = payload.get('headers', [])

                subject = next(
                    (h['value'] for h in headers if h['name'] == 'Subject'),
                    "No Subject"
                )

                sender = next(
                    (h['value'] for h in headers if h['name'] == 'From'),
                    "Unknown"
                )

                labels = ",".join(msg_data.get("labelIds", []))

                body = get_clean_text(payload)

                timestamp = int(msg_data['internalDate']) // 1000

                latest_time = max(latest_time, timestamp)

                if body == "NO_TEXT":
                    print(f"⚠️ Skipped (no text): {subject[:50]}")
                    skipped_count += 1
                    continue

                cursor.execute('''
                    INSERT OR IGNORE INTO emails
                    (id, user_id, subject, body, sender, labels, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (msg_id, user_id, subject, body, sender, labels, timestamp))

                # Check if actually inserted (not a duplicate)
                if cursor.rowcount > 0:
                    saved_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                print(f"❌ Error processing {msg_id}: {e}")
                failed_count += 1

        conn.commit()

    return saved_count, skipped_count, failed_count, latest_time


# =========================
# BACKFILL (FIRST RUN)
# =========================
def backfill(service, user_id, limit=None):
    """Fetch the last N emails for a user (initial population)."""
    fetch_limit = limit or DEFAULT_FETCH_LIMIT
    print(f"📥 Backfilling {fetch_limit} emails for user {user_id}...")

    messages = fetch_messages(service, max_results=fetch_limit)

    if not messages:
        return {"status": "empty", "message": "Inbox is empty.", "saved": 0, "failed": 0}

    saved, skipped, failed, last_time = process_and_store(service, messages, user_id)

    # Save sync state to user-specific file
    paths = get_user_paths(user_id)
    with open(paths["sync_file"], 'w') as f:
        json.dump({'last_sync_time': last_time}, f)

    total_stored = get_user_email_count(user_id)
    msg = f"Backfill complete — {saved} new, {skipped} skipped, {failed} failed ({total_stored} total in DB)"
    print(f"✅ {msg}")
    return {"status": "ok", "message": msg, "saved": saved, "failed": failed}


# =========================
# DELTA SYNC
# =========================
def delta_sync(service, user_id):
    """Fetch only new emails since last sync for a user."""
    print(f"🔄 Running delta sync for user {user_id}...")

    paths = get_user_paths(user_id)
    sync_file = paths["sync_file"]

    if not sync_file.exists():
        return {"status": "error", "message": "No sync history found. Run Initial Fetch first."}

    with open(sync_file, 'r') as f:
        last_sync = json.load(f)['last_sync_time']

    query = f"after:{last_sync - 10}"
    messages = fetch_messages(service, query=query)

    if not messages:
        return {"status": "ok", "message": "No new emails found.", "saved": 0, "failed": 0}

    saved, skipped, failed, last_time = process_and_store(service, messages, user_id)

    total_stored = get_user_email_count(user_id)
    msg = f"Delta sync — {saved} new, {skipped} skipped, {failed} failed ({total_stored} total in DB)"
    print(f"✅ {msg}")

    if last_time > 0:
        with open(sync_file, 'w') as f:
            json.dump({'last_sync_time': last_time}, f)

    return {"status": "ok", "message": msg, "saved": saved, "failed": failed}