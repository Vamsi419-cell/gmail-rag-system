import os
import sys
import base64
import json
import time
import sqlite3
import re
from pathlib import Path

# Fix Windows console encoding for emoji/unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# =========================
# CONFIG
# =========================
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Anchor all paths to the project root (gmail-rag-system/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "rag_database.db"
SYNC_FILE = DATA_DIR / "last_sync.json"
TOKEN_FILE = DATA_DIR / "token.json"
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"

INITIAL_FETCH_LIMIT = 100


# =========================
# AUTHENTICATION
# =========================
def authenticate_gmail():
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        print(" Authenticating with Gmail...")

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

        print("✅ Authentication successful!")

    return build('gmail', 'v1', credentials=creds)


# =========================
# DB SETUP
# =========================
def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emails (
            id TEXT PRIMARY KEY,
            subject TEXT,
            body TEXT,
            sender TEXT,
            labels TEXT,
            timestamp INTEGER
        )
    ''')

    conn.commit()
    conn.close()


# =========================
# EMAIL PARSING
# =========================
def get_clean_text(payload):
    """Extract plain text from email payload"""

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
# FETCH EMAILS
# =========================
def fetch_messages(service, query=None, max_results=100):
    results = service.users().messages().list(
        userId='me',
        q=query,
        maxResults=max_results
    ).execute()

    return results.get('messages', [])


def process_and_store(service, messages):
    import email
    from email.utils import parsedate_to_datetime

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    saved_count = 0
    failed_count = 0
    latest_time = 0

    for message in messages:
        msg_id = message['id']

        try:
            # Fetch full email
            msg_data = service.users().messages().get(
                userId='me',
                id=msg_id,
                format='full'
            ).execute()

            payload = msg_data['payload']
            headers = payload.get('headers', [])

            # Extract metadata
            subject = next(
                (h['value'] for h in headers if h['name'] == 'Subject'),
                "No Subject"
            )

            sender = next(
                (h['value'] for h in headers if h['name'] == 'From'),
                "Unknown"
            )

            labels = ",".join(msg_data.get("labelIds", []))

            # Extract body
            body = get_clean_text(payload)

            # ✅ BEST timestamp source (IMPORTANT FIX)
            timestamp = int(msg_data['internalDate']) // 1000

            # Track latest email time
            latest_time = max(latest_time, timestamp)

            # Skip empty emails
            if body == "NO_TEXT":
                print(f"⚠️ Skipped (no text): {subject[:50]}")
                failed_count += 1
                continue

            # Insert into DB
            cursor.execute('''
                INSERT OR IGNORE INTO emails 
                (id, subject, body, sender, labels, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (msg_id, subject, body, sender, labels, timestamp))

            saved_count += 1

        except Exception as e:
            print(f"❌ Error processing {msg_id}: {e}")
            failed_count += 1

    conn.commit()
    conn.close()

    return saved_count, failed_count, latest_time

# =========================
# BACKFILL (FIRST RUN)
# =========================
def backfill(service):
    print(f"📥 Backfilling {INITIAL_FETCH_LIMIT} emails...")

    messages = fetch_messages(service, max_results=INITIAL_FETCH_LIMIT)

    if not messages:
        print("Inbox empty.")
        return

    saved, failed, last_time = process_and_store(service, messages)

    with open(SYNC_FILE, 'w') as f:
        json.dump({'last_sync_time': last_time}, f)

    print(f"✅ Backfill done → {saved} saved, {failed} failed")


# =========================
# DELTA SYNC
# =========================
def delta_sync(service):
    print("🔄 Running delta sync...")

    if not SYNC_FILE.exists():
        print("⚠️ No sync file found. Run backfill first.")
        return

    with open(SYNC_FILE, 'r') as f:
        last_sync = json.load(f)['last_sync_time']

    # Gmail query for new emails
    query = f"after:{last_sync - 10}"  # small buffer to avoid missing emails

    messages = fetch_messages(service, query=query)

    if not messages:
        print("✅ No new emails.")
        return

    saved, failed, last_time = process_and_store(service, messages)

    print(f"✅ Delta sync → {saved} new, {failed} failed")

    # Update sync time with latest processed email
    with open(SYNC_FILE, 'w') as f:
        json.dump({'last_sync_time': last_time}, f)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    service = authenticate_gmail()
    init_db()

    # First run → backfill
    if not SYNC_FILE.exists():
        backfill(service)
    else:
        delta_sync(service)