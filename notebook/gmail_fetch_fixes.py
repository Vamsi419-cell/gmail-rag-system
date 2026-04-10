"""
GMAIL FETCH — FIXED CELLS
Copy these into your gmail_fetch.ipynb notebook to fix all bugs.
Only the cells that CHANGED are shown below.
"""

# ============================================================
# CELL: get_clean_text — FIXED (recursive multipart handling)
# Replace the existing get_clean_text cell with this
# ============================================================

import base64

def get_clean_text(payload):
    """Recursively extracts plain text from email payload.
    Handles nested multipart structures like:
    multipart/mixed → multipart/alternative → text/plain
    """
    
    # Direct body data
    if 'data' in payload.get('body', {}):
        return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='replace')
    
    # Check parts recursively
    if 'parts' in payload:
        for part in payload['parts']:
            mime = part.get('mimeType', '')
            
            # Found plain text directly
            if mime == 'text/plain' and 'data' in part.get('body', {}):
                return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')
            
            # Nested multipart — recurse into it
            if mime.startswith('multipart/'):
                result = get_clean_text(part)
                if result != "Could not find plain text body.":
                    return result
        
        # Fallback: try HTML if no plain text found anywhere
        for part in payload['parts']:
            if part.get('mimeType') == 'text/html' and 'data' in part.get('body', {}):
                html = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')
                # Strip HTML tags for a rough text version
                import re
                return re.sub(r'<[^>]+>', ' ', html)
    
    return "Could not find plain text body."

print("✅ Unscrambler tool is ready (with recursive multipart support)!")


# ============================================================
# CELL: CREATE TABLE — FIXED (close connection)
# Replace the existing CREATE TABLE cell with this
# ============================================================

import sqlite3

conn = sqlite3.connect('rag_database.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS emails (
        id TEXT PRIMARY KEY,
        subject TEXT,
        body TEXT,
        timestamp INTEGER
    )
''')
conn.commit()
conn.close()  # ← FIX: was missing!


# ============================================================
# CELL: BACKFILL (initial fetch) — FIXED (error logging, not silent)
# Replace the existing backfill cell with this
# ============================================================

import time
import json
import sqlite3

SYNC_FILE = 'last_sync.json'
INITIAL_FETCH_LIMIT = 100

print(f"🚀 BOOTSTRAP: Fetching the initial {INITIAL_FETCH_LIMIT} emails...")

results = service.users().messages().list(userId='me', maxResults=INITIAL_FETCH_LIMIT).execute()
messages = results.get('messages', [])

if not messages:
    print("Your inbox is empty. Nothing to fetch.")
else:
    conn = sqlite3.connect('rag_database.db')
    cursor = conn.cursor()
    saved_count = 0
    failed_count = 0

    for message in messages:
        msg_id = message['id']
        try:
            msg_data = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
            payload = msg_data['payload']

            headers = payload.get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
            body_text = get_clean_text(payload)
            
            # Skip emails where body extraction failed
            if body_text == "Could not find plain text body.":
                print(f"  ⚠️ Skipped (no text body): {subject[:60]}")
                failed_count += 1
                continue

            cursor.execute('''
                INSERT OR IGNORE INTO emails (id, subject, body, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (msg_id, subject, body_text, int(time.time())))
            saved_count += 1
        except Exception as e:
            # FIX: Log the error instead of silently passing
            print(f"  ❌ Failed to process {msg_id}: {e}")
            failed_count += 1

    conn.commit()
    conn.close()

    with open(SYNC_FILE, 'w') as f:
        json.dump({'last_sync_time': int(time.time())}, f)

    print(f"✅ Backfill Complete! {saved_count} emails saved, {failed_count} failed/skipped.")
    print("⏰ The sync clock has been started. You are ready for Delta Syncs.")


# ============================================================
# CELL: DELTA SYNC — FIXED (error logging, not silent)
# Replace the existing delta sync cell with this
# ============================================================

import time
import json
import sqlite3
import os

SYNC_FILE = 'last_sync.json'

print("🔄 Waking up Delta Sync Worker...")

if not os.path.exists(SYNC_FILE):
    print("⚠️ Error: No sync clock found. Please run the Backfill cell first!")
else:
    with open(SYNC_FILE, 'r') as f:
        last_sync = json.load(f).get('last_sync_time')

    print(f"Looking for emails received after timestamp: {last_sync}")
    
    results = service.users().messages().list(userId='me', q=f"after:{last_sync}").execute()
    messages = results.get('messages', [])

    if not messages:
        print("✅ Inbox is completely up to date. Going back to sleep.")
    else:
        conn = sqlite3.connect('rag_database.db')
        cursor = conn.cursor()
        saved_count = 0
        failed_count = 0
        
        for message in messages:
            msg_id = message['id']
            try:
                msg_data = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
                payload = msg_data['payload']

                headers = payload.get('headers', [])
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
                body_text = get_clean_text(payload)
                
                if body_text == "Could not find plain text body.":
                    print(f"  ⚠️ Skipped (no text body): {subject[:60]}")
                    failed_count += 1
                    continue

                cursor.execute('''
                    INSERT OR IGNORE INTO emails (id, subject, body, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (msg_id, subject, body_text, int(time.time())))
                saved_count += 1
            except Exception as e:
                # FIX: Log the error instead of silently passing
                print(f"  ❌ Failed to process {msg_id}: {e}")
                failed_count += 1

        conn.commit()
        conn.close()

        print(f"✅ Delta Sync Complete! Added {saved_count} new emails, {failed_count} failed/skipped.")

    # Reset the clock for next time
    with open(SYNC_FILE, 'w') as f:
        json.dump({'last_sync_time': int(time.time())}, f)
    print("🕒 Clock reset.")
