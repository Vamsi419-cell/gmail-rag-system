"""
services/database.py — Thin wrapper around the SQLite email store.

All DB access goes through this module so the rest of the app never
touches raw SQL.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from typing import Generator, List, Optional, Tuple

from config import settings

# ────────────────────────────────────────────────────────────────────────────
# Types
# ────────────────────────────────────────────────────────────────────────────

EmailRow = Tuple[str, str, str, int]   # (id, subject, body, timestamp)


# ────────────────────────────────────────────────────────────────────────────
# Context manager
# ────────────────────────────────────────────────────────────────────────────

@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(settings.DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ────────────────────────────────────────────────────────────────────────────

def ensure_tables() -> None:
    """Create tables if they don't exist yet."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                id        TEXT PRIMARY KEY,
                subject   TEXT,
                body      TEXT,
                timestamp INTEGER
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_emails_timestamp
            ON emails(timestamp)
        """)


# ────────────────────────────────────────────────────────────────────────────
# Reads
# ────────────────────────────────────────────────────────────────────────────

def get_all_emails() -> List[EmailRow]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, subject, body, timestamp FROM emails ORDER BY timestamp DESC"
        ).fetchall()
    return [(r["id"], r["subject"], r["body"], r["timestamp"]) for r in rows]


def count_emails() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]


def email_exists(email_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM emails WHERE id = ?", (email_id,)
        ).fetchone()
    return row is not None


# ────────────────────────────────────────────────────────────────────────────
# Writes
# ────────────────────────────────────────────────────────────────────────────

def upsert_email(email_id: str, subject: str, body: str) -> bool:
    """
    Insert a new email.  Returns True if a new row was inserted,
    False if it already existed (INSERT OR IGNORE).
    """
    ts = int(time.time())
    with get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO emails (id, subject, body, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (email_id, subject, body, ts),
        )
    return cursor.rowcount > 0


def bulk_upsert_emails(
    rows: List[Tuple[str, str, str]]   # (id, subject, body)
) -> int:
    """Insert many emails at once.  Returns count of actually new rows."""
    ts = int(time.time())
    inserted = 0
    with get_conn() as conn:
        for email_id, subject, body in rows:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO emails (id, subject, body, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (email_id, subject, body, ts),
            )
            inserted += cursor.rowcount
    return inserted
