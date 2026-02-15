"""SQLite persistence for chat sessions and messages.

DB file lives at <output_dir>/chats.db alongside the trained model.
Single-user, single-writer — no connection pooling needed.

Uses isolation_level=None (true autocommit) so every write is
immediately visible to subsequent reads. Connections are always
closed after use via the _conn() context manager.
"""
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "output" / "chats.db"


@contextmanager
def _conn():
    conn = sqlite3.connect(str(DB_PATH), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id    TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_chat_id
                ON messages(chat_id)
        """)


# ── Chats ──

def create_chat(title: str = "New Chat") -> dict:
    chat_id = uuid.uuid4().hex[:12]
    now = datetime.now().isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)",
            (chat_id, title, now),
        )
    return {"id": chat_id, "title": title, "created_at": now}


def list_chats() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT c.id, c.title, c.created_at,
                   COUNT(m.id) AS message_count
            FROM chats c
            LEFT JOIN messages m ON c.id = m.chat_id
            GROUP BY c.id
            ORDER BY c.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_chat(chat_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, title, created_at FROM chats WHERE id = ?",
            (chat_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_chat(chat_id: str) -> bool:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    return cur.rowcount > 0


def update_title(chat_id: str, title: str):
    with _conn() as conn:
        conn.execute(
            "UPDATE chats SET title = ? WHERE id = ?", (title, chat_id),
        )


# ── Messages ──

def get_messages(chat_id: str) -> list[dict]:
    """Return conversation in [{role, content}, ...] format for LLM."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def add_message(chat_id: str, role: str, content: str):
    now = datetime.now().isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (chat_id, role, content, now),
        )


def clear_messages(chat_id: str):
    """Delete all messages for a chat (reset)."""
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
