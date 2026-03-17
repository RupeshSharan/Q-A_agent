"""SQLite helpers for the DocMind backend."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "database.db"


def get_db_path() -> Path:
    """Resolve database path from env or default project root path."""
    raw_path = os.getenv("DOCMIND_DB_PATH", "").strip()
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return DEFAULT_DB_PATH


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row access by column name."""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(str(row["name"]) == column_name for row in rows)


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_def: str, column_name: str) -> None:
    if not _column_exists(conn, table_name, column_name):
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_def}")


def init_db() -> None:
    """Initialize all backend tables if they do not exist yet."""
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('admin', 'customer')),
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS admins (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'admin' CHECK (role IN ('superadmin', 'admin')),
                is_active INTEGER NOT NULL DEFAULT 1,
                portal_role TEXT NOT NULL DEFAULT 'admin' CHECK (portal_role IN ('admin', 'staff')),
                created_by TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS customers (
                id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT UNIQUE,
                phone TEXT,
                password_hash TEXT,
                is_guest INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                uploaded_by TEXT,
                uploaded_at TEXT NOT NULL,
                chunks_indexed INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (uploaded_by) REFERENCES admins(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                customer_id TEXT,
                session_token TEXT NOT NULL UNIQUE,
                admin_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE SET NULL,
                FOREIGN KEY (admin_id) REFERENCES admins(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                sources TEXT,
                session_id TEXT,
                customer_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS bookings (
                id TEXT PRIMARY KEY,
                customer_id TEXT,
                guest_name TEXT,
                guest_email TEXT,
                guest_phone TEXT,
                service_type TEXT,
                booking_date TEXT NOT NULL,
                booking_time TEXT NOT NULL,
                message TEXT,
                status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'cancelled')),
                admin_note TEXT,
                notified_via TEXT CHECK (notified_via IN ('email', 'whatsapp', 'both')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS booking_slots (
                id TEXT PRIMARY KEY,
                slot_date TEXT NOT NULL,
                slot_time TEXT NOT NULL,
                is_booked INTEGER NOT NULL DEFAULT 0,
                is_blocked INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE (slot_date, slot_time)
            );

            CREATE TABLE IF NOT EXISTS notifications (
                id TEXT PRIMARY KEY,
                booking_id TEXT,
                recipient_email TEXT,
                recipient_phone TEXT,
                channel TEXT NOT NULL CHECK (channel IN ('email', 'whatsapp')),
                message_body TEXT,
                status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed')),
                sent_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (booking_id) REFERENCES bookings(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS document_suggestions (
                id TEXT PRIMARY KEY,
                document_id INTEGER NOT NULL,
                suggested_by TEXT NOT NULL,
                suggestion_text TEXT NOT NULL,
                new_content TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
                reviewed_by TEXT,
                admin_note TEXT,
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY (suggested_by) REFERENCES admins(id) ON DELETE CASCADE,
                FOREIGN KEY (reviewed_by) REFERENCES admins(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS organisation_suggestions (
                id TEXT PRIMARY KEY,
                suggested_by TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL CHECK (category IN ('service', 'timing', 'pricing', 'other')),
                status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
                reviewed_by TEXT,
                admin_note TEXT,
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                FOREIGN KEY (suggested_by) REFERENCES admins(id) ON DELETE CASCADE,
                FOREIGN KEY (reviewed_by) REFERENCES admins(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
            CREATE INDEX IF NOT EXISTS idx_documents_uploaded_by ON documents(uploaded_by);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id ON chat_messages(user_id);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at);
            CREATE INDEX IF NOT EXISTS idx_admins_email ON admins(email);
            CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
            CREATE INDEX IF NOT EXISTS idx_customers_phone ON customers(phone);
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_token ON chat_sessions(session_token);
            CREATE INDEX IF NOT EXISTS idx_bookings_customer_id ON bookings(customer_id);
            CREATE INDEX IF NOT EXISTS idx_bookings_status ON bookings(status);
            CREATE INDEX IF NOT EXISTS idx_bookings_created_at ON bookings(created_at);
            CREATE INDEX IF NOT EXISTS idx_slots_datetime ON booking_slots(slot_date, slot_time);
            CREATE INDEX IF NOT EXISTS idx_notifications_booking_id ON notifications(booking_id);
            CREATE INDEX IF NOT EXISTS idx_document_suggestions_status ON document_suggestions(status);
            CREATE INDEX IF NOT EXISTS idx_document_suggestions_suggested_by ON document_suggestions(suggested_by);
            CREATE INDEX IF NOT EXISTS idx_org_suggestions_status ON organisation_suggestions(status);
            CREATE INDEX IF NOT EXISTS idx_org_suggestions_suggested_by ON organisation_suggestions(suggested_by);
            """
        )

        # Compatibility columns for legacy tables.
        _ensure_column(conn, "admins", "portal_role TEXT NOT NULL DEFAULT 'admin'", "portal_role")
        _ensure_column(conn, "admins", "created_by TEXT", "created_by")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_admins_portal_role ON admins(portal_role)")
        _ensure_column(conn, "documents", "file_path TEXT", "file_path")
        _ensure_column(conn, "documents", "file_size INTEGER", "file_size")
        _ensure_column(conn, "documents", "chunk_count INTEGER", "chunk_count")
        _ensure_column(conn, "documents", "is_active INTEGER NOT NULL DEFAULT 1", "is_active")
        _ensure_column(conn, "documents", "created_at TEXT", "created_at")
        _ensure_column(conn, "chat_messages", "sources TEXT", "sources")
        _ensure_column(conn, "chat_messages", "session_id TEXT", "session_id")
        _ensure_column(conn, "chat_messages", "customer_id TEXT", "customer_id")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id)")

        # Backfill compatibility columns from legacy values.
        conn.execute("UPDATE documents SET file_path = stored_path WHERE file_path IS NULL")
        conn.execute("UPDATE documents SET chunk_count = chunks_indexed WHERE chunk_count IS NULL")
        conn.execute("UPDATE documents SET created_at = uploaded_at WHERE created_at IS NULL")
        conn.execute("UPDATE chat_messages SET sources = sources_json WHERE sources IS NULL")
        conn.execute(
            """
            UPDATE admins
            SET portal_role = CASE
                WHEN portal_role IS NULL OR portal_role = '' THEN 'admin'
                ELSE portal_role
            END
            """
        )

        # Best-effort migration from legacy `users` table.
        if _table_exists(conn, "users"):
            conn.execute(
                """
                INSERT OR IGNORE INTO admins (id, name, email, password_hash, role, is_active, portal_role, created_at)
                SELECT
                    'legacy-admin-' || CAST(id AS TEXT),
                    COALESCE(NULLIF(substr(email, 1, instr(email, '@') - 1), ''), 'Admin'),
                    email,
                    password_hash,
                    'admin',
                    1,
                    'admin',
                    created_at
                FROM users
                WHERE role = 'admin'
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO customers (id, name, email, phone, password_hash, is_guest, created_at)
                SELECT
                    'legacy-customer-' || CAST(id AS TEXT),
                    NULL,
                    email,
                    NULL,
                    password_hash,
                    0,
                    created_at
                FROM users
                WHERE role = 'customer'
                """
            )
