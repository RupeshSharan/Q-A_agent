"""FastAPI backend for DocMind AI admin/customer portals."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import uuid
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .auth import AuthUser, get_current_user, get_optional_user, require_admin, require_portal_user
from .db import get_conn, init_db
from .rag_runtime import build_pipeline, ensure_backend_dirs, get_runtime, get_storage_dir, has_gemini_key
from .schemas import (
    AskRequest,
    AskResponse,
    AdminAccountOut,
    AuthResponse,
    BookingCreateRequest,
    BookingCustomerUpdateRequest,
    BookingOut,
    BookingSlotCreateRequest,
    BookingSlotOut,
    BookingSlotUpdateRequest,
    BookingStatusUpdateRequest,
    ChatMessageOut,
    ChatSessionOut,
    DocumentOut,
    DocumentSuggestionCreateRequest,
    DocumentSuggestionOut,
    DocumentSuggestionReviewRequest,
    HealthResponse,
    LoginRequest,
    NotificationOut,
    OrganisationSuggestionCreateRequest,
    OrganisationSuggestionOut,
    OrganisationSuggestionReviewRequest,
    RegisterRequest,
    SourceSnippet,
    StaffCreateRequest,
    StaffStatusUpdateRequest,
    UserPublic,
)
from .security import create_access_token, hash_password, verify_password

raw_origins = os.getenv("DOCMIND_CORS_ORIGINS", "*").strip()
allow_origins = ["*"] if raw_origins == "*" else [item.strip() for item in raw_origins.split(",") if item.strip()]

app = FastAPI(
    title="DocMind API",
    version="0.2.0",
    description="Backend for DocMind admin and customer portals.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKDAY_START = time(hour=9, minute=0)
WORKDAY_END = time(hour=17, minute=0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


def _sanitize_filename(filename: str) -> str:
    clean = Path(filename).name.strip()
    return clean or f"document-{uuid.uuid4().hex}.pdf"


def _parse_booking_date(raw_value: str) -> date:
    try:
        return date.fromisoformat(raw_value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid booking_date. Expected YYYY-MM-DD format.",
        ) from exc


def _parse_booking_time(raw_value: str) -> time:
    value = raw_value.strip()
    if not value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid booking_time. Expected HH:MM format.",
        )

    try:
        parsed = time.fromisoformat(value)
    except ValueError:
        try:
            parsed = datetime.strptime(value, "%H:%M").time()
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid booking_time. Expected HH:MM format.",
            ) from exc

    if parsed.tzinfo is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="booking_time must be in local HH:MM format (timezone offsets are not allowed).",
        )
    return parsed.replace(second=0, microsecond=0)


def _validate_workday_slot(raw_date: str, raw_time: str) -> tuple[str, str]:
    parsed_date = _parse_booking_date(raw_date)
    parsed_time = _parse_booking_time(raw_time)

    if parsed_date.weekday() == 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bookings are unavailable on Sunday. Please choose Monday to Saturday.",
        )

    if parsed_time < WORKDAY_START or parsed_time > WORKDAY_END:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bookings are allowed only between 09:00 and 17:00.",
        )

    return parsed_date.isoformat(), parsed_time.strftime("%H:%M")


def _find_slot(conn: sqlite3.Connection, slot_date: str, slot_time: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT id, is_booked, is_blocked
        FROM booking_slots
        WHERE slot_date = ? AND slot_time = ?
        LIMIT 1
        """,
        (slot_date, slot_time),
    ).fetchone()


def _assert_no_active_booking_conflict(
    conn: sqlite3.Connection,
    slot_date: str,
    slot_time: str,
    *,
    exclude_booking_id: str | None = None,
) -> None:
    query = """
        SELECT id
        FROM bookings
        WHERE booking_date = ? AND booking_time = ? AND status != 'cancelled'
    """
    params: list[str] = [slot_date, slot_time]
    if exclude_booking_id is not None:
        query += " AND id != ?"
        params.append(exclude_booking_id)

    conflict = conn.execute(f"{query} LIMIT 1", tuple(params)).fetchone()
    if conflict is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Selected slot is already booked")


def _reserve_slot(conn: sqlite3.Connection, slot_date: str, slot_time: str) -> None:
    slot = _find_slot(conn, slot_date, slot_time)
    if slot is not None:
        if bool(slot["is_blocked"]):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Selected slot is blocked")
        if bool(slot["is_booked"]):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Selected slot is already booked")
        conn.execute("UPDATE booking_slots SET is_booked = 1 WHERE id = ?", (slot["id"],))
        return

    conn.execute(
        """
        INSERT INTO booking_slots (id, slot_date, slot_time, is_booked, is_blocked, created_at)
        VALUES (?, ?, ?, 1, 0, ?)
        """,
        (_new_id(), slot_date, slot_time, _utc_now()),
    )


def _mark_slot_booked(conn: sqlite3.Connection, slot_date: str, slot_time: str) -> None:
    slot = _find_slot(conn, slot_date, slot_time)
    if slot is not None:
        conn.execute("UPDATE booking_slots SET is_booked = 1 WHERE id = ?", (slot["id"],))
        return

    conn.execute(
        """
        INSERT INTO booking_slots (id, slot_date, slot_time, is_booked, is_blocked, created_at)
        VALUES (?, ?, ?, 1, 0, ?)
        """,
        (_new_id(), slot_date, slot_time, _utc_now()),
    )


def _release_slot(conn: sqlite3.Connection, slot_date: str, slot_time: str) -> None:
    slot = _find_slot(conn, slot_date, slot_time)
    if slot is not None:
        conn.execute("UPDATE booking_slots SET is_booked = 0 WHERE id = ?", (slot["id"],))


def _normalize_portal_role(raw_role: Any) -> str:
    role = str(raw_role).strip().lower() if raw_role is not None else ""
    if role in {"admin", "staff", "customer"}:
        return role
    if role in {"superadmin", "admin"}:
        return "admin"
    return "customer"


def _user_public_from_row(row: sqlite3.Row) -> UserPublic:
    name = str(row["name"]) if "name" in row.keys() and row["name"] is not None else None
    role_value = row["portal_role"] if "portal_role" in row.keys() else row["role"]
    return UserPublic(
        id=str(row["id"]),
        name=name,
        email=str(row["email"]),
        role=_normalize_portal_role(role_value),
        created_at=str(row["created_at"]),
    )


def _admin_account_from_row(row: sqlite3.Row) -> AdminAccountOut:
    return AdminAccountOut(
        id=str(row["id"]),
        name=str(row["name"]),
        email=str(row["email"]),
        role=_normalize_portal_role(row["portal_role"]),
        is_active=bool(row["is_active"]),
        created_by=str(row["created_by"]) if row["created_by"] is not None else None,
        created_by_email=str(row["created_by_email"]) if row["created_by_email"] is not None else None,
        created_at=str(row["created_at"]),
    )


def _document_from_row(row: sqlite3.Row) -> DocumentOut:
    return DocumentOut(
        id=int(row["id"]),
        filename=str(row["filename"]),
        stored_path=str(row["stored_path"]),
        uploaded_by=str(row["uploaded_by"]) if row["uploaded_by"] is not None else None,
        uploaded_by_email=str(row["uploaded_by_email"]) if row["uploaded_by_email"] is not None else None,
        uploaded_at=str(row["uploaded_at"]),
        chunks_indexed=int(row["chunks_indexed"]),
    )


def _normalize_sources(raw_sources: Any) -> list[SourceSnippet]:
    if not isinstance(raw_sources, list):
        return []

    normalized: list[SourceSnippet] = []
    for item in raw_sources:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", ""))
        source = str(item.get("source", "unknown"))
        chunk_index_raw = item.get("chunk_index", -1)
        distance_raw = item.get("distance")
        try:
            chunk_index = int(chunk_index_raw)
        except Exception:
            chunk_index = -1
        distance: float | None
        try:
            distance = float(distance_raw) if distance_raw is not None else None
        except Exception:
            distance = None
        normalized.append(
            SourceSnippet(
                text=text,
                source=source,
                chunk_index=chunk_index,
                distance=distance,
            )
        )
    return normalized


def _chat_from_row(row: sqlite3.Row) -> ChatMessageOut:
    raw_sources = row["sources_json"]
    if raw_sources is None:
        raw_sources = row["sources"]
    try:
        parsed = json.loads(str(raw_sources)) if raw_sources else []
    except Exception:
        parsed = []

    user_email = str(row["user_email"]) if row["user_email"] else None
    if user_email in {"", "guest@docmind.local"}:
        user_email = None

    user_id = str(row["user_id"]) if row["user_id"] else None
    if user_id == "":
        user_id = None

    return ChatMessageOut(
        id=int(row["id"]),
        session_id=str(row["session_id"]) if row["session_id"] else None,
        customer_id=str(row["customer_id"]) if row["customer_id"] else None,
        user_id=user_id,
        user_email=user_email,
        question=str(row["question"]),
        answer=str(row["answer"]),
        sources=_normalize_sources(parsed),
        created_at=str(row["created_at"]),
    )


def _legacy_role_for_chat(role: str | None) -> str:
    if role in {"admin", "staff", "superadmin"}:
        return "admin"
    return "customer"


def _get_or_create_legacy_user_id(conn: sqlite3.Connection, email: str, role: str) -> int:
    existing = conn.execute("SELECT id FROM users WHERE email = ? LIMIT 1", (email,)).fetchone()
    if existing is not None:
        return int(existing["id"])

    placeholder_hash = hash_password(uuid.uuid4().hex)
    created_at = _utc_now()
    try:
        cursor = conn.execute(
            """
            INSERT INTO users (email, password_hash, role, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (email, placeholder_hash, role, created_at),
        )
        return int(cursor.lastrowid)
    except sqlite3.IntegrityError:
        existing = conn.execute("SELECT id FROM users WHERE email = ? LIMIT 1", (email,)).fetchone()
        if existing is None:
            raise
        return int(existing["id"])


def _resolve_chat_legacy_user_id(conn: sqlite3.Connection, actor: dict[str, str | None]) -> int:
    email = (actor.get("user_email") or "").strip().lower()
    role = _legacy_role_for_chat(actor.get("role"))
    if email:
        return _get_or_create_legacy_user_id(conn, email, role)
    return _get_or_create_legacy_user_id(conn, "guest@docmind.local", "customer")


def _documents_use_legacy_user_fk(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("PRAGMA foreign_key_list(documents)").fetchall()
    for row in rows:
        try:
            if str(row["from"]) == "uploaded_by" and str(row["table"]) == "users":
                return True
        except Exception:
            continue
    return False


def _resolve_document_uploader_id(conn: sqlite3.Connection, current_admin: AuthUser) -> str:
    if _documents_use_legacy_user_fk(conn):
        legacy_id = _get_or_create_legacy_user_id(conn, str(current_admin["email"]).strip().lower(), "admin")
        return str(legacy_id)
    return str(current_admin["id"])


def _booking_from_row(row: sqlite3.Row) -> BookingOut:
    return BookingOut(
        id=str(row["id"]),
        customer_id=str(row["customer_id"]) if row["customer_id"] else None,
        guest_name=str(row["guest_name"]) if row["guest_name"] else None,
        guest_email=str(row["guest_email"]) if row["guest_email"] else None,
        guest_phone=str(row["guest_phone"]) if row["guest_phone"] else None,
        service_type=str(row["service_type"]) if row["service_type"] else None,
        booking_date=str(row["booking_date"]),
        booking_time=str(row["booking_time"]),
        message=str(row["message"]) if row["message"] else None,
        status=str(row["status"]),
        admin_note=str(row["admin_note"]) if row["admin_note"] else None,
        notified_via=str(row["notified_via"]) if row["notified_via"] else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _slot_from_row(row: sqlite3.Row) -> BookingSlotOut:
    return BookingSlotOut(
        id=str(row["id"]),
        slot_date=str(row["slot_date"]),
        slot_time=str(row["slot_time"]),
        is_booked=bool(row["is_booked"]),
        is_blocked=bool(row["is_blocked"]),
        created_at=str(row["created_at"]),
    )


def _notification_from_row(row: sqlite3.Row) -> NotificationOut:
    return NotificationOut(
        id=str(row["id"]),
        booking_id=str(row["booking_id"]) if row["booking_id"] else None,
        recipient_email=str(row["recipient_email"]) if row["recipient_email"] else None,
        recipient_phone=str(row["recipient_phone"]) if row["recipient_phone"] else None,
        channel=str(row["channel"]),
        message_body=str(row["message_body"]) if row["message_body"] else None,
        status=str(row["status"]),
        sent_at=str(row["sent_at"]) if row["sent_at"] else None,
        created_at=str(row["created_at"]),
    )


def _document_suggestion_from_row(row: sqlite3.Row) -> DocumentSuggestionOut:
    return DocumentSuggestionOut(
        id=str(row["id"]),
        document_id=int(row["document_id"]),
        document_filename=str(row["document_filename"]) if row["document_filename"] is not None else None,
        suggested_by=str(row["suggested_by"]),
        suggested_by_name=str(row["suggested_by_name"]) if row["suggested_by_name"] is not None else None,
        suggested_by_email=str(row["suggested_by_email"]) if row["suggested_by_email"] is not None else None,
        suggestion_text=str(row["suggestion_text"]),
        new_content=str(row["new_content"]),
        status=str(row["status"]),
        reviewed_by=str(row["reviewed_by"]) if row["reviewed_by"] is not None else None,
        reviewed_by_email=str(row["reviewed_by_email"]) if row["reviewed_by_email"] is not None else None,
        admin_note=str(row["admin_note"]) if row["admin_note"] is not None else None,
        created_at=str(row["created_at"]),
        reviewed_at=str(row["reviewed_at"]) if row["reviewed_at"] is not None else None,
    )


def _organisation_suggestion_from_row(row: sqlite3.Row) -> OrganisationSuggestionOut:
    return OrganisationSuggestionOut(
        id=str(row["id"]),
        suggested_by=str(row["suggested_by"]),
        suggested_by_name=str(row["suggested_by_name"]) if row["suggested_by_name"] is not None else None,
        suggested_by_email=str(row["suggested_by_email"]) if row["suggested_by_email"] is not None else None,
        title=str(row["title"]),
        description=str(row["description"]),
        category=str(row["category"]),
        status=str(row["status"]),
        reviewed_by=str(row["reviewed_by"]) if row["reviewed_by"] is not None else None,
        reviewed_by_email=str(row["reviewed_by_email"]) if row["reviewed_by_email"] is not None else None,
        admin_note=str(row["admin_note"]) if row["admin_note"] is not None else None,
        created_at=str(row["created_at"]),
        reviewed_at=str(row["reviewed_at"]) if row["reviewed_at"] is not None else None,
    )


def _get_document_override_path(document_id: int, filename: str) -> Path:
    override_dir = get_storage_dir().parent / "document_updates"
    override_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem.strip() or f"document-{document_id}"
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in stem).strip("-_")
    safe_stem = safe_stem or f"document-{document_id}"
    return override_dir / f"{document_id}_{safe_stem}.txt"


def _apply_document_suggestion(document_id: int, filename: str, new_content: str) -> int:
    override_path = _get_document_override_path(document_id, filename)
    override_path.write_text(new_content, encoding="utf-8")

    pipeline = build_pipeline(require_llm=False)
    pipeline.vector_store.delete_source(filename)
    return pipeline.ingest_file(override_path, source_name=filename)


def _ensure_chat_session(
    session_token: str,
    *,
    customer_id: str | None = None,
    admin_id: str | None = None,
) -> sqlite3.Row:
    token = session_token.strip()
    if not token:
        token = uuid.uuid4().hex

    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, customer_id, session_token, admin_id, created_at
            FROM chat_sessions
            WHERE session_token = ?
            LIMIT 1
            """,
            (token,),
        ).fetchone()

        if row is None:
            created_at = _utc_now()
            session_id = _new_id()
            conn.execute(
                """
                INSERT INTO chat_sessions (id, customer_id, session_token, admin_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, customer_id, token, admin_id, created_at),
            )
            row = conn.execute(
                """
                SELECT id, customer_id, session_token, admin_id, created_at
                FROM chat_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        else:
            if customer_id and row["customer_id"] is None:
                conn.execute(
                    "UPDATE chat_sessions SET customer_id = ? WHERE id = ?",
                    (customer_id, row["id"]),
                )
            if admin_id and row["admin_id"] is None:
                conn.execute(
                    "UPDATE chat_sessions SET admin_id = ? WHERE id = ?",
                    (admin_id, row["id"]),
                )
            row = conn.execute(
                """
                SELECT id, customer_id, session_token, admin_id, created_at
                FROM chat_sessions
                WHERE id = ?
                """,
                (row["id"],),
            ).fetchone()

    if row is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create chat session")
    return row


def _resolve_chat_actor(current_user: AuthUser | None, session_token: str | None) -> dict[str, str | None]:
    token = (session_token or "").strip()

    if current_user is not None:
        role = str(current_user["role"])
        user_id = str(current_user["id"])
        email = str(current_user.get("email", "")).strip() or None

        if role in {"admin", "staff", "superadmin"}:
            token = token or f"{role}-{user_id}"
            session_row = _ensure_chat_session(token, admin_id=user_id)
            return {
                "session_id": str(session_row["id"]),
                "session_token": str(session_row["session_token"]),
                "customer_id": str(session_row["customer_id"]) if session_row["customer_id"] else None,
                "user_id": user_id,
                "user_email": email,
                "role": role,
            }

        if role == "customer":
            token = token or f"customer-{user_id}"
            session_row = _ensure_chat_session(token, customer_id=user_id)
            return {
                "session_id": str(session_row["id"]),
                "session_token": str(session_row["session_token"]),
                "customer_id": user_id,
                "user_id": user_id,
                "user_email": email,
                "role": role,
            }

    if not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing session token for guest chat. Send X-Session-Token header.",
        )

    session_row = _ensure_chat_session(token)
    return {
        "session_id": str(session_row["id"]),
        "session_token": str(session_row["session_token"]),
        "customer_id": str(session_row["customer_id"]) if session_row["customer_id"] else None,
        "user_id": None,
        "user_email": None,
        "role": "guest",
    }


@app.on_event("startup")
def startup() -> None:
    ensure_backend_dirs()
    init_db()


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(gemini_configured=has_gemini_key())


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(gemini_configured=has_gemini_key())


@app.post("/api/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register_user(payload: RegisterRequest) -> AuthResponse:
    created_at = _utc_now()
    password_hash = hash_password(payload.password)

    if payload.role == "admin":
        with get_conn() as conn:
            admin_exists = conn.execute("SELECT 1 FROM admins WHERE is_active = 1 LIMIT 1").fetchone()
            if admin_exists:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin account already exists. Register as customer.",
                )

            email_exists = conn.execute("SELECT 1 FROM admins WHERE email = ? LIMIT 1", (payload.email,)).fetchone()
            if email_exists:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

            admin_id = _new_id()
            name = payload.name or payload.email.split("@", 1)[0]
            conn.execute(
                """
                INSERT INTO admins (id, name, email, password_hash, role, is_active, portal_role, created_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (admin_id, name, payload.email, password_hash, "superadmin", 1, "admin", None, created_at),
            )
            row = conn.execute(
                "SELECT id, name, email, portal_role, created_at FROM admins WHERE id = ?",
                (admin_id,),
            ).fetchone()
    else:
        with get_conn() as conn:
            email_exists = conn.execute("SELECT 1 FROM customers WHERE email = ? LIMIT 1", (payload.email,)).fetchone()
            if email_exists:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

            customer_id = _new_id()
            conn.execute(
                """
                INSERT INTO customers (id, name, email, phone, password_hash, is_guest, created_at)
                VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (customer_id, payload.name, payload.email, payload.phone, password_hash, created_at),
            )
            row = conn.execute(
                "SELECT id, name, email, 'customer' AS role, created_at FROM customers WHERE id = ?",
                (customer_id,),
            ).fetchone()

    if row is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User creation failed")

    user = _user_public_from_row(row)
    token = create_access_token(user.id, user.role)
    return AuthResponse(access_token=token, user=user)


@app.post("/api/auth/login", response_model=AuthResponse)
def login_user(payload: LoginRequest) -> AuthResponse:
    with get_conn() as conn:
        admin_row = conn.execute(
            """
            SELECT id, name, email, password_hash, role, portal_role, created_at
            FROM admins
            WHERE email = ? AND is_active = 1
            LIMIT 1
            """,
            (payload.email,),
        ).fetchone()
        customer_row = conn.execute(
            """
            SELECT id, name, email, password_hash, created_at
            FROM customers
            WHERE email = ? AND password_hash IS NOT NULL AND is_guest = 0
            LIMIT 1
            """,
            (payload.email,),
        ).fetchone()
        legacy_row = conn.execute(
            """
            SELECT CAST(id AS TEXT) AS id, NULL AS name, email, password_hash, role, created_at
            FROM users
            WHERE email = ?
            LIMIT 1
            """,
            (payload.email,),
        ).fetchone()

    if admin_row is not None and verify_password(payload.password, str(admin_row["password_hash"])):
        portal_role = _normalize_portal_role(admin_row["portal_role"])
        if payload.expected_role is not None and payload.expected_role != portal_role:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Selected role does not match this account")
        user = UserPublic(
            id=str(admin_row["id"]),
            name=str(admin_row["name"]),
            email=str(admin_row["email"]),
            role=portal_role,
            created_at=str(admin_row["created_at"]),
        )
        token = create_access_token(user.id, user.role)
        return AuthResponse(access_token=token, user=user)

    if customer_row is not None and verify_password(payload.password, str(customer_row["password_hash"])):
        if payload.expected_role is not None and payload.expected_role != "customer":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Selected role does not match this account")
        user = UserPublic(
            id=str(customer_row["id"]),
            name=str(customer_row["name"]) if customer_row["name"] is not None else None,
            email=str(customer_row["email"]),
            role="customer",
            created_at=str(customer_row["created_at"]),
        )
        token = create_access_token(user.id, user.role)
        return AuthResponse(access_token=token, user=user)

    if legacy_row is not None and verify_password(payload.password, str(legacy_row["password_hash"])):
        legacy_role = _normalize_portal_role(legacy_row["role"])
        if payload.expected_role is not None and payload.expected_role != legacy_role:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Selected role does not match this account")
        user = UserPublic(
            id=str(legacy_row["id"]),
            name=None,
            email=str(legacy_row["email"]),
            role=legacy_role,
            created_at=str(legacy_row["created_at"]),
        )
        token = create_access_token(user.id, user.role)
        return AuthResponse(access_token=token, user=user)

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")


@app.get("/api/auth/me", response_model=UserPublic)
def auth_me(current_user: AuthUser = Depends(get_current_user)) -> UserPublic:
    return UserPublic(
        id=str(current_user["id"]),
        name=str(current_user["name"]) if current_user.get("name") else None,
        email=str(current_user["email"]),
        role=_normalize_portal_role(current_user["role"]),
        created_at=str(current_user["created_at"]),
    )


@app.get("/api/users", response_model=list[UserPublic])
def list_users(_: AuthUser = Depends(require_admin)) -> list[UserPublic]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, name, email, role, portal_role, created_at
            FROM (
                SELECT id, name, email, role, portal_role, created_at
                FROM admins
                WHERE is_active = 1
                UNION ALL
                SELECT id, name, email, 'customer' AS role, 'customer' AS portal_role, created_at
                FROM customers
                WHERE email IS NOT NULL
            )
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [_user_public_from_row(row) for row in rows]


@app.get("/api/staff", response_model=list[AdminAccountOut])
def list_staff_accounts(_: AuthUser = Depends(require_admin)) -> list[AdminAccountOut]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                a.id,
                a.name,
                a.email,
                a.portal_role,
                a.is_active,
                a.created_by,
                creator.email AS created_by_email,
                a.created_at
            FROM admins a
            LEFT JOIN admins creator ON creator.id = a.created_by
            ORDER BY
                CASE WHEN a.portal_role = 'admin' THEN 0 ELSE 1 END,
                a.created_at DESC
            """
        ).fetchall()
    return [_admin_account_from_row(row) for row in rows]


@app.post("/api/staff", response_model=AdminAccountOut, status_code=status.HTTP_201_CREATED)
def create_staff_account(
    payload: StaffCreateRequest,
    current_admin: AuthUser = Depends(require_admin),
) -> AdminAccountOut:
    created_at = _utc_now()
    staff_id = _new_id()
    password_hash = hash_password(payload.password)

    with get_conn() as conn:
        email_in_admins = conn.execute("SELECT 1 FROM admins WHERE email = ? LIMIT 1", (payload.email,)).fetchone()
        email_in_customers = conn.execute("SELECT 1 FROM customers WHERE email = ? LIMIT 1", (payload.email,)).fetchone()
        if email_in_admins or email_in_customers:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

        conn.execute(
            """
            INSERT INTO admins (id, name, email, password_hash, role, is_active, portal_role, created_by, created_at)
            VALUES (?, ?, ?, ?, 'admin', 1, 'staff', ?, ?)
            """,
            (staff_id, payload.name, payload.email, password_hash, str(current_admin["id"]), created_at),
        )
        row = conn.execute(
            """
            SELECT
                a.id,
                a.name,
                a.email,
                a.portal_role,
                a.is_active,
                a.created_by,
                creator.email AS created_by_email,
                a.created_at
            FROM admins a
            LEFT JOIN admins creator ON creator.id = a.created_by
            WHERE a.id = ?
            LIMIT 1
            """,
            (staff_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Staff account creation failed")
    return _admin_account_from_row(row)


@app.patch("/api/staff/{staff_id}", response_model=AdminAccountOut)
def update_staff_account_status(
    staff_id: str,
    payload: StaffStatusUpdateRequest,
    current_admin: AuthUser = Depends(require_admin),
) -> AdminAccountOut:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT
                a.id,
                a.name,
                a.email,
                a.portal_role,
                a.is_active,
                a.created_by,
                creator.email AS created_by_email,
                a.created_at
            FROM admins a
            LEFT JOIN admins creator ON creator.id = a.created_by
            WHERE a.id = ?
            LIMIT 1
            """,
            (staff_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Staff account not found")
        if _normalize_portal_role(row["portal_role"]) != "staff":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only staff accounts can be updated here")
        if staff_id == str(current_admin["id"]) and not payload.is_active:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot deactivate your own account")

        conn.execute("UPDATE admins SET is_active = ? WHERE id = ?", (1 if payload.is_active else 0, staff_id))
        updated = conn.execute(
            """
            SELECT
                a.id,
                a.name,
                a.email,
                a.portal_role,
                a.is_active,
                a.created_by,
                creator.email AS created_by_email,
                a.created_at
            FROM admins a
            LEFT JOIN admins creator ON creator.id = a.created_by
            WHERE a.id = ?
            LIMIT 1
            """,
            (staff_id,),
        ).fetchone()

    if updated is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Staff account update failed")
    return _admin_account_from_row(updated)


@app.get("/api/document-suggestions", response_model=list[DocumentSuggestionOut])
def list_document_suggestions(current_user: AuthUser = Depends(require_portal_user)) -> list[DocumentSuggestionOut]:
    current_role = _normalize_portal_role(current_user["role"])
    query = """
        SELECT
            ds.id,
            ds.document_id,
            d.filename AS document_filename,
            ds.suggested_by,
            sugg.name AS suggested_by_name,
            sugg.email AS suggested_by_email,
            ds.suggestion_text,
            ds.new_content,
            ds.status,
            ds.reviewed_by,
            reviewer.email AS reviewed_by_email,
            ds.admin_note,
            ds.created_at,
            ds.reviewed_at
        FROM document_suggestions ds
        LEFT JOIN documents d ON d.id = ds.document_id
        LEFT JOIN admins sugg ON sugg.id = ds.suggested_by
        LEFT JOIN admins reviewer ON reviewer.id = ds.reviewed_by
    """
    params: tuple[str, ...] = ()
    if current_role == "staff":
        query += " WHERE ds.suggested_by = ?"
        params = (str(current_user["id"]),)
    query += " ORDER BY CASE WHEN ds.status = 'pending' THEN 0 ELSE 1 END, ds.created_at DESC"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_document_suggestion_from_row(row) for row in rows]


@app.post("/api/document-suggestions", response_model=DocumentSuggestionOut, status_code=status.HTTP_201_CREATED)
def create_document_suggestion(
    payload: DocumentSuggestionCreateRequest,
    current_user: AuthUser = Depends(require_portal_user),
) -> DocumentSuggestionOut:
    suggestion_id = _new_id()
    created_at = _utc_now()
    actor_id = str(current_user["id"])

    with get_conn() as conn:
        document_row = conn.execute(
            """
            SELECT id, filename
            FROM documents
            WHERE id = ? AND COALESCE(is_active, 1) = 1
            LIMIT 1
            """,
            (payload.document_id,),
        ).fetchone()
        if document_row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

        conn.execute(
            """
            INSERT INTO document_suggestions (
                id, document_id, suggested_by, suggestion_text, new_content,
                status, reviewed_by, admin_note, created_at, reviewed_at
            )
            VALUES (?, ?, ?, ?, ?, 'pending', NULL, NULL, ?, NULL)
            """,
            (suggestion_id, payload.document_id, actor_id, payload.suggestion_text, payload.new_content, created_at),
        )
        row = conn.execute(
            """
            SELECT
                ds.id,
                ds.document_id,
                d.filename AS document_filename,
                ds.suggested_by,
                sugg.name AS suggested_by_name,
                sugg.email AS suggested_by_email,
                ds.suggestion_text,
                ds.new_content,
                ds.status,
                ds.reviewed_by,
                reviewer.email AS reviewed_by_email,
                ds.admin_note,
                ds.created_at,
                ds.reviewed_at
            FROM document_suggestions ds
            LEFT JOIN documents d ON d.id = ds.document_id
            LEFT JOIN admins sugg ON sugg.id = ds.suggested_by
            LEFT JOIN admins reviewer ON reviewer.id = ds.reviewed_by
            WHERE ds.id = ?
            LIMIT 1
            """,
            (suggestion_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document suggestion creation failed")
    return _document_suggestion_from_row(row)


@app.patch("/api/document-suggestions/{suggestion_id}/review", response_model=DocumentSuggestionOut)
def review_document_suggestion(
    suggestion_id: str,
    payload: DocumentSuggestionReviewRequest,
    current_admin: AuthUser = Depends(require_admin),
) -> DocumentSuggestionOut:
    reviewed_at = _utc_now()
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT
                ds.id,
                ds.document_id,
                d.filename AS document_filename,
                ds.suggested_by,
                sugg.name AS suggested_by_name,
                sugg.email AS suggested_by_email,
                ds.suggestion_text,
                ds.new_content,
                ds.status,
                ds.reviewed_by,
                reviewer.email AS reviewed_by_email,
                ds.admin_note,
                ds.created_at,
                ds.reviewed_at
            FROM document_suggestions ds
            LEFT JOIN documents d ON d.id = ds.document_id
            LEFT JOIN admins sugg ON sugg.id = ds.suggested_by
            LEFT JOIN admins reviewer ON reviewer.id = ds.reviewed_by
            WHERE ds.id = ?
            LIMIT 1
            """,
            (suggestion_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document suggestion not found")
        if str(row["status"]) != "pending":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Suggestion has already been reviewed")
        if row["document_filename"] is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Linked document not found")

        chunks_indexed: int | None = None
        if payload.status == "approved":
            chunks_indexed = _apply_document_suggestion(
                int(row["document_id"]),
                str(row["document_filename"]),
                str(row["new_content"]),
            )
            conn.execute(
                """
                UPDATE documents
                SET chunks_indexed = ?, chunk_count = ?
                WHERE id = ?
                """,
                (chunks_indexed, chunks_indexed, int(row["document_id"])),
            )

        conn.execute(
            """
            UPDATE document_suggestions
            SET status = ?, reviewed_by = ?, admin_note = ?, reviewed_at = ?
            WHERE id = ?
            """,
            (payload.status, str(current_admin["id"]), payload.admin_note, reviewed_at, suggestion_id),
        )
        updated = conn.execute(
            """
            SELECT
                ds.id,
                ds.document_id,
                d.filename AS document_filename,
                ds.suggested_by,
                sugg.name AS suggested_by_name,
                sugg.email AS suggested_by_email,
                ds.suggestion_text,
                ds.new_content,
                ds.status,
                ds.reviewed_by,
                reviewer.email AS reviewed_by_email,
                ds.admin_note,
                ds.created_at,
                ds.reviewed_at
            FROM document_suggestions ds
            LEFT JOIN documents d ON d.id = ds.document_id
            LEFT JOIN admins sugg ON sugg.id = ds.suggested_by
            LEFT JOIN admins reviewer ON reviewer.id = ds.reviewed_by
            WHERE ds.id = ?
            LIMIT 1
            """,
            (suggestion_id,),
        ).fetchone()

    if updated is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document suggestion review failed")
    return _document_suggestion_from_row(updated)


@app.get("/api/organisation-suggestions", response_model=list[OrganisationSuggestionOut])
def list_organisation_suggestions(current_user: AuthUser = Depends(require_portal_user)) -> list[OrganisationSuggestionOut]:
    current_role = _normalize_portal_role(current_user["role"])
    query = """
        SELECT
            os.id,
            os.suggested_by,
            sugg.name AS suggested_by_name,
            sugg.email AS suggested_by_email,
            os.title,
            os.description,
            os.category,
            os.status,
            os.reviewed_by,
            reviewer.email AS reviewed_by_email,
            os.admin_note,
            os.created_at,
            os.reviewed_at
        FROM organisation_suggestions os
        LEFT JOIN admins sugg ON sugg.id = os.suggested_by
        LEFT JOIN admins reviewer ON reviewer.id = os.reviewed_by
    """
    params: tuple[str, ...] = ()
    if current_role == "staff":
        query += " WHERE os.suggested_by = ?"
        params = (str(current_user["id"]),)
    query += " ORDER BY CASE WHEN os.status = 'pending' THEN 0 ELSE 1 END, os.created_at DESC"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_organisation_suggestion_from_row(row) for row in rows]


@app.post("/api/organisation-suggestions", response_model=OrganisationSuggestionOut, status_code=status.HTTP_201_CREATED)
def create_organisation_suggestion(
    payload: OrganisationSuggestionCreateRequest,
    current_user: AuthUser = Depends(require_portal_user),
) -> OrganisationSuggestionOut:
    suggestion_id = _new_id()
    created_at = _utc_now()

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO organisation_suggestions (
                id, suggested_by, title, description, category,
                status, reviewed_by, admin_note, created_at, reviewed_at
            )
            VALUES (?, ?, ?, ?, ?, 'pending', NULL, NULL, ?, NULL)
            """,
            (suggestion_id, str(current_user["id"]), payload.title, payload.description, payload.category, created_at),
        )
        row = conn.execute(
            """
            SELECT
                os.id,
                os.suggested_by,
                sugg.name AS suggested_by_name,
                sugg.email AS suggested_by_email,
                os.title,
                os.description,
                os.category,
                os.status,
                os.reviewed_by,
                reviewer.email AS reviewed_by_email,
                os.admin_note,
                os.created_at,
                os.reviewed_at
            FROM organisation_suggestions os
            LEFT JOIN admins sugg ON sugg.id = os.suggested_by
            LEFT JOIN admins reviewer ON reviewer.id = os.reviewed_by
            WHERE os.id = ?
            LIMIT 1
            """,
            (suggestion_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Organisation suggestion creation failed")
    return _organisation_suggestion_from_row(row)


@app.patch("/api/organisation-suggestions/{suggestion_id}/review", response_model=OrganisationSuggestionOut)
def review_organisation_suggestion(
    suggestion_id: str,
    payload: OrganisationSuggestionReviewRequest,
    current_admin: AuthUser = Depends(require_admin),
) -> OrganisationSuggestionOut:
    reviewed_at = _utc_now()
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, status
            FROM organisation_suggestions
            WHERE id = ?
            LIMIT 1
            """,
            (suggestion_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organisation suggestion not found")
        if str(row["status"]) != "pending":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Suggestion has already been reviewed")

        conn.execute(
            """
            UPDATE organisation_suggestions
            SET status = ?, reviewed_by = ?, admin_note = ?, reviewed_at = ?
            WHERE id = ?
            """,
            (payload.status, str(current_admin["id"]), payload.admin_note, reviewed_at, suggestion_id),
        )
        updated = conn.execute(
            """
            SELECT
                os.id,
                os.suggested_by,
                sugg.name AS suggested_by_name,
                sugg.email AS suggested_by_email,
                os.title,
                os.description,
                os.category,
                os.status,
                os.reviewed_by,
                reviewer.email AS reviewed_by_email,
                os.admin_note,
                os.created_at,
                os.reviewed_at
            FROM organisation_suggestions os
            LEFT JOIN admins sugg ON sugg.id = os.suggested_by
            LEFT JOIN admins reviewer ON reviewer.id = os.reviewed_by
            WHERE os.id = ?
            LIMIT 1
            """,
            (suggestion_id,),
        ).fetchone()

    if updated is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Organisation suggestion review failed")
    return _organisation_suggestion_from_row(updated)


@app.post("/api/documents/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    current_admin: AuthUser = Depends(require_admin),
) -> DocumentOut:
    original_name = _sanitize_filename(file.filename or "upload.pdf")
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only .pdf and .txt files are supported")

    storage_dir = get_storage_dir()
    storage_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}_{original_name}"
    stored_path = storage_dir / stored_name

    try:
        with stored_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save file: {exc}") from exc

    try:
        pipeline = build_pipeline(require_llm=False)
        chunks_indexed = pipeline.ingest_file(stored_path, source_name=original_name)
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ingestion failed: {exc}") from exc

    if chunks_indexed <= 0:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No extractable text was found in the uploaded file. "
                "This PDF is likely scanned or image-only. Upload a text-based PDF/TXT file or enable OCR first."
            ),
        )

    uploaded_at = _utc_now()
    file_size = stored_path.stat().st_size if stored_path.exists() else None
    with get_conn() as conn:
        uploaded_by = _resolve_document_uploader_id(conn, current_admin)
        cursor = conn.execute(
            """
            INSERT INTO documents (
                filename, stored_path, uploaded_by, uploaded_at, chunks_indexed,
                file_path, file_size, chunk_count, is_active, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                original_name,
                str(stored_path),
                uploaded_by,
                uploaded_at,
                chunks_indexed,
                str(stored_path),
                file_size,
                chunks_indexed,
                uploaded_at,
            ),
        )
        document_id = int(cursor.lastrowid)
        row = conn.execute(
            """
            SELECT
                d.id, d.filename, d.stored_path, d.uploaded_by, d.uploaded_at, d.chunks_indexed,
                COALESCE(a.email, u.email) AS uploaded_by_email
            FROM documents d
            LEFT JOIN admins a ON a.id = d.uploaded_by
            LEFT JOIN users u ON CAST(u.id AS TEXT) = CAST(d.uploaded_by AS TEXT)
            WHERE d.id = ?
            """,
            (document_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document insert failed")
    return _document_from_row(row)


@app.get("/api/documents", response_model=list[DocumentOut])
def list_documents(_: AuthUser = Depends(require_portal_user)) -> list[DocumentOut]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                d.id, d.filename, d.stored_path, d.uploaded_by, d.uploaded_at, d.chunks_indexed,
                COALESCE(a.email, u.email) AS uploaded_by_email
            FROM documents d
            LEFT JOIN admins a ON a.id = d.uploaded_by
            LEFT JOIN users u ON CAST(u.id AS TEXT) = CAST(d.uploaded_by AS TEXT)
            WHERE COALESCE(d.is_active, 1) = 1
            ORDER BY d.id DESC
            """
        ).fetchall()
    return [_document_from_row(row) for row in rows]


@app.get("/api/documents/{document_id}/download")
def download_document(document_id: int, _: AuthUser = Depends(require_portal_user)) -> FileResponse:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT filename, COALESCE(stored_path, file_path) AS stored_path FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    stored_path = Path(str(row["stored_path"]))
    if not stored_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stored file is missing")

    return FileResponse(
        path=stored_path,
        filename=str(row["filename"]),
        media_type="application/octet-stream",
    )


@app.post("/api/chat/session", response_model=ChatSessionOut, status_code=status.HTTP_201_CREATED)
def create_or_get_chat_session(
    current_user: AuthUser | None = Depends(get_optional_user),
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
) -> ChatSessionOut:
    token = (x_session_token or "").strip() or uuid.uuid4().hex
    customer_id: str | None = None
    admin_id: str | None = None
    if current_user is not None:
        role = str(current_user["role"])
        if role == "customer":
            customer_id = str(current_user["id"])
        elif role in {"admin", "staff", "superadmin"}:
            admin_id = str(current_user["id"])
    row = _ensure_chat_session(token, customer_id=customer_id, admin_id=admin_id)
    return ChatSessionOut(
        id=str(row["id"]),
        customer_id=str(row["customer_id"]) if row["customer_id"] else None,
        session_token=str(row["session_token"]),
        created_at=str(row["created_at"]),
    )


@app.post("/api/chat/ask", response_model=AskResponse)
def ask_question(
    payload: AskRequest,
    current_user: AuthUser | None = Depends(get_optional_user),
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
) -> AskResponse:
    actor = _resolve_chat_actor(current_user, x_session_token)

    pipeline = build_pipeline(require_llm=True)
    if pipeline.vector_store.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No indexed documents found. Ask an admin to upload and ingest PDFs first.",
        )
    if pipeline.answer_generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API key is not configured on the backend.",
        )

    try:
        answer, contexts = pipeline.answer_question(payload.question, top_k=payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Answer generation failed: {exc}") from exc

    normalized_sources = _normalize_sources(contexts)
    sources_json = json.dumps([item.model_dump() for item in normalized_sources])

    with get_conn() as conn:
        user_id_for_insert = _resolve_chat_legacy_user_id(conn, actor)
        conn.execute(
            """
            INSERT INTO chat_messages (
                user_id, question, answer, sources_json, sources, session_id, customer_id, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id_for_insert,
                payload.question,
                answer,
                sources_json,
                sources_json,
                actor["session_id"],
                actor["customer_id"],
                _utc_now(),
            ),
        )

    return AskResponse(answer=answer, sources=normalized_sources)


@app.get("/api/chat/history", response_model=list[ChatMessageOut])
def get_chat_history(
    current_user: AuthUser | None = Depends(get_optional_user),
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
) -> list[ChatMessageOut]:
    base_query = """
        SELECT
            m.id,
            m.session_id,
            m.customer_id,
            CAST(m.user_id AS TEXT) AS user_id,
            COALESCE(c.email, a.email, u.email, '') AS user_email,
            m.question,
            m.answer,
            m.sources_json,
            m.sources,
            m.created_at
        FROM chat_messages m
        LEFT JOIN chat_sessions s ON s.id = m.session_id
        LEFT JOIN customers c ON c.id = COALESCE(m.customer_id, s.customer_id)
        LEFT JOIN admins a ON a.id = s.admin_id
        LEFT JOIN users u ON CAST(u.id AS TEXT) = CAST(m.user_id AS TEXT)
    """

    with get_conn() as conn:
        if current_user is not None and str(current_user["role"]) == "admin":
            rows = conn.execute(f"{base_query} ORDER BY m.id DESC LIMIT 500").fetchall()
            return [_chat_from_row(row) for row in rows]

        if current_user is not None and str(current_user["role"]) == "staff":
            rows = conn.execute(
                f"{base_query} WHERE s.admin_id = ? ORDER BY m.id DESC LIMIT 500",
                (str(current_user["id"]),),
            ).fetchall()
            return [_chat_from_row(row) for row in rows]

        if current_user is not None and str(current_user["role"]) == "customer":
            customer_id = str(current_user["id"])
            if x_session_token:
                token = x_session_token.strip()
                rows = conn.execute(
                    f"{base_query} WHERE s.session_token = ? OR m.customer_id = ? ORDER BY m.id DESC LIMIT 500",
                    (token, customer_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"{base_query} WHERE m.customer_id = ? ORDER BY m.id DESC LIMIT 500",
                    (customer_id,),
                ).fetchall()
            return [_chat_from_row(row) for row in rows]

        token = (x_session_token or "").strip()
        if not token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing session token for guest history. Send X-Session-Token header.",
            )

        rows = conn.execute(
            f"{base_query} WHERE s.session_token = ? ORDER BY m.id DESC LIMIT 500",
            (token,),
        ).fetchall()
        return [_chat_from_row(row) for row in rows]


@app.post("/api/bookings", response_model=BookingOut, status_code=status.HTTP_201_CREATED)
def create_booking(
    payload: BookingCreateRequest,
    current_user: AuthUser | None = Depends(get_optional_user),
) -> BookingOut:
    customer_id: str | None = None
    if current_user is not None and str(current_user["role"]) == "customer":
        customer_id = str(current_user["id"])

    guest_name = payload.guest_name
    guest_email = payload.guest_email
    guest_phone = payload.guest_phone
    if customer_id is None and (not guest_name or (not guest_email and not guest_phone)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Guest booking requires guest_name and at least guest_email or guest_phone.",
        )

    booking_date, booking_time = _validate_workday_slot(payload.booking_date, payload.booking_time)
    now = _utc_now()
    booking_id = _new_id()
    service_type = payload.service_type or "consultation"

    with get_conn() as conn:
        if customer_id is not None:
            customer_row = conn.execute(
                """
                SELECT name, email, phone
                FROM customers
                WHERE id = ?
                LIMIT 1
                """,
                (customer_id,),
            ).fetchone()
            if customer_row is not None:
                if guest_name is None and customer_row["name"] is not None:
                    guest_name = str(customer_row["name"])
                if guest_email is None and customer_row["email"] is not None:
                    guest_email = str(customer_row["email"])
                if guest_phone is None and customer_row["phone"] is not None:
                    guest_phone = str(customer_row["phone"])

        _assert_no_active_booking_conflict(conn, booking_date, booking_time)
        _reserve_slot(conn, booking_date, booking_time)

        conn.execute(
            """
            INSERT INTO bookings (
                id, customer_id, guest_name, guest_email, guest_phone,
                service_type, booking_date, booking_time, message,
                status, admin_note, notified_via, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, NULL, ?, ?)
            """,
            (
                booking_id,
                customer_id,
                guest_name,
                guest_email,
                guest_phone,
                service_type,
                booking_date,
                booking_time,
                payload.message,
                now,
                now,
            ),
        )

        row = conn.execute(
            """
            SELECT
                id, customer_id, guest_name, guest_email, guest_phone, service_type,
                booking_date, booking_time, message, status, admin_note, notified_via,
                created_at, updated_at
            FROM bookings
            WHERE id = ?
            """,
            (booking_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Booking creation failed")
    return _booking_from_row(row)


@app.get("/api/bookings", response_model=list[BookingOut])
def list_bookings(_: AuthUser = Depends(require_portal_user)) -> list[BookingOut]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                id, customer_id, guest_name, guest_email, guest_phone, service_type,
                booking_date, booking_time, message, status, admin_note, notified_via,
                created_at, updated_at
            FROM bookings
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [_booking_from_row(row) for row in rows]


@app.get("/api/bookings/me", response_model=list[BookingOut])
def list_my_bookings(current_user: AuthUser = Depends(get_current_user)) -> list[BookingOut]:
    if str(current_user["role"]) != "customer":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Customer access required")

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                id, customer_id, guest_name, guest_email, guest_phone, service_type,
                booking_date, booking_time, message, status, admin_note, notified_via,
                created_at, updated_at
            FROM bookings
            WHERE customer_id = ?
            ORDER BY created_at DESC
            """,
            (str(current_user["id"]),),
        ).fetchall()
    return [_booking_from_row(row) for row in rows]


@app.patch("/api/bookings/{booking_id}/me", response_model=BookingOut)
def update_my_booking(
    booking_id: str,
    payload: BookingCustomerUpdateRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> BookingOut:
    if str(current_user["role"]) != "customer":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Customer access required")

    if payload.status is None and payload.booking_date is None and payload.booking_time is None and payload.message is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No updates provided")

    if (payload.booking_date is None) != (payload.booking_time is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide both booking_date and booking_time when rescheduling.",
        )

    updated_at = _utc_now()
    customer_id = str(current_user["id"])

    with get_conn() as conn:
        current_row = conn.execute(
            """
            SELECT
                id, customer_id, guest_name, guest_email, guest_phone, service_type,
                booking_date, booking_time, message, status, admin_note, notified_via,
                created_at, updated_at
            FROM bookings
            WHERE id = ? AND customer_id = ?
            LIMIT 1
            """,
            (booking_id, customer_id),
        ).fetchone()
        if current_row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Booking not found")

        current_status = str(current_row["status"])
        if current_status == "cancelled":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Cancelled bookings cannot be modified")

        next_status = current_status
        if payload.status is not None:
            if payload.status != "cancelled":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Customers can only change status to cancelled.",
                )
            next_status = "cancelled"

        next_date = str(current_row["booking_date"])
        next_time = str(current_row["booking_time"])
        if payload.booking_date is not None and payload.booking_time is not None:
            next_date, next_time = _validate_workday_slot(payload.booking_date, payload.booking_time)

        moving_slot = (
            payload.booking_date is not None
            and payload.booking_time is not None
            and (next_date != str(current_row["booking_date"]) or next_time != str(current_row["booking_time"]))
        )
        old_date = str(current_row["booking_date"])
        old_time = str(current_row["booking_time"])

        if next_status != "cancelled":
            _assert_no_active_booking_conflict(conn, next_date, next_time, exclude_booking_id=booking_id)
            if moving_slot:
                _reserve_slot(conn, next_date, next_time)
            else:
                _mark_slot_booked(conn, next_date, next_time)

        next_message = str(current_row["message"]) if current_row["message"] is not None else None
        if payload.message is not None:
            next_message = payload.message

        conn.execute(
            """
            UPDATE bookings
            SET booking_date = ?, booking_time = ?, message = ?, status = ?, updated_at = ?
            WHERE id = ? AND customer_id = ?
            """,
            (next_date, next_time, next_message, next_status, updated_at, booking_id, customer_id),
        )

        if next_status == "cancelled":
            _release_slot(conn, old_date, old_time)
        elif moving_slot:
            _release_slot(conn, old_date, old_time)

        updated = conn.execute(
            """
            SELECT
                id, customer_id, guest_name, guest_email, guest_phone, service_type,
                booking_date, booking_time, message, status, admin_note, notified_via,
                created_at, updated_at
            FROM bookings
            WHERE id = ? AND customer_id = ?
            """,
            (booking_id, customer_id),
        ).fetchone()

    if updated is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Booking update failed")
    return _booking_from_row(updated)


@app.patch("/api/bookings/{booking_id}/status", response_model=BookingOut)
def update_booking_status(
    booking_id: str,
    payload: BookingStatusUpdateRequest,
    _: AuthUser = Depends(require_admin),
) -> BookingOut:
    updated_at = _utc_now()
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT
                b.id, b.customer_id, b.guest_name, b.guest_email, b.guest_phone, b.service_type,
                b.booking_date, b.booking_time, b.message, b.status, b.admin_note, b.notified_via,
                b.created_at, b.updated_at, c.email AS customer_email, c.phone AS customer_phone
            FROM bookings b
            LEFT JOIN customers c ON c.id = b.customer_id
            WHERE b.id = ?
            LIMIT 1
            """,
            (booking_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Booking not found")

        slot_date = str(row["booking_date"])
        slot_time = str(row["booking_time"])
        current_status = str(row["status"])

        if payload.status == "cancelled":
            if current_status != "cancelled":
                _release_slot(conn, slot_date, slot_time)
        else:
            _assert_no_active_booking_conflict(conn, slot_date, slot_time, exclude_booking_id=booking_id)
            if current_status == "cancelled":
                _reserve_slot(conn, slot_date, slot_time)
            else:
                _mark_slot_booked(conn, slot_date, slot_time)

        conn.execute(
            """
            UPDATE bookings
            SET status = ?, admin_note = ?, notified_via = ?, updated_at = ?
            WHERE id = ?
            """,
            (payload.status, payload.admin_note, payload.notified_via, updated_at, booking_id),
        )

        recipient_email = str(row["guest_email"]) if row["guest_email"] else None
        recipient_phone = str(row["guest_phone"]) if row["guest_phone"] else None
        if recipient_email is None and row["customer_email"]:
            recipient_email = str(row["customer_email"])
        if recipient_phone is None and row["customer_phone"]:
            recipient_phone = str(row["customer_phone"])

        message_body = f"Booking {booking_id} is now {payload.status}."
        if payload.notified_via in {"email", "both"} and recipient_email:
            conn.execute(
                """
                INSERT INTO notifications (
                    id, booking_id, recipient_email, recipient_phone, channel, message_body, status, sent_at, created_at
                )
                VALUES (?, ?, ?, ?, 'email', ?, 'pending', NULL, ?)
                """,
                (_new_id(), booking_id, recipient_email, recipient_phone, message_body, updated_at),
            )
        if payload.notified_via in {"whatsapp", "both"} and recipient_phone:
            conn.execute(
                """
                INSERT INTO notifications (
                    id, booking_id, recipient_email, recipient_phone, channel, message_body, status, sent_at, created_at
                )
                VALUES (?, ?, ?, ?, 'whatsapp', ?, 'pending', NULL, ?)
                """,
                (_new_id(), booking_id, recipient_email, recipient_phone, message_body, updated_at),
            )

        updated = conn.execute(
            """
            SELECT
                id, customer_id, guest_name, guest_email, guest_phone, service_type,
                booking_date, booking_time, message, status, admin_note, notified_via,
                created_at, updated_at
            FROM bookings
            WHERE id = ?
            """,
            (booking_id,),
        ).fetchone()

    if updated is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Booking update failed")
    return _booking_from_row(updated)


@app.get("/api/booking-slots", response_model=list[BookingSlotOut])
def list_booking_slots(available_only: bool = Query(default=True)) -> list[BookingSlotOut]:
    with get_conn() as conn:
        if available_only:
            rows = conn.execute(
                """
                SELECT id, slot_date, slot_time, is_booked, is_blocked, created_at
                FROM booking_slots
                WHERE is_booked = 0 AND is_blocked = 0
                ORDER BY slot_date ASC, slot_time ASC
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, slot_date, slot_time, is_booked, is_blocked, created_at
                FROM booking_slots
                ORDER BY slot_date ASC, slot_time ASC
                """
            ).fetchall()
    return [_slot_from_row(row) for row in rows]


@app.post("/api/booking-slots", response_model=BookingSlotOut, status_code=status.HTTP_201_CREATED)
def create_booking_slot(
    payload: BookingSlotCreateRequest,
    _: AuthUser = Depends(require_admin),
) -> BookingSlotOut:
    slot_date, slot_time = _validate_workday_slot(payload.slot_date, payload.slot_time)
    slot_id = _new_id()
    created_at = _utc_now()
    try:
        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO booking_slots (id, slot_date, slot_time, is_booked, is_blocked, created_at)
                VALUES (?, ?, ?, 0, 0, ?)
                """,
                (slot_id, slot_date, slot_time, created_at),
            )
            row = conn.execute(
                """
                SELECT id, slot_date, slot_time, is_booked, is_blocked, created_at
                FROM booking_slots
                WHERE id = ?
                """,
                (slot_id,),
            ).fetchone()
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Slot already exists") from exc

    if row is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Slot creation failed")
    return _slot_from_row(row)


@app.patch("/api/booking-slots/{slot_id}", response_model=BookingSlotOut)
def update_booking_slot(
    slot_id: str,
    payload: BookingSlotUpdateRequest,
    _: AuthUser = Depends(require_admin),
) -> BookingSlotOut:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, slot_date, slot_time, is_booked, is_blocked, created_at
            FROM booking_slots
            WHERE id = ?
            LIMIT 1
            """,
            (slot_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Slot not found")

        is_booked = bool(row["is_booked"]) if payload.is_booked is None else bool(payload.is_booked)
        is_blocked = bool(row["is_blocked"]) if payload.is_blocked is None else bool(payload.is_blocked)
        conn.execute(
            "UPDATE booking_slots SET is_booked = ?, is_blocked = ? WHERE id = ?",
            (1 if is_booked else 0, 1 if is_blocked else 0, slot_id),
        )
        updated = conn.execute(
            """
            SELECT id, slot_date, slot_time, is_booked, is_blocked, created_at
            FROM booking_slots
            WHERE id = ?
            """,
            (slot_id,),
        ).fetchone()

    if updated is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Slot update failed")
    return _slot_from_row(updated)


@app.get("/api/notifications", response_model=list[NotificationOut])
def list_notifications(_: AuthUser = Depends(require_admin)) -> list[NotificationOut]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                id, booking_id, recipient_email, recipient_phone,
                channel, message_body, status, sent_at, created_at
            FROM notifications
            ORDER BY created_at DESC
            LIMIT 500
            """
        ).fetchall()
    return [_notification_from_row(row) for row in rows]


@app.get("/api/rag/status")
def rag_status(current_user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    runtime = get_runtime()
    return {
        "user_id": str(current_user["id"]),
        "vector_backend": runtime.vector_store.backend_name,
        "vector_count": runtime.vector_store.count(),
        "collection": runtime.settings.collection_name,
        "chroma_dir": str(runtime.settings.chroma_dir),
    }
