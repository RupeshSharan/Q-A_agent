"""Authentication dependencies for FastAPI routes."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .db import get_conn
from .security import decode_access_token

AuthUser = Dict[str, Any]
http_bearer = HTTPBearer(auto_error=False)


def _load_admin(admin_id: str) -> AuthUser | None:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, name, email, role, portal_role, created_by, created_at
            FROM admins
            WHERE id = ? AND is_active = 1
            """,
            (admin_id,),
        ).fetchone()
    if row is None:
        return None
    raw_portal_role = str(row["portal_role"]).strip().lower() if row["portal_role"] is not None else ""
    effective_role = raw_portal_role if raw_portal_role in {"admin", "staff"} else "admin"
    return {
        "id": str(row["id"]),
        "name": str(row["name"]),
        "email": str(row["email"]),
        "role": effective_role,
        "db_role": str(row["role"]),
        "portal_role": effective_role,
        "created_by": str(row["created_by"]) if row["created_by"] is not None else None,
        "created_at": str(row["created_at"]),
    }


def _load_customer(customer_id: str) -> AuthUser | None:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, name, email, created_at
            FROM customers
            WHERE id = ?
            """,
            (customer_id,),
        ).fetchone()
    if row is None:
        return None
    email_value = str(row["email"]) if row["email"] is not None else ""
    return {
        "id": str(row["id"]),
        "name": str(row["name"]) if row["name"] is not None else None,
        "email": email_value,
        "role": "customer",
        "created_at": str(row["created_at"]),
    }


def _load_legacy_user(user_id: str) -> AuthUser | None:
    try:
        numeric_id = int(user_id)
    except ValueError:
        return None

    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, email, role, created_at
            FROM users
            WHERE id = ?
            """,
            (numeric_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": str(row["id"]),
        "name": None,
        "email": str(row["email"]),
        "role": str(row["role"]),
        "created_at": str(row["created_at"]),
    }


def _resolve_user(credentials: HTTPAuthorizationCredentials | None) -> AuthUser | None:
    if credentials is None:
        return None

    try:
        payload = decode_access_token(credentials.credentials)
        user_id = str(payload.get("sub", "")).strip()
        role = str(payload.get("role", "")).strip().lower()
        if not user_id:
            raise ValueError("Missing subject")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        ) from exc

    if role in {"admin", "staff", "superadmin"}:
        user = _load_admin(user_id)
        if user is not None:
            return user
        return _load_legacy_user(user_id)

    if role == "customer":
        user = _load_customer(user_id)
        if user is not None:
            return user
        return _load_legacy_user(user_id)

    return _load_legacy_user(user_id)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(http_bearer),
) -> AuthUser:
    """Return current authenticated user from Bearer token."""
    user = _resolve_user(credentials)
    if user is None:
        detail = "Missing authentication token" if credentials is None else "User not found"
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)
    return user


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(http_bearer),
) -> AuthUser | None:
    """Return current user if a valid bearer token is present, else None."""
    return _resolve_user(credentials)


def require_admin(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    """Ensure authenticated user has admin role."""
    if user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


def require_portal_user(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    """Ensure authenticated user is an admin or staff portal user."""
    if user["role"] not in {"admin", "staff"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or staff access required",
        )
    return user
