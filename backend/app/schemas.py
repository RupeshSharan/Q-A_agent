"""Pydantic API schemas for DocMind backend."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class UserPublic(BaseModel):
    id: str
    name: str | None = None
    email: str
    role: Literal["admin", "staff", "customer"]
    created_at: str


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=256)
    role: Literal["admin", "customer"] = "customer"
    name: str | None = Field(default=None, max_length=100)
    phone: str | None = Field(default=None, max_length=20)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
            raise ValueError("Invalid email format")
        return normalized

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("phone")
    @classmethod
    def normalize_phone(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class LoginRequest(BaseModel):
    email: str
    password: str
    expected_role: Literal["admin", "staff", "customer"] | None = None

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
            raise ValueError("Invalid email format")
        return normalized

    @field_validator("expected_role")
    @classmethod
    def normalize_expected_role(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower()
        return cleaned or None


class AuthResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: UserPublic


class SourceSnippet(BaseModel):
    text: str
    source: str
    chunk_index: int
    distance: float | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4_000)
    top_k: int | None = Field(default=None, ge=1, le=25)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Question must not be empty")
        return text


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceSnippet]


class DocumentOut(BaseModel):
    id: int
    filename: str
    stored_path: str
    uploaded_by: str | None = None
    uploaded_by_email: str | None = None
    uploaded_at: str
    chunks_indexed: int


class AdminAccountOut(BaseModel):
    id: str
    name: str
    email: str
    role: Literal["admin", "staff"]
    is_active: bool
    created_by: str | None = None
    created_by_email: str | None = None
    created_at: str


class StaffCreateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: str
    password: str = Field(min_length=8, max_length=256)

    @field_validator("name")
    @classmethod
    def normalize_name_required(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Name must not be empty")
        return cleaned

    @field_validator("email")
    @classmethod
    def normalize_staff_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
            raise ValueError("Invalid email format")
        return normalized


class StaffStatusUpdateRequest(BaseModel):
    is_active: bool


class DocumentSuggestionCreateRequest(BaseModel):
    document_id: int
    suggestion_text: str = Field(min_length=5, max_length=4_000)
    new_content: str = Field(min_length=5, max_length=20_000)

    @field_validator("suggestion_text", "new_content")
    @classmethod
    def normalize_document_suggestion_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Text must not be empty")
        return cleaned


class DocumentSuggestionReviewRequest(BaseModel):
    status: Literal["approved", "rejected"]
    admin_note: str | None = Field(default=None, max_length=2_000)

    @field_validator("admin_note")
    @classmethod
    def normalize_document_review_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class DocumentSuggestionOut(BaseModel):
    id: str
    document_id: int
    document_filename: str | None = None
    suggested_by: str
    suggested_by_name: str | None = None
    suggested_by_email: str | None = None
    suggestion_text: str
    new_content: str
    status: Literal["pending", "approved", "rejected"]
    reviewed_by: str | None = None
    reviewed_by_email: str | None = None
    admin_note: str | None = None
    created_at: str
    reviewed_at: str | None = None


class OrganisationSuggestionCreateRequest(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=5, max_length=5_000)
    category: Literal["service", "timing", "pricing", "other"]

    @field_validator("title", "description")
    @classmethod
    def normalize_org_suggestion_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Text must not be empty")
        return cleaned


class OrganisationSuggestionReviewRequest(BaseModel):
    status: Literal["approved", "rejected"]
    admin_note: str | None = Field(default=None, max_length=2_000)

    @field_validator("admin_note")
    @classmethod
    def normalize_org_review_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class OrganisationSuggestionOut(BaseModel):
    id: str
    suggested_by: str
    suggested_by_name: str | None = None
    suggested_by_email: str | None = None
    title: str
    description: str
    category: Literal["service", "timing", "pricing", "other"]
    status: Literal["pending", "approved", "rejected"]
    reviewed_by: str | None = None
    reviewed_by_email: str | None = None
    admin_note: str | None = None
    created_at: str
    reviewed_at: str | None = None


class ChatSessionOut(BaseModel):
    id: str
    customer_id: str | None = None
    session_token: str
    created_at: str


class ChatMessageOut(BaseModel):
    id: int
    session_id: str | None = None
    customer_id: str | None = None
    user_id: str | None = None
    user_email: str | None = None
    question: str
    answer: str
    sources: list[SourceSnippet]
    created_at: str


class BookingCreateRequest(BaseModel):
    guest_name: str | None = Field(default=None, max_length=100)
    guest_email: str | None = Field(default=None, max_length=255)
    guest_phone: str | None = Field(default=None, max_length=20)
    service_type: str | None = Field(default=None, max_length=100)
    booking_date: str
    booking_time: str
    message: str | None = Field(default=None, max_length=2_000)

    @field_validator("guest_name", "guest_email", "guest_phone", "service_type", "message")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("guest_email")
    @classmethod
    def normalize_guest_email(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
            raise ValueError("Invalid email format")
        return normalized


class BookingStatusUpdateRequest(BaseModel):
    status: Literal["pending", "confirmed", "cancelled"]
    admin_note: str | None = Field(default=None, max_length=2_000)
    notified_via: Literal["email", "whatsapp", "both"] | None = None

    @field_validator("admin_note")
    @classmethod
    def normalize_admin_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class BookingCustomerUpdateRequest(BaseModel):
    status: Literal["cancelled"] | None = None
    booking_date: str | None = None
    booking_time: str | None = None
    message: str | None = Field(default=None, max_length=2_000)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class BookingOut(BaseModel):
    id: str
    customer_id: str | None = None
    guest_name: str | None = None
    guest_email: str | None = None
    guest_phone: str | None = None
    service_type: str | None = None
    booking_date: str
    booking_time: str
    message: str | None = None
    status: Literal["pending", "confirmed", "cancelled"]
    admin_note: str | None = None
    notified_via: Literal["email", "whatsapp", "both"] | None = None
    created_at: str
    updated_at: str


class BookingSlotCreateRequest(BaseModel):
    slot_date: str
    slot_time: str


class BookingSlotUpdateRequest(BaseModel):
    is_booked: bool | None = None
    is_blocked: bool | None = None


class BookingSlotOut(BaseModel):
    id: str
    slot_date: str
    slot_time: str
    is_booked: bool
    is_blocked: bool
    created_at: str


class NotificationOut(BaseModel):
    id: str
    booking_id: str | None = None
    recipient_email: str | None = None
    recipient_phone: str | None = None
    channel: Literal["email", "whatsapp"]
    message_body: str | None = None
    status: Literal["pending", "sent", "failed"]
    sent_at: str | None = None
    created_at: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    gemini_configured: bool
