export type UserRole = "admin" | "staff" | "customer";

export interface UserPublic {
  id: string;
  name: string | null;
  email: string;
  role: UserRole;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: "bearer";
  user: UserPublic;
}

export interface SourceSnippet {
  text: string;
  source: string;
  chunk_index: number;
  distance: number | null;
}

export interface AskResponse {
  answer: string;
  sources: SourceSnippet[];
}

export interface DocumentRecord {
  id: number;
  filename: string;
  stored_path: string;
  uploaded_by: string | null;
  uploaded_by_email: string | null;
  uploaded_at: string;
  chunks_indexed: number;
}

export interface ChatHistoryRecord {
  id: number;
  session_id: string | null;
  customer_id: string | null;
  user_id: string | null;
  user_email: string | null;
  question: string;
  answer: string;
  sources: SourceSnippet[];
  created_at: string;
}

export interface BookingRecord {
  id: string;
  customer_id: string | null;
  guest_name: string | null;
  guest_email: string | null;
  guest_phone: string | null;
  service_type: string | null;
  booking_date: string;
  booking_time: string;
  message: string | null;
  status: "pending" | "confirmed" | "cancelled";
  admin_note: string | null;
  notified_via: "email" | "whatsapp" | "both" | null;
  created_at: string;
  updated_at: string;
}

export interface BookingSlotRecord {
  id: string;
  slot_date: string;
  slot_time: string;
  is_booked: boolean;
  is_blocked: boolean;
  created_at: string;
}

export interface NotificationRecord {
  id: string;
  booking_id: string | null;
  recipient_email: string | null;
  recipient_phone: string | null;
  channel: "email" | "whatsapp";
  message_body: string | null;
  status: "pending" | "sent" | "failed";
  sent_at: string | null;
  created_at: string;
}

export interface AdminAccountRecord {
  id: string;
  name: string;
  email: string;
  role: "admin" | "staff";
  is_active: boolean;
  created_by: string | null;
  created_by_email: string | null;
  created_at: string;
}

export interface DocumentSuggestionRecord {
  id: string;
  document_id: number;
  document_filename: string | null;
  suggested_by: string;
  suggested_by_name: string | null;
  suggested_by_email: string | null;
  suggestion_text: string;
  new_content: string;
  status: "pending" | "approved" | "rejected";
  reviewed_by: string | null;
  reviewed_by_email: string | null;
  admin_note: string | null;
  created_at: string;
  reviewed_at: string | null;
}

export interface OrganisationSuggestionRecord {
  id: string;
  suggested_by: string;
  suggested_by_name: string | null;
  suggested_by_email: string | null;
  title: string;
  description: string;
  category: "service" | "timing" | "pricing" | "other";
  status: "pending" | "approved" | "rejected";
  reviewed_by: string | null;
  reviewed_by_email: string | null;
  admin_note: string | null;
  created_at: string;
  reviewed_at: string | null;
}

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

function apiCandidates(): string[] {
  const candidates: string[] = [];
  const push = (value: string) => {
    const normalized = value.trim().replace(/\/+$/, "");
    if (!normalized || candidates.includes(normalized)) {
      return;
    }
    candidates.push(normalized);
  };

  if (typeof window !== "undefined" && window.location?.origin) {
    push(window.location.origin);
  }
  push(API_URL);
  push("http://localhost:8000");
  push("http://127.0.0.1:8000");
  if (typeof window !== "undefined" && window.location?.hostname) {
    push(`http://${window.location.hostname}:8000`);
  }
  return candidates;
}

async function request<T>(path: string, init: RequestInit = {}, token?: string): Promise<T> {
  const headers = new Headers(init.headers ?? {});
  if (!(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const candidates = apiCandidates();
  let networkFailure = false;
  let lastApiError: Error | null = null;

  try {
    for (const base of candidates) {
      let response: Response;
      try {
        response = await fetch(`${base}${path}`, { ...init, headers });
      } catch (err) {
        networkFailure = true;
        if (err instanceof TypeError) {
          continue;
        }
        throw err;
      }

      if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try {
          const payload = (await response.json()) as { detail?: string };
          if (payload.detail) {
            detail = payload.detail;
          }
        } catch {
          // Keep fallback error detail.
        }
        const apiError = new Error(detail);
        if (response.status === 404 || response.status === 502 || response.status === 503) {
          lastApiError = apiError;
          continue;
        }
        throw apiError;
      }

      if (response.status === 204) {
        return undefined as T;
      }
      return (await response.json()) as T;
    }
  } catch (err) {
    if (err instanceof Error && err.message !== "Failed to fetch") {
      throw err;
    }
    networkFailure = true;
  }

  if (networkFailure) {
    throw new Error(`Cannot reach backend API on port 8000. Tried: ${candidates.join(", ")}`);
  }
  if (lastApiError) {
    throw lastApiError;
  }
  throw new Error("Request failed");
}

export async function registerAdmin(email: string, password: string, name?: string): Promise<AuthResponse> {
  return request<AuthResponse>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password, name, role: "admin" })
  });
}

export async function login(email: string, password: string, expectedRole: "admin" | "staff"): Promise<AuthResponse> {
  return request<AuthResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password, expected_role: expectedRole })
  });
}

export async function getCurrentUser(token: string): Promise<UserPublic> {
  return request<UserPublic>("/api/auth/me", {}, token);
}

export async function listUsers(token: string): Promise<UserPublic[]> {
  return request<UserPublic[]>("/api/users", {}, token);
}

export async function listStaffAccounts(token: string): Promise<AdminAccountRecord[]> {
  return request<AdminAccountRecord[]>("/api/staff", {}, token);
}

export async function createStaffAccount(
  payload: { name: string; email: string; password: string },
  token: string
): Promise<AdminAccountRecord> {
  return request<AdminAccountRecord>(
    "/api/staff",
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    token
  );
}

export async function updateStaffAccountStatus(
  staffId: string,
  payload: { is_active: boolean },
  token: string
): Promise<AdminAccountRecord> {
  return request<AdminAccountRecord>(
    `/api/staff/${staffId}`,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    token
  );
}

export async function uploadDocument(file: File, token: string): Promise<DocumentRecord> {
  const formData = new FormData();
  formData.append("file", file);
  return request<DocumentRecord>(
    "/api/documents/upload",
    {
      method: "POST",
      body: formData
    },
    token
  );
}

export async function listDocuments(token: string): Promise<DocumentRecord[]> {
  return request<DocumentRecord[]>("/api/documents", {}, token);
}

export async function askQuestion(question: string, token: string): Promise<AskResponse> {
  return request<AskResponse>(
    "/api/chat/ask",
    {
      method: "POST",
      body: JSON.stringify({ question })
    },
    token
  );
}

export async function getHistory(token: string): Promise<ChatHistoryRecord[]> {
  return request<ChatHistoryRecord[]>("/api/chat/history", {}, token);
}

export async function listBookings(token: string): Promise<BookingRecord[]> {
  return request<BookingRecord[]>("/api/bookings", {}, token);
}

export async function updateBookingStatus(
  bookingId: string,
  payload: {
    status: "pending" | "confirmed" | "cancelled";
    admin_note?: string | null;
    notified_via?: "email" | "whatsapp" | "both" | null;
  },
  token: string
): Promise<BookingRecord> {
  return request<BookingRecord>(
    `/api/bookings/${bookingId}/status`,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    token
  );
}

export async function listBookingSlots(token: string, availableOnly = false): Promise<BookingSlotRecord[]> {
  const query = availableOnly ? "?available_only=true" : "?available_only=false";
  return request<BookingSlotRecord[]>(`/api/booking-slots${query}`, {}, token);
}

export async function listNotifications(token: string): Promise<NotificationRecord[]> {
  return request<NotificationRecord[]>("/api/notifications", {}, token);
}

export async function listDocumentSuggestions(token: string): Promise<DocumentSuggestionRecord[]> {
  return request<DocumentSuggestionRecord[]>("/api/document-suggestions", {}, token);
}

export async function createDocumentSuggestion(
  payload: { document_id: number; suggestion_text: string; new_content: string },
  token: string
): Promise<DocumentSuggestionRecord> {
  return request<DocumentSuggestionRecord>(
    "/api/document-suggestions",
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    token
  );
}

export async function reviewDocumentSuggestion(
  suggestionId: string,
  payload: { status: "approved" | "rejected"; admin_note?: string | null },
  token: string
): Promise<DocumentSuggestionRecord> {
  return request<DocumentSuggestionRecord>(
    `/api/document-suggestions/${suggestionId}/review`,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    token
  );
}

export async function listOrganisationSuggestions(token: string): Promise<OrganisationSuggestionRecord[]> {
  return request<OrganisationSuggestionRecord[]>("/api/organisation-suggestions", {}, token);
}

export async function createOrganisationSuggestion(
  payload: { title: string; description: string; category: "service" | "timing" | "pricing" | "other" },
  token: string
): Promise<OrganisationSuggestionRecord> {
  return request<OrganisationSuggestionRecord>(
    "/api/organisation-suggestions",
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    token
  );
}

export async function reviewOrganisationSuggestion(
  suggestionId: string,
  payload: { status: "approved" | "rejected"; admin_note?: string | null },
  token: string
): Promise<OrganisationSuggestionRecord> {
  return request<OrganisationSuggestionRecord>(
    `/api/organisation-suggestions/${suggestionId}/review`,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    token
  );
}
