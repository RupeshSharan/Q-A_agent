export type UserRole = "customer" | "admin" | "superadmin";

export interface UserPublic {
  id: string;
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

export interface ChatSessionResponse {
  id: string;
  customer_id: string | null;
  session_token: string;
  created_at: string;
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

export interface BookingSlotRecord {
  id: string;
  slot_date: string;
  slot_time: string;
  is_booked: boolean;
  is_blocked: boolean;
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

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

function apiCandidates(): string[] {
  const candidates: string[] = [];
  const push = (value: string) => {
    const normalized = value.trim().replace(/\/+$/, "");
    if (!normalized) {
      return;
    }
    if (!candidates.includes(normalized)) {
      candidates.push(normalized);
    }
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

async function request<T>(
  path: string,
  init: RequestInit = {},
  options: { sessionToken?: string; authToken?: string } = {}
): Promise<T> {
  const headers = new Headers(init.headers ?? {});
  if (!(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  if (options.sessionToken) {
    headers.set("X-Session-Token", options.sessionToken);
  }
  if (options.authToken) {
    headers.set("Authorization", `Bearer ${options.authToken}`);
  }

  const candidates = apiCandidates();
  let networkFailure = false;
  let lastApiError: Error | null = null;

  try {
    for (const base of candidates) {
      let response: Response;
      try {
        response = await fetch(`${base}${path}`, {
          ...init,
          headers
        });
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
          // Keep fallback detail.
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

export async function registerCustomer(
  email: string,
  password: string,
  name?: string,
  phone?: string
): Promise<AuthResponse> {
  return request<AuthResponse>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password, role: "customer", name, phone })
  });
}

export async function loginCustomer(email: string, password: string): Promise<AuthResponse> {
  return request<AuthResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password })
  });
}

export async function getCurrentUser(authToken: string): Promise<UserPublic> {
  return request<UserPublic>("/api/auth/me", {}, { authToken });
}

export async function ensureSession(sessionToken: string, authToken?: string): Promise<ChatSessionResponse> {
  return request<ChatSessionResponse>(
    "/api/chat/session",
    {
      method: "POST"
    },
    { sessionToken, authToken }
  );
}

export async function askQuestion(question: string, sessionToken: string, authToken?: string): Promise<AskResponse> {
  return request<AskResponse>(
    "/api/chat/ask",
    {
      method: "POST",
      body: JSON.stringify({ question })
    },
    { sessionToken, authToken }
  );
}

export async function getHistory(sessionToken: string, authToken?: string): Promise<ChatHistoryRecord[]> {
  return request<ChatHistoryRecord[]>("/api/chat/history", {}, { sessionToken, authToken });
}

export async function listBookingSlots(availableOnly = true): Promise<BookingSlotRecord[]> {
  const query = availableOnly ? "?available_only=true" : "?available_only=false";
  return request<BookingSlotRecord[]>(`/api/booking-slots${query}`);
}

export async function createBooking(
  payload: {
    guest_name?: string;
    guest_email?: string;
    guest_phone?: string;
    service_type?: string;
    booking_date: string;
    booking_time: string;
    message?: string;
  },
  sessionToken?: string,
  authToken?: string
): Promise<BookingRecord> {
  return request<BookingRecord>(
    "/api/bookings",
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    { sessionToken, authToken }
  );
}

export async function getMyBookings(authToken: string): Promise<BookingRecord[]> {
  return request<BookingRecord[]>("/api/bookings/me", {}, { authToken });
}

export async function updateMyBooking(
  bookingId: string,
  payload: {
    status?: "cancelled";
    booking_date?: string;
    booking_time?: string;
    message?: string;
  },
  authToken: string
): Promise<BookingRecord> {
  return request<BookingRecord>(
    `/api/bookings/${bookingId}/me`,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    { authToken }
  );
}
