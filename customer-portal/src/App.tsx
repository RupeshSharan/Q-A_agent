import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  AskResponse,
  BookingRecord,
  BookingSlotRecord,
  ChatHistoryRecord,
  UserPublic,
  askQuestion,
  createBooking,
  ensureSession,
  getCurrentUser,
  getHistory,
  getMyBookings,
  listBookingSlots,
  loginCustomer,
  registerCustomer,
  updateMyBooking
} from "./api";

const SESSION_TOKEN_KEY = "docmind_customer_session_token";
const AUTH_TOKEN_KEY = "docmind_customer_auth_token";

type CustomerSection = "dashboard" | "bookings";

interface NavItem {
  id: CustomerSection;
  label: string;
}

interface NavGroup {
  title: string;
  items: NavItem[];
}

const NAV_GROUPS: NavGroup[] = [
  {
    title: "",
    items: [
      { id: "dashboard", label: "Dashboard" },
      { id: "bookings", label: "Bookings" }
    ]
  }
];

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function buildSessionToken(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID().replace(/-/g, "");
  }
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 14)}`;
}

function getOrCreateSessionToken(): string {
  const existing = localStorage.getItem(SESSION_TOKEN_KEY);
  if (existing) {
    return existing;
  }
  const created = buildSessionToken();
  localStorage.setItem(SESSION_TOKEN_KEY, created);
  return created;
}

function App() {
  const [activeSection, setActiveSection] = useState<CustomerSection>("dashboard");
  const [sessionToken, setSessionToken] = useState<string>(() => getOrCreateSessionToken());
  const [authToken, setAuthToken] = useState<string>(() => localStorage.getItem(AUTH_TOKEN_KEY) ?? "");
  const [currentUser, setCurrentUser] = useState<UserPublic | null>(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authName, setAuthName] = useState("");
  const [authEmail, setAuthEmail] = useState("");
  const [authPhone, setAuthPhone] = useState("");
  const [authPassword, setAuthPassword] = useState("");

  const [question, setQuestion] = useState("");
  const [latestAnswer, setLatestAnswer] = useState<AskResponse | null>(null);
  const [history, setHistory] = useState<ChatHistoryRecord[]>([]);

  const [slots, setSlots] = useState<BookingSlotRecord[]>([]);
  const [serviceType, setServiceType] = useState("consultation");
  const [bookingName, setBookingName] = useState("");
  const [bookingEmail, setBookingEmail] = useState("");
  const [bookingPhone, setBookingPhone] = useState("");
  const [bookingDate, setBookingDate] = useState("");
  const [bookingTime, setBookingTime] = useState("");
  const [bookingMessage, setBookingMessage] = useState("");
  const [myBookings, setMyBookings] = useState<BookingRecord[]>([]);
  const [rescheduleDate, setRescheduleDate] = useState<Record<string, string>>({});
  const [rescheduleTime, setRescheduleTime] = useState<Record<string, string>>({});
  const [rescheduleMessage, setRescheduleMessage] = useState<Record<string, string>>({});

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  const isLoggedIn = useMemo(() => Boolean(authToken && currentUser), [authToken, currentUser]);

  useEffect(() => {
    void initializeSession(sessionToken, authToken || undefined);
  }, [sessionToken, authToken]);

  useEffect(() => {
    void refreshSlots();
  }, []);

  useEffect(() => {
    if (!authToken) {
      setCurrentUser(null);
      setMyBookings([]);
      return;
    }
    void initializeCustomer(authToken);
  }, [authToken]);

  async function initializeSession(activeSessionToken: string, activeAuthToken?: string): Promise<void> {
    setLoading(true);
    setError("");
    try {
      const session = await ensureSession(activeSessionToken, activeAuthToken);
      if (session.session_token && session.session_token !== activeSessionToken) {
        localStorage.setItem(SESSION_TOKEN_KEY, session.session_token);
        setSessionToken(session.session_token);
        return;
      }
      const rows = await getHistory(activeSessionToken, activeAuthToken);
      setHistory(rows);
      setStatus("Session ready.");
    } catch (err) {
      handleError(err);
    } finally {
      setLoading(false);
    }
  }

  async function initializeCustomer(activeAuthToken: string): Promise<void> {
    try {
      const me = await getCurrentUser(activeAuthToken);
      setCurrentUser(me);
      setBookingEmail((prev) => (prev ? prev : me.email));
      await refreshMyBookings(activeAuthToken);
    } catch (err) {
      handleError(err);
      clearAuth();
    }
  }

  async function refreshSlots(): Promise<void> {
    try {
      const available = await listBookingSlots(true);
      setSlots(available);
    } catch (err) {
      handleError(err);
    }
  }

  async function refreshMyBookings(activeAuthToken = authToken): Promise<void> {
    if (!activeAuthToken) {
      return;
    }
    const rows = await getMyBookings(activeAuthToken);
    setMyBookings(rows);
  }

  function handleError(err: unknown): void {
    if (err instanceof Error) {
      setError(err.message);
      return;
    }
    setError("Unexpected error");
  }

  function clearAuth(): void {
    setAuthToken("");
    setCurrentUser(null);
    setMyBookings([]);
    setIsAuthModalOpen(false);
    localStorage.removeItem(AUTH_TOKEN_KEY);
  }

  function resetGuestSession(): void {
    const nextToken = buildSessionToken();
    localStorage.setItem(SESSION_TOKEN_KEY, nextToken);
    setSessionToken(nextToken);
    setQuestion("");
    setLatestAnswer(null);
    setHistory([]);
    setError("");
    setStatus("Started a new guest session.");
  }

  async function onAuthSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setLoading(true);
    setError("");
    setStatus("");
    try {
      const result =
        authMode === "login"
          ? await loginCustomer(authEmail.trim(), authPassword)
          : await registerCustomer(authEmail.trim(), authPassword, authName.trim(), authPhone.trim());
      localStorage.setItem(AUTH_TOKEN_KEY, result.access_token);
      setAuthToken(result.access_token);
      setCurrentUser(result.user);
      setAuthPassword("");
      setIsAuthModalOpen(false);
      setStatus(authMode === "login" ? "Logged in successfully." : "Customer account created.");
      await refreshMyBookings(result.access_token);
      setActiveSection("bookings");
    } catch (err) {
      handleError(err);
    } finally {
      setLoading(false);
    }
  }

  async function onAskSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!question.trim()) {
      return;
    }
    setLoading(true);
    setError("");
    setStatus("");
    try {
      const answer = await askQuestion(question.trim(), sessionToken, authToken || undefined);
      setLatestAnswer(answer);
      setQuestion("");
      const rows = await getHistory(sessionToken, authToken || undefined);
      setHistory(rows);
      setActiveSection("dashboard");
    } catch (err) {
      handleError(err);
    } finally {
      setLoading(false);
    }
  }

  async function onBookingSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!bookingDate || !bookingTime) {
      return;
    }

    setLoading(true);
    setError("");
    setStatus("");
    try {
      const payload = {
        guest_name: isLoggedIn ? undefined : bookingName.trim(),
        guest_email: isLoggedIn ? undefined : bookingEmail.trim(),
        guest_phone: isLoggedIn ? undefined : bookingPhone.trim(),
        service_type: serviceType.trim(),
        booking_date: bookingDate,
        booking_time: bookingTime,
        message: bookingMessage.trim() || undefined
      };
      const booking = await createBooking(payload, sessionToken, authToken || undefined);
      setStatus(`Booking ${booking.id} created. Admin will manage approval and confirmation.`);
      setBookingDate("");
      setBookingTime("");
      setBookingMessage("");
      if (!isLoggedIn) {
        setBookingName("");
        setBookingEmail("");
        setBookingPhone("");
      }
      await refreshSlots();
      if (authToken) {
        await refreshMyBookings(authToken);
        setActiveSection("bookings");
      } else {
        setActiveSection("bookings");
      }
    } catch (err) {
      handleError(err);
    } finally {
      setLoading(false);
    }
  }

  async function onCancelBooking(bookingId: string): Promise<void> {
    if (!authToken) {
      return;
    }
    setLoading(true);
    setError("");
    setStatus("");
    try {
      await updateMyBooking(bookingId, { status: "cancelled" }, authToken);
      setStatus(`Booking ${bookingId} cancelled.`);
      await Promise.all([refreshMyBookings(authToken), refreshSlots()]);
    } catch (err) {
      handleError(err);
    } finally {
      setLoading(false);
    }
  }

  async function onRescheduleBooking(bookingId: string): Promise<void> {
    if (!authToken) {
      return;
    }
    const nextDate = (rescheduleDate[bookingId] ?? "").trim();
    const nextTime = (rescheduleTime[bookingId] ?? "").trim();
    const nextMessage = (rescheduleMessage[bookingId] ?? "").trim();
    if (!nextDate || !nextTime) {
      setError("Select both date and time to reschedule.");
      return;
    }
    setLoading(true);
    setError("");
    setStatus("");
    try {
      await updateMyBooking(
        bookingId,
        {
          booking_date: nextDate,
          booking_time: nextTime,
          message: nextMessage || undefined
        },
        authToken
      );
      setStatus(`Booking ${bookingId} rescheduled.`);
      setRescheduleDate((prev) => ({ ...prev, [bookingId]: "" }));
      setRescheduleTime((prev) => ({ ...prev, [bookingId]: "" }));
      setRescheduleMessage((prev) => ({ ...prev, [bookingId]: "" }));
      await Promise.all([refreshMyBookings(authToken), refreshSlots()]);
    } catch (err) {
      handleError(err);
    } finally {
      setLoading(false);
    }
  }

  function renderAskQuestionSection(): JSX.Element {
    return (
      <section className="card">
        <h2>Ask a Question</h2>
        <form onSubmit={onAskSubmit} className="form-grid">
          <label>
            Question
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What are the key terms in the uploaded policy?"
              rows={3}
              required
            />
          </label>
          <div className="form-actions">
            <button type="submit" disabled={loading || !question.trim()}>
              {loading ? "Generating..." : "Ask"}
            </button>
          </div>
        </form>

        {latestAnswer ? (
          <div className="answer-box">
            <h3>Latest Answer</h3>
            <p>{latestAnswer.answer}</p>
            <h4>Sources</h4>
            {latestAnswer.sources.length === 0 ? (
              <p>No sources returned.</p>
            ) : (
              <ul>
                {latestAnswer.sources.map((source, idx) => (
                  <li key={`${source.source}-${idx}`}>
                    [{idx + 1}] {source.source} (chunk {source.chunk_index})
                  </li>
                ))}
              </ul>
            )}
          </div>
        ) : null}
      </section>
    );
  }

  function renderChatHistorySection(): JSX.Element {
    return (
      <section className="card">
        <h2>Your Chat History</h2>
        {history.length === 0 ? (
          <p>No chat history yet.</p>
        ) : (
          <div className="history-list">
            {history.map((item) => (
              <article key={item.id} className="history-item">
                <header>
                  <strong>{formatDate(item.created_at)}</strong>
                </header>
                <p>
                  <strong>Q:</strong> {item.question}
                </p>
                <p>
                  <strong>A:</strong> {item.answer}
                </p>
              </article>
            ))}
          </div>
        )}
      </section>
    );
  }

  function renderBookingSection(): JSX.Element {
    return (
      <section className="card">
        <h2>Book Appointment</h2>
        <p className="muted">Slots are available Monday to Saturday between 09:00 and 17:00. Admin manages approvals.</p>
        <form onSubmit={onBookingSubmit} className="form-grid">
          {!isLoggedIn ? (
            <>
              <label>
                Name
                <input value={bookingName} onChange={(e) => setBookingName(e.target.value)} required />
              </label>
              <label>
                Email
                <input type="email" value={bookingEmail} onChange={(e) => setBookingEmail(e.target.value)} required />
              </label>
              <label>
                Phone
                <input value={bookingPhone} onChange={(e) => setBookingPhone(e.target.value)} required />
              </label>
            </>
          ) : null}
          <label>
            Service Type
            <input value={serviceType} onChange={(e) => setServiceType(e.target.value)} required />
          </label>
          <label>
            Date
            <input type="date" value={bookingDate} onChange={(e) => setBookingDate(e.target.value)} required />
          </label>
          <label>
            Time
            <input type="time" value={bookingTime} onChange={(e) => setBookingTime(e.target.value)} required />
          </label>
          <label>
            Message
            <textarea
              value={bookingMessage}
              onChange={(e) => setBookingMessage(e.target.value)}
              rows={3}
              placeholder="Optional note for the admin"
            />
          </label>
          <div className="form-actions">
            <button type="submit" disabled={loading || !bookingDate || !bookingTime}>
              {loading ? "Saving..." : "Book Appointment"}
            </button>
          </div>
        </form>
      </section>
    );
  }

  function renderSlotsSection(): JSX.Element {
    return (
      <section className="card">
        <h2>Available Slots</h2>
        {slots.length === 0 ? (
          <p>No available slots right now.</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Time</th>
                  <th>Pick</th>
                </tr>
              </thead>
              <tbody>
                {slots.map((slot) => (
                  <tr key={slot.id}>
                    <td>{slot.slot_date}</td>
                    <td>{slot.slot_time}</td>
                    <td>
                      <button
                        type="button"
                        onClick={() => {
                          setBookingDate(slot.slot_date);
                          setBookingTime(slot.slot_time);
                          setActiveSection("bookings");
                        }}
                        disabled={loading}
                      >
                        Use slot
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    );
  }

  function renderMyBookingsSection(): JSX.Element {
    if (!isLoggedIn) {
      return (
        <section className="card">
          <h2>My Bookings</h2>
          <p>Login or sign up to view and manage your bookings.</p>
          <div className="form-actions">
            <button type="button" onClick={() => setIsAuthModalOpen(true)}>
              Login / Sign Up
            </button>
          </div>
        </section>
      );
    }

    return (
      <section className="card">
        <h2>My Bookings</h2>
        {myBookings.length === 0 ? (
          <p>No bookings yet.</p>
        ) : (
          <div className="history-list">
            {myBookings.map((booking) => (
              <article key={booking.id} className="history-item">
                <header>
                  <strong>
                    {booking.booking_date} {booking.booking_time}
                  </strong>
                  <span>{booking.status}</span>
                </header>
                <p>
                  <strong>Service:</strong> {booking.service_type ?? "consultation"}
                </p>
                <p>
                  <strong>Message:</strong> {booking.message ?? "N/A"}
                </p>
                <p>
                  <strong>Admin Note:</strong> {booking.admin_note ?? "N/A"}
                </p>
                {booking.status !== "cancelled" ? (
                  <>
                    <div className="form-grid inline-grid">
                      <label>
                        New Date
                        <input
                          type="date"
                          value={rescheduleDate[booking.id] ?? ""}
                          onChange={(e) =>
                            setRescheduleDate((prev) => ({
                              ...prev,
                              [booking.id]: e.target.value
                            }))
                          }
                        />
                      </label>
                      <label>
                        New Time
                        <input
                          type="time"
                          value={rescheduleTime[booking.id] ?? ""}
                          onChange={(e) =>
                            setRescheduleTime((prev) => ({
                              ...prev,
                              [booking.id]: e.target.value
                            }))
                          }
                        />
                      </label>
                      <label>
                        Update Message
                        <input
                          value={rescheduleMessage[booking.id] ?? ""}
                          onChange={(e) =>
                            setRescheduleMessage((prev) => ({
                              ...prev,
                              [booking.id]: e.target.value
                            }))
                          }
                          placeholder="Optional update note"
                        />
                      </label>
                    </div>
                    <div className="form-actions">
                      <button type="button" onClick={() => onRescheduleBooking(booking.id)} disabled={loading}>
                        Reschedule
                      </button>
                      <button
                        type="button"
                        className="secondary"
                        onClick={() => onCancelBooking(booking.id)}
                        disabled={loading}
                      >
                        Cancel Booking
                      </button>
                    </div>
                  </>
                ) : null}
              </article>
            ))}
          </div>
        )}
      </section>
    );
  }

  function renderDashboardBucket(): JSX.Element {
    return (
      <>
        {renderAskQuestionSection()}
        {renderChatHistorySection()}
      </>
    );
  }

  function renderBookingsBucket(): JSX.Element {
    return (
      <>
        {renderBookingSection()}
        {renderSlotsSection()}
        {renderMyBookingsSection()}
      </>
    );
  }

  function renderContent(): JSX.Element {
    switch (activeSection) {
      case "dashboard":
        return renderDashboardBucket();
      case "bookings":
        return renderBookingsBucket();
      default:
        return renderDashboardBucket();
    }
  }

  return (
    <div className="page">
      <div className="portal-shell">
        <aside className="sidebar">
          <div className="brand-block">
            <h1>DocMind</h1>
            <p>Customer Portal</p>
          </div>
          {NAV_GROUPS.map((group) => (
            <section key={group.title} className="nav-group">
              {group.title.trim() ? <h2>{group.title}</h2> : null}
              <div className="nav-items">
                {group.items.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className={`nav-item ${activeSection === item.id ? "active-nav-item" : ""}`}
                    onClick={() => setActiveSection(item.id)}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </section>
          ))}
        </aside>

        <section className="workspace">
          <header className="workspace-topbar">
            <div>
              <h2>Ask questions and manage bookings</h2>
              <p>Session: {sessionToken.slice(0, 12)}...</p>
            </div>
            <div className="workspace-actions">
              {isLoggedIn ? <span className="account-pill">{currentUser?.email}</span> : <span className="account-pill">Guest mode</span>}
              {isLoggedIn ? (
                <button type="button" className="secondary" onClick={clearAuth} disabled={loading}>
                  Logout
                </button>
              ) : (
                <button type="button" onClick={() => setIsAuthModalOpen(true)} disabled={loading}>
                  Login / Sign Up
                </button>
              )}
              <button type="button" className="secondary" onClick={resetGuestSession} disabled={loading}>
                New Session
              </button>
            </div>
          </header>

          <main className="workspace-content">{renderContent()}</main>
        </section>
      </div>

      {isAuthModalOpen ? (
        <div className="modal-backdrop" role="dialog" aria-modal="true" onClick={() => setIsAuthModalOpen(false)}>
          <section className="auth-modal" onClick={(event) => event.stopPropagation()}>
            <header>
              <h3>{authMode === "login" ? "Customer Login" : "Create Customer Account"}</h3>
              <button type="button" className="secondary" onClick={() => setIsAuthModalOpen(false)}>
                Close
              </button>
            </header>
            <p>Login is optional. Use it to manage booking history and rescheduling.</p>
            <form onSubmit={onAuthSubmit} className="form-grid">
              {authMode === "register" ? (
                <>
                  <label>
                    Name
                    <input value={authName} onChange={(e) => setAuthName(e.target.value)} required />
                  </label>
                  <label>
                    Phone
                    <input value={authPhone} onChange={(e) => setAuthPhone(e.target.value)} required />
                  </label>
                </>
              ) : null}
              <label>
                Email
                <input type="email" value={authEmail} onChange={(e) => setAuthEmail(e.target.value)} required />
              </label>
              <label>
                Password
                <input
                  type="password"
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  minLength={8}
                  required
                />
              </label>
              <div className="form-actions">
                <button type="submit" disabled={loading}>
                  {loading ? "Please wait..." : authMode === "login" ? "Login" : "Register"}
                </button>
                <button
                  type="button"
                  className="secondary"
                  onClick={() => setAuthMode((current) => (current === "login" ? "register" : "login"))}
                  disabled={loading}
                >
                  {authMode === "login" ? "Create account" : "Back to login"}
                </button>
              </div>
            </form>
          </section>
        </div>
      ) : null}

      {error ? <p className="notice error">{error}</p> : null}
      {status ? <p className="notice success">{status}</p> : null}
    </div>
  );
}

export default App;
