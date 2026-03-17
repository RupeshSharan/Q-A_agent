import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  AdminAccountRecord,
  AskResponse,
  BookingRecord,
  BookingSlotRecord,
  ChatHistoryRecord,
  DocumentRecord,
  DocumentSuggestionRecord,
  NotificationRecord,
  OrganisationSuggestionRecord,
  UserPublic,
  askQuestion,
  createDocumentSuggestion,
  createOrganisationSuggestion,
  createStaffAccount,
  getCurrentUser,
  getHistory,
  listBookingSlots,
  listBookings,
  listDocumentSuggestions,
  listDocuments,
  listNotifications,
  listOrganisationSuggestions,
  listStaffAccounts,
  listUsers,
  login,
  registerAdmin,
  reviewDocumentSuggestion,
  reviewOrganisationSuggestion,
  updateBookingStatus,
  updateStaffAccountStatus,
  uploadDocument
} from "./api";

const TOKEN_KEY = "docmind_admin_token";

type PortalRole = "admin" | "staff";
type Section =
  | "dashboard"
  | "documents"
  | "bookings"
  | "staff-suggestions"
  | "document-updates"
  | "manage-staff"
  | "analytics"
  | "settings"
  | "chat"
  | "view-bookings"
  | "suggest-document"
  | "suggest-improvement"
  | "my-submissions";

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function statusClass(status: string): string {
  return `status-pill status-${status}`;
}

function isAuthFailure(err: unknown): boolean {
  return err instanceof Error && (err.message.includes("401") || err.message.toLowerCase().includes("token"));
}

function App() {
  const [token, setToken] = useState<string>(() => localStorage.getItem(TOKEN_KEY) ?? "");
  const [user, setUser] = useState<UserPublic | null>(null);
  const [entryRole, setEntryRole] = useState<PortalRole | null>(null);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [activeSection, setActiveSection] = useState<Section>("dashboard");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [registerName, setRegisterName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [latestAnswer, setLatestAnswer] = useState<AskResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [users, setUsers] = useState<UserPublic[]>([]);
  const [staffAccounts, setStaffAccounts] = useState<AdminAccountRecord[]>([]);
  const [history, setHistory] = useState<ChatHistoryRecord[]>([]);
  const [bookings, setBookings] = useState<BookingRecord[]>([]);
  const [slots, setSlots] = useState<BookingSlotRecord[]>([]);
  const [notifications, setNotifications] = useState<NotificationRecord[]>([]);
  const [docSuggestions, setDocSuggestions] = useState<DocumentSuggestionRecord[]>([]);
  const [orgSuggestions, setOrgSuggestions] = useState<OrganisationSuggestionRecord[]>([]);
  const [bookingFilter, setBookingFilter] = useState<"all" | "pending" | "confirmed" | "cancelled">("all");
  const [bookingDrafts, setBookingDrafts] = useState<Record<string, { status: BookingRecord["status"]; adminNote: string; notifiedVia: "" | "email" | "whatsapp" | "both" }>>({});
  const [docDrafts, setDocDrafts] = useState<Record<string, { status: "approved" | "rejected"; adminNote: string }>>({});
  const [orgDrafts, setOrgDrafts] = useState<Record<string, { status: "approved" | "rejected"; adminNote: string }>>({});
  const [docSuggestionDocumentId, setDocSuggestionDocumentId] = useState(0);
  const [docSuggestionText, setDocSuggestionText] = useState("");
  const [docSuggestionContent, setDocSuggestionContent] = useState("");
  const [orgTitle, setOrgTitle] = useState("");
  const [orgCategory, setOrgCategory] = useState<"service" | "timing" | "pricing" | "other">("service");
  const [orgDescription, setOrgDescription] = useState("");
  const [staffName, setStaffName] = useState("");
  const [staffEmail, setStaffEmail] = useState("");
  const [staffPassword, setStaffPassword] = useState("");
  const [showNotifications, setShowNotifications] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  const role = user?.role === "admin" || user?.role === "staff" ? user.role : null;
  const isAdmin = role === "admin";
  const filteredBookings = bookingFilter === "all" ? bookings : bookings.filter((item) => item.status === bookingFilter);
  const pendingBookings = bookings.filter((item) => item.status === "pending").length;
  const pendingDocSuggestions = docSuggestions.filter((item) => item.status === "pending").length;
  const pendingOrgSuggestions = orgSuggestions.filter((item) => item.status === "pending").length;
  const navItems = useMemo(
    () =>
      role === "admin"
        ? [
            ["dashboard", "Dashboard"],
            ["documents", "Documents"],
            ["bookings", `Bookings${pendingBookings ? ` (${pendingBookings})` : ""}`],
            ["staff-suggestions", `Staff Suggestions${pendingOrgSuggestions ? ` (${pendingOrgSuggestions})` : ""}`],
            ["document-updates", `Document Updates${pendingDocSuggestions ? ` (${pendingDocSuggestions})` : ""}`],
            ["manage-staff", "Manage Staff"],
            ["analytics", "Analytics"],
            ["settings", "Settings"]
          ]
        : [
            ["dashboard", "Dashboard"],
            ["chat", "Chat with AI"],
            ["view-bookings", "View Bookings"],
            ["suggest-document", "Suggest Document Update"],
            ["suggest-improvement", "Suggest Improvement"],
            ["my-submissions", "My Submissions"]
          ],
    [pendingBookings, pendingDocSuggestions, pendingOrgSuggestions, role]
  ) as Array<[Section, string]>;

  useEffect(() => {
    if (!token) {
      return;
    }
    void (async () => {
      try {
        const me = await getCurrentUser(token);
        if (me.role !== "admin" && me.role !== "staff") {
          throw new Error("This portal is only for admin and staff accounts.");
        }
        setUser(me);
        setEntryRole(me.role);
        await refresh(token, me.role);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unexpected error");
        if (isAuthFailure(err)) {
          logout();
        }
      }
    })();
  }, [token]);

  async function refresh(activeToken = token, activeRole = role): Promise<void> {
    if (!activeToken || !activeRole) {
      return;
    }
    if (activeRole === "admin") {
      const [docs, allUsers, staffRows, chats, bookingRows, slotRows, noticeRows, docRows, orgRows] = await Promise.all([
        listDocuments(activeToken),
        listUsers(activeToken),
        listStaffAccounts(activeToken),
        getHistory(activeToken),
        listBookings(activeToken),
        listBookingSlots(activeToken, false),
        listNotifications(activeToken),
        listDocumentSuggestions(activeToken),
        listOrganisationSuggestions(activeToken)
      ]);
      setUsers(allUsers);
      setStaffAccounts(staffRows);
      setSlots(slotRows);
      setNotifications(noticeRows);
      setDocuments(docs);
      setHistory(chats);
      setBookings(bookingRows);
      setDocSuggestions(docRows);
      setOrgSuggestions(orgRows);
      if (!docSuggestionDocumentId && docs.length > 0) {
        setDocSuggestionDocumentId(docs[0].id);
      }
    } else {
      const [docs, chats, bookingRows, docRows, orgRows] = await Promise.all([
        listDocuments(activeToken),
        getHistory(activeToken),
        listBookings(activeToken),
        listDocumentSuggestions(activeToken),
        listOrganisationSuggestions(activeToken)
      ]);
      setDocuments(docs);
      setHistory(chats);
      setBookings(bookingRows);
      setDocSuggestions(docRows);
      setOrgSuggestions(orgRows);
      setUsers([]);
      setStaffAccounts([]);
      setSlots([]);
      setNotifications([]);
      if (!docSuggestionDocumentId && docs.length > 0) {
        setDocSuggestionDocumentId(docs[0].id);
      }
    }
  }

  function logout(): void {
    localStorage.removeItem(TOKEN_KEY);
    setToken("");
    setUser(null);
    setEntryRole(null);
    setActiveSection("dashboard");
  }

  async function onAuthSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!entryRole) {
      return;
    }
    setLoading(true);
    setError("");
    setStatus("");
    try {
      const result =
        entryRole === "admin" && authMode === "register"
          ? await registerAdmin(email.trim(), password, registerName.trim() || undefined)
          : await login(email.trim(), password, entryRole);
      persist(result.access_token);
      setUser(result.user);
      setStatus(authMode === "register" ? "Admin account created." : "Login successful.");
      await refresh(result.access_token, result.user.role as PortalRole);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  function persist(nextToken: string): void {
    setToken(nextToken);
    localStorage.setItem(TOKEN_KEY, nextToken);
  }

  async function runAction(action: () => Promise<unknown>, successMessage?: string): Promise<void> {
    setLoading(true);
    setError("");
    setStatus("");
    try {
      await action();
      if (successMessage) {
        setStatus(successMessage);
      }
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  const bookingDraft = (booking: BookingRecord) =>
    bookingDrafts[booking.id] ?? { status: booking.status, adminNote: booking.admin_note ?? "", notifiedVia: "" as const };
  const docDraft = (id: string) => docDrafts[id] ?? { status: "approved" as const, adminNote: "" };
  const orgDraft = (id: string) => orgDrafts[id] ?? { status: "approved" as const, adminNote: "" };

  function renderHistory(title: string): JSX.Element {
    return (
      <section className="card">
        <h2>{title}</h2>
        {history.length === 0 ? <p>No chat history yet.</p> : (
          <div className="history-list">
            {history.slice(0, 10).map((item) => (
              <article key={item.id} className="history-item">
                <header><strong>{item.user_email ?? item.customer_id ?? "Guest"}</strong><span>{formatDate(item.created_at)}</span></header>
                <p><strong>Q:</strong> {item.question}</p>
                <p><strong>A:</strong> {item.answer}</p>
              </article>
            ))}
          </div>
        )}
      </section>
    );
  }

  function renderDocuments(title: string): JSX.Element {
    return (
      <section className="card">
        <h2>{title}</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>ID</th><th>Filename</th><th>Chunks</th><th>Uploader</th><th>Uploaded</th></tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.id}>
                  <td>{doc.id}</td>
                  <td>{doc.filename}</td>
                  <td>{doc.chunks_indexed}</td>
                  <td>{doc.uploaded_by_email ?? "Unknown"}</td>
                  <td>{formatDate(doc.uploaded_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    );
  }

  function renderBookings(readOnly: boolean): JSX.Element {
    return (
      <>
        <section className="card">
          <h2>{readOnly ? "View Bookings" : "Bookings"}</h2>
          <div className="toolbar">
            <label className="compact">Filter<select value={bookingFilter} onChange={(e) => setBookingFilter(e.target.value as "all" | "pending" | "confirmed" | "cancelled")}><option value="all">All</option><option value="pending">Pending</option><option value="confirmed">Confirmed</option><option value="cancelled">Cancelled</option></select></label>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>When</th><th>Customer</th><th>Service</th><th>Status</th>{readOnly ? null : <th>Note</th>}{readOnly ? null : <th>Notify</th>}{readOnly ? null : <th>Action</th>}</tr>
              </thead>
              <tbody>
                {filteredBookings.map((booking) => {
                  const draft = bookingDraft(booking);
                  return (
                    <tr key={booking.id}>
                      <td>{booking.booking_date} {booking.booking_time}</td>
                      <td>{booking.guest_name || booking.guest_email || booking.guest_phone || booking.customer_id || "N/A"}</td>
                      <td>{booking.service_type ?? "consultation"}</td>
                      <td>{readOnly ? <span className={statusClass(booking.status)}>{booking.status}</span> : <select value={draft.status} onChange={(e) => setBookingDrafts((prev) => ({ ...prev, [booking.id]: { ...draft, status: e.target.value as BookingRecord["status"] } }))}><option value="pending">pending</option><option value="confirmed">confirmed</option><option value="cancelled">cancelled</option></select>}</td>
                      {readOnly ? null : <td><input value={draft.adminNote} onChange={(e) => setBookingDrafts((prev) => ({ ...prev, [booking.id]: { ...draft, adminNote: e.target.value } }))} /></td>}
                      {readOnly ? null : <td><select value={draft.notifiedVia} onChange={(e) => setBookingDrafts((prev) => ({ ...prev, [booking.id]: { ...draft, notifiedVia: e.target.value as "" | "email" | "whatsapp" | "both" } }))}><option value="">none</option><option value="email">email</option><option value="whatsapp">whatsapp</option><option value="both">both</option></select></td>}
                      {readOnly ? null : <td><button type="button" onClick={() => void runAction(() => updateBookingStatus(booking.id, { status: draft.status, admin_note: draft.adminNote || null, notified_via: draft.notifiedVia || null }, token), "Booking updated.")} disabled={loading}>Save</button></td>}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
        {readOnly ? null : <section className="card"><h2>Slot Status</h2><div className="table-wrap"><table><thead><tr><th>Date</th><th>Time</th><th>Booked</th><th>Blocked</th></tr></thead><tbody>{slots.map((slot) => <tr key={slot.id}><td>{slot.slot_date}</td><td>{slot.slot_time}</td><td>{slot.is_booked ? "yes" : "no"}</td><td>{slot.is_blocked ? "yes" : "no"}</td></tr>)}</tbody></table></div></section>}
      </>
    );
  }

  function renderContent(): JSX.Element {
    if (role === "admin") {
      switch (activeSection) {
        case "dashboard":
          return <><section className="card"><h2>Dashboard</h2><div className="stats-grid"><article className="stat-tile"><strong>{documents.length}</strong><span>Documents</span></article><article className="stat-tile"><strong>{pendingBookings}</strong><span>Pending bookings</span></article><article className="stat-tile"><strong>{pendingDocSuggestions}</strong><span>Pending doc updates</span></article><article className="stat-tile"><strong>{pendingOrgSuggestions}</strong><span>Pending suggestions</span></article><article className="stat-tile"><strong>{history.length}</strong><span>Total chats</span></article><article className="stat-tile"><strong>{staffAccounts.filter((item) => item.role === "staff" && item.is_active).length}</strong><span>Active staff</span></article></div></section>{renderHistory("Recent Chat Activity")}</>;
        case "documents":
          return <><section className="card"><h2>Upload Documents</h2><form onSubmit={(event) => { event.preventDefault(); if (!file) { return; } void runAction(async () => { await uploadDocument(file, token); setFile(null); }, "Document uploaded."); }} className="form-grid"><label>Upload PDF/TXT<input type="file" accept=".pdf,.txt" onChange={(e) => setFile(e.target.files?.[0] ?? null)} required /></label><div className="form-actions"><button type="submit" disabled={loading || !file}>{loading ? "Uploading..." : "Upload + Ingest"}</button></div></form></section><section className="card"><h2>Chat with AI</h2><form onSubmit={(event) => { event.preventDefault(); if (!question.trim()) { return; } void runAction(async () => { setLatestAnswer(await askQuestion(question.trim(), token)); setQuestion(""); }); }} className="form-grid"><label>Question<textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={3} required /></label><div className="form-actions"><button type="submit" disabled={loading || !question.trim()}>{loading ? "Generating..." : "Ask"}</button></div></form>{latestAnswer ? <div className="answer-box"><h3>Latest Answer</h3><p>{latestAnswer.answer}</p></div> : null}</section>{renderDocuments("Uploaded Documents")}</>;
        case "bookings":
          return renderBookings(false);
        case "staff-suggestions":
          return <section className="card"><h2>Staff Suggestions</h2><div className="history-list">{orgSuggestions.map((item) => { const draft = orgDraft(item.id); return <article key={item.id} className="history-item"><header><strong>{item.title}</strong><span className={statusClass(item.status)}>{item.status}</span></header><p><strong>Category:</strong> {item.category}</p><p>{item.description}</p>{item.status === "pending" ? <div className="form-grid"><label>Decision<select value={draft.status} onChange={(e) => setOrgDrafts((prev) => ({ ...prev, [item.id]: { ...draft, status: e.target.value as "approved" | "rejected" } }))}><option value="approved">approve</option><option value="rejected">reject</option></select></label><label>Admin Note<textarea value={draft.adminNote} rows={2} onChange={(e) => setOrgDrafts((prev) => ({ ...prev, [item.id]: { ...draft, adminNote: e.target.value } }))} /></label><div className="form-actions"><button type="button" onClick={() => void runAction(() => reviewOrganisationSuggestion(item.id, { status: draft.status, admin_note: draft.adminNote || null }, token), "Suggestion reviewed.")} disabled={loading}>Save Review</button></div></div> : <p><strong>Admin Note:</strong> {item.admin_note ?? "No note"}</p>}</article>; })}</div></section>;
        case "document-updates":
          return <section className="card"><h2>Document Updates</h2><div className="history-list">{docSuggestions.map((item) => { const draft = docDraft(item.id); return <article key={item.id} className="history-item"><header><strong>{item.document_filename ?? `Document ${item.document_id}`}</strong><span className={statusClass(item.status)}>{item.status}</span></header><p>{item.suggestion_text}</p><p><strong>New Content:</strong> {item.new_content}</p>{item.status === "pending" ? <div className="form-grid"><label>Decision<select value={draft.status} onChange={(e) => setDocDrafts((prev) => ({ ...prev, [item.id]: { ...draft, status: e.target.value as "approved" | "rejected" } }))}><option value="approved">approve</option><option value="rejected">reject</option></select></label><label>Admin Note<textarea value={draft.adminNote} rows={2} onChange={(e) => setDocDrafts((prev) => ({ ...prev, [item.id]: { ...draft, adminNote: e.target.value } }))} /></label><div className="form-actions"><button type="button" onClick={() => void runAction(() => reviewDocumentSuggestion(item.id, { status: draft.status, admin_note: draft.adminNote || null }, token), "Document update reviewed.")} disabled={loading}>Save Review</button></div></div> : <p><strong>Admin Note:</strong> {item.admin_note ?? "No note"}</p>}</article>; })}</div></section>;
        case "manage-staff":
          return <><section className="card"><h2>Create Staff Account</h2><form onSubmit={(event) => { event.preventDefault(); void runAction(async () => { await createStaffAccount({ name: staffName.trim(), email: staffEmail.trim(), password: staffPassword }, token); setStaffName(""); setStaffEmail(""); setStaffPassword(""); }, "Staff account created."); }} className="form-grid"><label>Name<input value={staffName} onChange={(e) => setStaffName(e.target.value)} required /></label><label>Email<input type="email" value={staffEmail} onChange={(e) => setStaffEmail(e.target.value)} required /></label><label>Password<input type="password" value={staffPassword} onChange={(e) => setStaffPassword(e.target.value)} minLength={8} required /></label><div className="form-actions"><button type="submit" disabled={loading}>{loading ? "Creating..." : "Create Staff"}</button></div></form></section><section className="card"><h2>Portal Accounts</h2><div className="table-wrap"><table><thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Status</th><th>Created By</th><th>Action</th></tr></thead><tbody>{staffAccounts.map((account) => <tr key={account.id}><td>{account.name}</td><td>{account.email}</td><td>{account.role}</td><td>{account.is_active ? "active" : "inactive"}</td><td>{account.created_by_email ?? "System"}</td><td>{account.role === "staff" ? <button type="button" onClick={() => void runAction(() => updateStaffAccountStatus(account.id, { is_active: !account.is_active }, token), "Staff account updated.")} disabled={loading}>{account.is_active ? "Deactivate" : "Activate"}</button> : <span className="muted">Admin account</span>}</td></tr>)}</tbody></table></div></section></>;
        case "analytics":
          return <><section className="card"><h2>Analytics</h2><div className="stats-grid"><article className="stat-tile"><strong>{users.length}</strong><span>Total users</span></article><article className="stat-tile"><strong>{users.filter((item) => item.role === "customer").length}</strong><span>Customers</span></article><article className="stat-tile"><strong>{bookings.length}</strong><span>Bookings</span></article><article className="stat-tile"><strong>{notifications.length}</strong><span>Notifications</span></article></div></section>{renderHistory("Chat History")}</>;
        case "settings":
          return <section className="card"><h2>Settings</h2><p><strong>Name:</strong> {user?.name ?? "Not set"}</p><p><strong>Email:</strong> {user?.email}</p><p>Bookings run Monday to Saturday from 09:00 to 17:00. Staff can submit suggestions, but only admin can approve them.</p></section>;
        default:
          return <section className="card"><h2>Dashboard</h2></section>;
      }
    }

    switch (activeSection) {
      case "dashboard":
        return <section className="card"><h2>Dashboard</h2><div className="stats-grid"><article className="stat-tile"><strong>{docSuggestions.length + orgSuggestions.length}</strong><span>Submissions</span></article><article className="stat-tile"><strong>{pendingDocSuggestions + pendingOrgSuggestions}</strong><span>Pending review</span></article><article className="stat-tile"><strong>{bookings.filter((item) => item.status !== "cancelled").length}</strong><span>Visible bookings</span></article><article className="stat-tile"><strong>{history.length}</strong><span>Your chats</span></article></div></section>;
      case "chat":
        return <><section className="card"><h2>Chat with AI</h2><form onSubmit={(event) => { event.preventDefault(); if (!question.trim()) { return; } void runAction(async () => { setLatestAnswer(await askQuestion(question.trim(), token)); setQuestion(""); }); }} className="form-grid"><label>Question<textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={3} required /></label><div className="form-actions"><button type="submit" disabled={loading || !question.trim()}>{loading ? "Generating..." : "Ask"}</button></div></form>{latestAnswer ? <div className="answer-box"><h3>Latest Answer</h3><p>{latestAnswer.answer}</p></div> : null}</section>{renderHistory("Your Chat History")}</>;
      case "view-bookings":
        return renderBookings(true);
      case "suggest-document":
        return <><section className="card"><h2>Suggest Document Update</h2><form onSubmit={(event) => { event.preventDefault(); void runAction(async () => { await createDocumentSuggestion({ document_id: docSuggestionDocumentId, suggestion_text: docSuggestionText.trim(), new_content: docSuggestionContent.trim() }, token); setDocSuggestionText(""); setDocSuggestionContent(""); setActiveSection("my-submissions"); }, "Document update suggestion submitted."); }} className="form-grid"><label>Document<select value={docSuggestionDocumentId} onChange={(e) => setDocSuggestionDocumentId(Number(e.target.value))}><option value={0}>Select a document</option>{documents.map((doc) => <option key={doc.id} value={doc.id}>#{doc.id} {doc.filename}</option>)}</select></label><label>What should change?<textarea value={docSuggestionText} onChange={(e) => setDocSuggestionText(e.target.value)} rows={3} required /></label><label>Proposed new content<textarea value={docSuggestionContent} onChange={(e) => setDocSuggestionContent(e.target.value)} rows={5} required /></label><div className="form-actions"><button type="submit" disabled={loading || !docSuggestionDocumentId}>{loading ? "Submitting..." : "Submit Update"}</button></div></form></section>{renderDocuments("Available Documents")}</>;
      case "suggest-improvement":
        return <section className="card"><h2>Suggest Improvement</h2><form onSubmit={(event) => { event.preventDefault(); void runAction(async () => { await createOrganisationSuggestion({ title: orgTitle.trim(), description: orgDescription.trim(), category: orgCategory }, token); setOrgTitle(""); setOrgDescription(""); setActiveSection("my-submissions"); }, "Improvement suggestion submitted."); }} className="form-grid"><label>Title<input value={orgTitle} onChange={(e) => setOrgTitle(e.target.value)} required /></label><label>Category<select value={orgCategory} onChange={(e) => setOrgCategory(e.target.value as "service" | "timing" | "pricing" | "other")}><option value="service">service</option><option value="timing">timing</option><option value="pricing">pricing</option><option value="other">other</option></select></label><label>Description<textarea value={orgDescription} onChange={(e) => setOrgDescription(e.target.value)} rows={5} required /></label><div className="form-actions"><button type="submit" disabled={loading}>{loading ? "Submitting..." : "Submit Improvement"}</button></div></form></section>;
      case "my-submissions":
        return <><section className="card"><h2>Document Update Submissions</h2><div className="history-list">{docSuggestions.map((item) => <article key={item.id} className="history-item"><header><strong>{item.document_filename ?? `Document ${item.document_id}`}</strong><span className={statusClass(item.status)}>{item.status}</span></header><p>{item.suggestion_text}</p><p><strong>Admin Note:</strong> {item.admin_note ?? "No note yet"}</p></article>)}</div></section><section className="card"><h2>Improvement Submissions</h2><div className="history-list">{orgSuggestions.map((item) => <article key={item.id} className="history-item"><header><strong>{item.title}</strong><span className={statusClass(item.status)}>{item.status}</span></header><p>{item.description}</p><p><strong>Admin Note:</strong> {item.admin_note ?? "No note yet"}</p></article>)}</div></section></>;
      default:
        return <section className="card"><h2>Dashboard</h2></section>;
    }
  }

  if (!role) {
    return (
      <div className="page">
        {entryRole ? (
          <section className="auth-card">
            <h1>{entryRole === "admin" ? "Admin Access" : "Staff Access"}</h1>
            <p>{entryRole === "admin" ? "Register only for the first admin account." : "Staff accounts are created by an admin."}</p>
            <form onSubmit={onAuthSubmit} className="form-grid">
              {entryRole === "admin" && authMode === "register" ? <label>Name<input value={registerName} onChange={(e) => setRegisterName(e.target.value)} required /></label> : null}
              <label>Email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required /></label>
              <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} minLength={8} required /></label>
              <div className="form-actions">
                <button type="submit" disabled={loading}>{loading ? "Please wait..." : authMode === "register" ? "Create Admin" : "Login"}</button>
                {entryRole === "admin" ? <button type="button" className="secondary" onClick={() => setAuthMode((current) => current === "login" ? "register" : "login")}>{authMode === "login" ? "Need first admin?" : "Back to login"}</button> : null}
                <button type="button" className="secondary" onClick={() => { setEntryRole(null); setAuthMode("login"); }}>Back</button>
              </div>
            </form>
          </section>
        ) : (
          <section className="auth-card">
            <h1>DocMind AI</h1>
            <p>Choose who you are. The same URL loads a different interface after login.</p>
            <div className="role-pick-grid">
              <button type="button" className="role-card" onClick={() => setEntryRole("admin")}><strong>Admin</strong><span>Documents, bookings, approvals, analytics.</span></button>
              <button type="button" className="role-card" onClick={() => setEntryRole("staff")}><strong>Staff</strong><span>AI chat, booking visibility, and suggestions.</span></button>
            </div>
          </section>
        )}
        {error ? <p className="notice error">{error}</p> : null}
        {status ? <p className="notice success">{status}</p> : null}
      </div>
    );
  }

  return (
    <div className="page">
      <div className="portal-shell">
        <aside className="sidebar">
          <div className="brand-block">
            <h1>DocMind</h1>
            <p>{role === "admin" ? "Admin Portal" : "Staff Portal"}</p>
          </div>
          <section className="nav-group">
            <div className="nav-items">
              {navItems.map(([id, label]) => (
                <button key={id} type="button" className={`nav-item ${activeSection === id ? "active-nav-item" : ""}`} onClick={() => setActiveSection(id)}>
                  {label}
                </button>
              ))}
            </div>
          </section>
        </aside>
        <section className="workspace">
          <header className="workspace-topbar">
            <div>
              <h2>{user?.name ?? user?.email}</h2>
              <p>{role === "admin" ? "Full admin controls and approval workflow." : "Staff workspace for suggestions and AI chat."}</p>
            </div>
            <div className="workspace-actions">
              <button type="button" className="secondary" onClick={() => void refresh()} disabled={loading}>Refresh</button>
              {isAdmin ? (
                <button type="button" className={`icon-button ${showNotifications ? "active-icon-button" : ""}`} onClick={() => setShowNotifications((current) => !current)} aria-label="Toggle notifications">
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 4a4 4 0 0 1 4 4v2.7c0 .9.3 1.8.9 2.5l1.4 1.7a1 1 0 0 1-.8 1.6H6.5a1 1 0 0 1-.8-1.6l1.4-1.7c.6-.7.9-1.6.9-2.5V8a4 4 0 0 1 4-4Z" fill="currentColor" />
                    <path d="M9.5 18a2.5 2.5 0 0 0 5 0" fill="none" stroke="currentColor" strokeWidth="1.7" />
                  </svg>
                  {notifications.length > 0 ? <span className="badge">{notifications.length}</span> : null}
                </button>
              ) : null}
              <button type="button" onClick={logout}>Logout</button>
            </div>
          </header>
          <main className="workspace-content">{renderContent()}</main>
        </section>
        {isAdmin && showNotifications ? (
          <aside className="notifications-drawer">
            <header>
              <h3>Notifications Log</h3>
              <button type="button" className="secondary" onClick={() => setShowNotifications(false)}>Close</button>
            </header>
            <div className="history-list">
              {notifications.map((item) => (
                <article key={item.id} className="history-item">
                  <header><strong>{item.channel}</strong><span>{formatDate(item.created_at)}</span></header>
                  <p><strong>Booking:</strong> {item.booking_id ?? "N/A"}</p>
                  <p><strong>Recipient:</strong> {item.recipient_email ?? item.recipient_phone ?? "N/A"}</p>
                  <p><strong>Status:</strong> {item.status}</p>
                </article>
              ))}
            </div>
          </aside>
        ) : null}
      </div>
      {error ? <p className="notice error">{error}</p> : null}
      {status ? <p className="notice success">{status}</p> : null}
    </div>
  );
}

export default App;
