# DocMind Platform

DocMind is now structured as a two-portal product on top of one FastAPI backend:

- `admin-portal` (React + TypeScript): admin login, document upload, user list, chat/history.
- `customer-portal` (React + TypeScript): customer login, Q&A chat, personal history.
- `backend` (FastAPI + SQLite + JWT + existing RAG modules): auth, docs, chat APIs.

The existing local RAG engine remains in `src/` and is reused by the backend.

## Architecture

1. Admin uploads PDF/TXT.
2. Backend saves file to `backend/storage/pdfs`.
3. Backend ingests file with existing pipeline (`src/ingestion.py`, `src/rag_pipeline.py`).
4. Chunks and embeddings are stored in Chroma (`backend/data/chroma` by default).
5. Customer asks a question.
6. Backend retrieves relevant chunks and generates answer via Gemini.
7. Q&A history is stored in SQLite (`database.db`).

## Project Structure

```text
Q-A_agent/
|-- admin-portal/
|-- customer-portal/
|-- backend/
|   |-- app/
|   |   |-- main.py
|   |   |-- auth.py
|   |   |-- db.py
|   |   |-- security.py
|   |   |-- schemas.py
|   |   `-- rag_runtime.py
|   |-- storage/pdfs/
|   `-- data/chroma/
|-- src/                  # existing RAG modules reused by backend
|-- tests/
|-- app.py                # legacy Streamlit UI (still usable)
|-- cli.py                # legacy CLI (still usable)
`-- database.db           # created at runtime
```

## Backend Features Implemented

- JWT auth (implemented manually with HMAC SHA256, no external auth provider).
- Password hashing with PBKDF2.
- Role-based access:
  - admin: upload docs, list users/docs, all chat history.
  - customer: ask questions, own chat history.
- SQLite tables:
  - `users`
  - `documents`
  - `chat_messages`
- RAG-backed endpoints:
  - `/api/chat/ask`
  - `/api/chat/history`
- Document ingestion endpoint:
  - `/api/documents/upload`

## API Endpoints

- `POST /api/auth/register` (first admin or customer)
- `POST /api/auth/login`
- `GET /api/auth/me`
- `GET /api/users` (admin)
- `POST /api/documents/upload` (admin)
- `GET /api/documents` (admin)
- `GET /api/documents/{id}/download` (admin)
- `POST /api/chat/ask` (admin/customer)
- `GET /api/chat/history` (admin/customer)
- `GET /api/rag/status` (admin/customer)
- `GET /api/health`

## Setup

### 1. Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Environment Variables

Create or update `.env` in the project root:

```env
GEMINI_API_KEY=your_key_here
JWT_SECRET_KEY=change_this_secret
DOCMIND_CORS_ORIGINS=http://localhost:5173,http://localhost:5174
```

Optional:

```env
DOCMIND_DB_PATH=database.db
DOCMIND_STORAGE_DIR=backend/storage/pdfs
CHROMA_DIR=backend/data/chroma
```

### 3. Run Backend

```powershell
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run Admin Portal

```powershell
cd admin-portal
npm install
npm run dev
```

Admin portal default URL: `http://localhost:5173`

### 5. Run Customer Portal

```powershell
cd customer-portal
npm install
npm run dev
```

Customer portal default URL: `http://localhost:5174`

## Auth Notes

- Admin registration is allowed only for the first admin account.
- After first admin exists, additional `role=admin` registrations are blocked.
- Customers can self-register using `role=customer`.

## Testing

Current tests still validate the reusable RAG modules:

```powershell
python -m pytest -q
```

## Legacy Interfaces (Still Available)

- Streamlit app: `streamlit run app.py`
- CLI:
  - `python cli.py ingest <files...>`
  - `python cli.py ask "question" --show-sources`
