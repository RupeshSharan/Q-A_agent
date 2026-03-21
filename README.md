<div align="center">

<img src="https://img.shields.io/badge/DocMind-AI%20Platform-00ff88?style=for-the-badge&logo=openai&logoColor=white" alt="DocMind AI"/>

# 🧠 DocMind AI

## Document-Powered Customer Support & Booking Platform

**Ask anything. Book instantly. Answers grounded in your actual documents.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Local%20Vector%20DB-FF6B35?style=flat-square)](https://www.trychroma.com)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## What Is DocMind AI?

DocMind AI is a full-stack **Retrieval-Augmented Generation (RAG)** platform that turns your company documents into an intelligent, always-available support agent.

Upload your PDFs. Customers ask questions. The AI answers instantly, accurately, and only from what your documents actually say. No hallucinations. No hardcoded scripts. No call center queues.

> **Real example**: A hospital uploads their service brochure. A patient asks *"What is the cost of a general consultation?"* — DocMind finds the exact answer from the document and responds in seconds.

---

## Platform Overview

DocMind is a **three-component monorepo** — two React frontends sharing one intelligent FastAPI backend.

```text
                        ┌─────────────────────────────┐
                        │        FastAPI Backend        │
                        │                               │
   ┌──────────────┐     │  ┌─────────┐  ┌──────────┐  │     ┌──────────────────┐
   │ Admin Portal │────▶│  │   RAG   │  │   Auth   │  │◀────│ Customer Portal  │
   │  (React/TS)  │     │  │Pipeline │  │  (JWT)   │  │     │   (React/TS)     │
   └──────────────┘     │  └────┬────┘  └──────────┘  │     └──────────────────┘
                        └───────┼───────────────────────┘
                                │
                   ┌────────────┴────────────┐
                   │                         │
            ┌───────▼───────┐         ┌────────▼───────┐
            │   ChromaDB    │         │  SQLite / PG   │
            │(Vector Store) │         │  (Users, Docs, │
            │   + Chunks)   │         │   Bookings)    │
            └───────────────┘         └────────────────┘
```

---

## Core Features

### Intelligent RAG Pipeline
- Sentence-aware chunking preserves meaning across splits
- Local embeddings via `sentence-transformers` — no data leaves your server
- Dual-pass retrieval combining semantic search and keyword matching for higher recall
- Automatic typo correction in queries before retrieval
- Grounded answers with source citations — the AI never makes things up

### Admin Portal
- Secure login with JWT authentication
- Upload and manage PDF and TXT documents
- View all customer chat histories and queries
- Manage bookings — confirm or cancel appointments
- Full document lifecycle control including soft delete and re-ingestion

### Customer Portal
- No login required to ask questions — instant access for anyone
- Intelligent chat interface with real-time AI responses
- Optional registration for booking management and history tracking
- Guest booking with just name, email, and phone — no password needed
- View personal booking history when registered

### Booking System
- Admin sets available time slots
- Customers pick a slot and submit a booking with an optional message
- Bookings appear instantly on the admin dashboard
- Email notifications via Gmail SMTP (free)
- WhatsApp notifications via Meta Business API (1000 free messages per month)

---

## Project Structure

```text
Q-A_agent/
│
├── admin-portal/              # React + TypeScript (Admin Interface)
│   ├── src/
│   │   ├── pages/             # Dashboard, Documents, Bookings, Analytics
│   │   ├── components/        # Reusable UI components
│   │   └── api/               # Typed API client
│   └── package.json
│
├── customer-portal/           # React + TypeScript (Customer Interface)
│   ├── src/
│   │   ├── pages/             # Chat, Booking, Login, My Bookings
│   │   ├── components/        # Chat window, booking form, etc.
│   │   └── api/               # Typed API client
│   └── package.json
│
├── backend/                   # FastAPI Application
│   ├── app/
│   │   ├── routes/            # auth, documents, chat, bookings
│   │   ├── models/            # SQLAlchemy DB models
│   │   └── schemas/           # Pydantic request/response schemas
│   ├── storage/pdfs/          # Uploaded PDF files (persisted on disk)
│   └── data/chroma/           # ChromaDB local vector store
│
├── src/                       # Reusable RAG Engine
│   ├── ingestion.py           # PDF/TXT parsing + sentence-aware chunking
│   ├── embeddings.py          # Local sentence-transformer wrapper
│   ├── vector_store.py        # ChromaDB interface
│   ├── retriever.py           # Dual-pass retrieval + query expansion
│   ├── llm.py                 # LLM client (Gemini / Groq / Ollama)
│   ├── rag_pipeline.py        # End-to-end RAG orchestration
│   └── prompts.py             # Chain-of-thought prompt templates
│
├── tests/                     # Automated Test Suite
├── app.py                     # Legacy Streamlit UI
├── cli.py                     # Legacy CLI Interface
├── database.db                # Auto-generated SQLite database
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) *(or use Ollama or Groq — see LLM Options below)*

---

### Step 1 — Clone and Set Up Python Environment

```bash
git clone https://github.com/RupeshSharan/Q-A_agent.git
cd Q-A_agent

python -m venv .venv

# Activate on Windows
.\.venv\Scripts\Activate.ps1

# Activate on Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

---

### Step 2 — Configure Environment Variables

Create a `.env` file in the project root:

```env
# LLM Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Auth (generate a long random string)
JWT_SECRET_KEY=your_long_random_secret_key_here

# CORS for frontend dev servers
DOCMIND_CORS_ORIGINS=http://localhost:5173,http://localhost:5174
```

> No Gemini key? See LLM Options below for free unlimited alternatives including Ollama and Groq.

---

### Step 3 — Start All Three Services

Open three terminal windows and run one command in each.

**Terminal 1 — Backend**
```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Admin Portal**
```bash
cd admin-portal
npm install
npm run dev
```

**Terminal 3 — Customer Portal**
```bash
cd customer-portal
npm install
npm run dev
```

---

### Step 4 — Open the Portals

| Interface | URL | Who Uses It |
|---|---|---|
| Admin Portal | http://localhost:5173 | Company staff |
| Customer Portal | http://localhost:5174 | End customers |
| API Docs (Swagger) | http://localhost:8000/docs | Developers |
| API Docs (ReDoc) | http://localhost:8000/redoc | Developers |

---

## LLM Options

DocMind is LLM-agnostic. Switch providers by changing environment variables — no code changes needed.

### Option A — Ollama (Best for Development)

Run powerful models locally with zero API limits and zero cost.

```bash
# Install Ollama from https://ollama.com
# Pull a model once (downloads the model file)
ollama pull llama3.2

# Add to your .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

No internet required after the initial download. Perfect for heavy testing and development.

### Option B — Groq (Best for Production)

Free cloud API with extremely generous daily limits.

```bash
# Sign up at https://groq.com and get a free API key

LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

### Option C — Google Gemini

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

| Provider | Daily Limit | Needs Internet | Cost | Best For |
|---|---|---|---|---|
| Ollama | Unlimited | No | Free | Development |
| Groq | 14,400 requests | Yes | Free | Production |
| Gemini Flash | 1,500 requests | Yes | Free | Quick testing |

---

## Configuration Reference

All settings are controlled via environment variables in your `.env` file.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | Options: `gemini`, `groq`, `ollama` |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `GROQ_API_KEY` | — | Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `JWT_SECRET_KEY` | — | Required. Long random string for token signing |
| `JWT_EXPIRE_MINUTES` | `60` | How long login tokens stay valid |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHUNK_SIZE` | `400` | Max tokens per document chunk |
| `CHUNK_OVERLAP` | `100` | Overlap tokens between adjacent chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `RETRIEVAL_MIN_SIMILARITY` | `0.25` | Minimum similarity score to include a chunk |
| `CHROMA_DIR` | `backend/data/chroma` | ChromaDB storage directory |
| `STORAGE_DIR` | `backend/storage/pdfs` | Uploaded PDF storage directory |
| `DOCMIND_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed frontend origins |

---

## Database Schema

DocMind uses SQLite for development and PostgreSQL for production. Switching is a single environment variable change — SQLAlchemy handles both identically.

| Table | Purpose |
|---|---|
| `admins` | Admin and staff accounts with role-based access |
| `customers` | Optional customer accounts (guest sessions supported without this) |
| `documents` | PDF metadata — actual files are stored on disk not in the DB |
| `chat_sessions` | Conversation sessions tracked by token for guests or user ID for accounts |
| `chat_messages` | Individual question and answer pairs with document source references |
| `bookings` | Appointment requests from customers including guest bookings |
| `booking_slots` | Available time slots configured by admin |
| `notifications` | Log of all emails and WhatsApp messages sent with delivery status |

---

## Key API Endpoints

```text
# Auth
POST   /auth/login                    Login as admin or customer
POST   /auth/register                 Register a new customer account

# Documents (admin only)
POST   /admin/documents/upload        Upload a PDF or TXT document
GET    /admin/documents               List all uploaded documents
DELETE /admin/documents/{id}          Remove a document

# Chat (open to everyone)
POST   /chat/ask                      Ask a question — no auth required
GET    /chat/history                  Retrieve session chat history

# Bookings
POST   /bookings                      Create a booking as guest or registered user
GET    /admin/bookings                View all bookings (admin only)
PATCH  /admin/bookings/{id}           Confirm or cancel a booking (admin only)

# Slots (admin only)
GET    /admin/slots                   View available booking slots
POST   /admin/slots                   Create new available time slots
```

Full interactive documentation is available at **http://localhost:8000/docs** once the backend is running.

---

## Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run individual test files
pytest tests/test_ingestion.py -v      # Document chunking and parsing
pytest tests/test_pipeline.py -v       # Full RAG pipeline with fakes
pytest tests/test_vector_store.py -v   # Vector similarity search ordering
```

---

## Deployment — Zero Cost Stack

| Component | Service | Cost |
|---|---|---|
| Admin Portal | Vercel | Free |
| Customer Portal | Vercel | Free |
| Backend | Render or Railway | Free tier |
| Vector DB | ChromaDB on server disk | Free |
| PDF Storage | Server disk | Free |
| User Database | SQLite in dev, Neon PostgreSQL in prod | Free |
| LLM | Groq API | Free up to 14,400 requests per day |
| Email | Gmail SMTP | Free |
| WhatsApp | Meta Business API | Free up to 1,000 messages per month |
| **Total** | | **$0** |

> Free hosting on Render and Railway sleeps after 15 minutes of inactivity. First request after sleep takes around 30 seconds to wake. This is acceptable for a portfolio project. Upgrade when you get real traffic.

---

## Legacy Interfaces

The original interfaces remain available for quick testing and scripting.

```bash
# Launch the Streamlit web app
streamlit run app.py

# Ask a single question via CLI
python cli.py ask "What are your office hours?" --show-sources

# Ingest a document via CLI
python cli.py ingest docs/services.pdf

# Start an interactive CLI chat session
python cli.py chat --show-sources

# View vector store statistics
python cli.py stats

# Clear all indexed data
python cli.py clear
```

---

## Security

- Passwords hashed with PBKDF2 — never stored in plain text
- All admin routes protected by JWT Bearer token verification
- Uploaded PDFs stored server-side — customers never receive direct file access or URLs
- Embeddings generated locally — document content never sent to external APIs during indexing
- Only the final answer generation step uses an external LLM API

---

## Roadmap

- Staff module with role-based interface inside the admin portal
- Document update suggestion workflow for staff
- Organisation improvement suggestion system with admin approval
- WhatsApp booking confirmation and reminder integration
- Analytics dashboard showing query trends and popular topics
- Multi-tenant support for serving multiple companies
- Streaming responses in the chat interface

---

## Tech Stack

| Layer | Technology |
|---|---|
| Admin Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| Customer Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| Backend | FastAPI, Python 3.9+, Uvicorn |
| Authentication | JWT via python-jose, PBKDF2 password hashing |
| Database | SQLite (dev) to PostgreSQL (prod) via SQLAlchemy + Alembic |
| Vector Store | ChromaDB, local and persistent |
| Embeddings | sentence-transformers, all-MiniLM-L6-v2 |
| LLM | Gemini, Groq, or Ollama — swappable via config |
| PDF Parsing | pdfplumber with PyMuPDF fallback |
| File Storage | Local filesystem |

---

## License

This project is for educational and portfolio use. MIT License.

---

<div align="center">

Built by Rupesh

*Turn your documents into your smartest team member.*

</div>