# Q-A Agent (DocMind Platform)

DocMind is a robust Retrieval-Augmented Generation (RAG) platform structured as a two-portal product powered by a centralized FastAPI backend and Gemini LLM.

## Architecture & Portals

The platform is divided into three main components:

- **Admin Portal** (`admin-portal`): Built with React + TypeScript. Allows administrators to upload documents (PDF/TXT), manage user access, and view chat histories.
- **Customer Portal** (`customer-portal`): Built with React + TypeScript. Provides a customized interface for authenticated customers to ask questions via a Q&A chat interface and view their personal query history.
- **Backend Core** (`backend`): Built with FastAPI, SQLite, and JWT authentication. It manages users, securely stores documents, and exposes RAG (Retrieval-Augmented Generation) APIs.

The underlying RAG engine (located in `src/`) processes the documents, generates embeddings using ChromaDB, and retrieves context for accurate LLM answers.

## Key Features

- **Retrieval-Augmented Generation (RAG)**: Leverages Gemini to provide accurate answers based on the uploaded documents.
- **Secure Authentication**: Built-in JWT authentication with role-based access control (Admin vs. Customer). Password hashing is done via PBKDF2.
- **Document Management**: Admins can securely upload and manage PDF and TXT files. The system automatically chunks and indexes these documents.
- **Chat History**: Full conversation history is stored in an SQLite database, allowing users to review past interactions.

## Project Structure

```text
Q-A_agent/
├── admin-portal/         # React admin application
├── customer-portal/      # React customer application
├── backend/              # FastAPI application
│   ├── app/              # Core API routes, DB models, and schemas
│   ├── storage/pdfs/     # Uploaded documents storage
│   └── data/chroma/      # ChromaDB local vector store
├── src/                  # Reusable RAG pipeline (Ingestion, Embeddings)
├── tests/                # Automated testing suite
├── app.py                # Legacy Streamlit UI interface
├── cli.py                # Legacy Command Line Interface
└── database.db           # Auto-generated SQLite database
```

## Setup Instructions

### 1. Prerequisites

Ensure you have Python 3.9+ and Node.js installed.

### 2. Backend Setup

```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows: .\.venv\Scripts\Activate.ps1
# On Linux/Mac: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
JWT_SECRET_KEY=generate_a_secure_random_string
DOCMIND_CORS_ORIGINS=http://localhost:5173,http://localhost:5174
```

### 4. Running the Services

You need to run the Backend, Admin Portal, and Customer Portal simultaneously.

**Terminal 1: Backend**
```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2: Admin Portal**
```bash
cd admin-portal
npm install
npm run dev
# Accessible at http://localhost:5173
```

**Terminal 3: Customer Portal**
```bash
cd customer-portal
npm install
npm run dev
# Accessible at http://localhost:5174
```

## API Documentation

Once the backend is running, you can view the fully interactive API documentation at:
`http://localhost:8000/docs`

## Legacy Interfaces

The original interfaces are still available for basic usage:
- **Streamlit App**: `streamlit run app.py`
- **CLI Q&A**: `python cli.py ask "your question" --show-sources`
- **CLI Ingestion**: `python cli.py ingest document.pdf`
