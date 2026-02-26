# Mini Document Q&A Agent (Local RAG + Gemini)

A lightweight Retrieval-Augmented Generation (RAG) app that lets you upload PDF/TXT files, index them locally, and ask grounded questions in either Streamlit or CLI mode.

## Features

- Upload `.pdf` and `.txt` files
- Extract text from PDFs via `pdfplumber` with PyMuPDF fallback
- Chunk text with overlapping sliding windows (`500` tokens, `50` overlap by default)
- Generate embeddings locally with `sentence-transformers` (`all-MiniLM-L6-v2` by default)
- Persist vectors + metadata in local `ChromaDB` storage
- Retrieve top-k semantically similar chunks using cosine similarity
- Generate final answers with **Gemini API** using retrieved context
- Show source chunk references and retrieved snippets in the UI
- CLI mode for ingestion, one-shot asking, and interactive chat

## Project Structure

```text
Q-A_agent/
├── app.py
├── cli.py
├── pytest.ini
├── requirements.txt
├── README.md
├── data/
│   └── .gitkeep
├── tests/
│   ├── test_ingestion.py
│   ├── test_pipeline.py
│   └── test_vector_store.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── ingestion.py
    ├── embeddings.py
    ├── vector_store.py
    ├── retriever.py
    ├── prompts.py
    ├── llm.py
    └── rag_pipeline.py
```

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Add your Gemini API key.

- Option A: set environment variable

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

- Option B: create a `.env` file in the project root and fill values.

4. Run the app.

```powershell
streamlit run app.py
```

5. (Optional) Use CLI mode.

```powershell
python cli.py ingest .\docs\file1.pdf .\docs\file2.txt
python cli.py ask "What are the key points?" --show-sources
python cli.py chat --show-sources
```

6. Run tests.

```powershell
pytest
```

## How It Works

1. **Ingestion**: uploaded files are parsed into raw text.
2. **Chunking**: text is split into overlapping token chunks.
3. **Embedding**: chunks are embedded locally with sentence-transformers.
4. **Storage**: vectors + metadata are persisted in local ChromaDB (`data/chroma`).
5. **Retrieval**: user query is embedded and top-k similar chunks are retrieved.
6. **Generation**: retrieved chunks are passed to Gemini prompt for grounded answers.

## Configuration

Environment variables (optional):

- `GEMINI_API_KEY`
- `GEMINI_MAX_RETRIES` (default: `3`)
- `GEMINI_RETRY_WAIT_SECONDS` (default: `60`)
- `GEMINI_MODEL` (default: `gemini-2.5-flash`)
- `EMBEDDING_MODEL_NAME` (default: `all-MiniLM-L6-v2`)
- `CHUNK_SIZE` (default: `500`)
- `CHUNK_OVERLAP` (default: `50`)
- `TOP_K` (default: `5`)
- `RETRIEVAL_MIN_SIMILARITY` (default: `0.1`)
- `CHROMA_DIR` (default: `data/chroma`)
- `CHROMA_COLLECTION` (default: `document_chunks`)

## Notes

- Embeddings and retrieval are fully local.
- Only answer generation uses Gemini API (via `google-genai`).
- Re-uploading the same content updates existing chunks (deterministic chunk IDs).
- On Python 3.14+, `chromadb` can fail due upstream `pydantic.v1` compatibility.  
  The app automatically falls back to a local persistent store backend in that case.
- If you need native Chroma backend specifically, use Python 3.12 or 3.13.
- Scanned/image-only PDFs can produce zero chunks unless OCR text is embedded.
