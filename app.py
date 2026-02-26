from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, List, Literal, cast

import streamlit as st
from dotenv import load_dotenv

from src.config import Settings
from src.embeddings import LocalEmbeddingModel
from src.llm import GeminiAnswerGenerator
from src.rag_pipeline import RAGPipeline
from src.types import ChatMessage, SourceContext
from src.vector_store import ChromaVectorStore

# Load environment variables from .env if present.
load_dotenv()
settings = Settings()

st.set_page_config(
    page_title="DocMind AI – Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS: Dark theme, glassmorphism, gradients
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global overrides */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Hero header */
.hero-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    margin-bottom: 1rem;
}
.hero-header h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #10B981, #34D399, #A3E635);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.35rem;
    letter-spacing: -0.5px;
}
.hero-header p {
    color: #A1AEBB;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
}

/* Section headers */
.section-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #E8F5E9;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Upload area styling */
.upload-zone {
    border: 2px dashed rgba(16, 185, 129, 0.45);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    background: rgba(16, 185, 129, 0.06);
    transition: all 0.3s ease;
    margin-bottom: 1rem;
}
.upload-zone:hover {
    border-color: rgba(52, 211, 153, 0.75);
    background: rgba(16, 185, 129, 0.12);
}

/* Status badges */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 500;
}
.status-ready {
    background: rgba(34, 197, 94, 0.18);
    color: #4ADE80;
    border: 1px solid rgba(34, 197, 94, 0.35);
}
.status-empty {
    background: rgba(234, 179, 8, 0.15);
    color: #FACC15;
    border: 1px solid rgba(234, 179, 8, 0.3);
}
.status-error {
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Similarity score badges */
.sim-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Inter', monospace;
}
.sim-high {
    background: rgba(34, 197, 94, 0.2);
    color: #4ADE80;
    border: 1px solid rgba(34, 197, 94, 0.35);
}
.sim-medium {
    background: rgba(234, 179, 8, 0.2);
    color: #FACC15;
    border: 1px solid rgba(234, 179, 8, 0.35);
}
.sim-low {
    background: rgba(239, 68, 68, 0.2);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

/* Context snippet cards */
.context-card {
    background: rgba(20, 40, 30, 0.55);
    border: 1px solid rgba(52, 211, 153, 0.2);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    backdrop-filter: blur(8px);
}
.context-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #A7F3D0;
}

/* Sidebar glass effect — radiant green */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(6, 78, 59, 0.92) 0%, rgba(4, 47, 35, 0.95) 100%) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(52, 211, 153, 0.25) !important;
}

/* Make ALL sidebar text bright and visible */
section[data-testid="stSidebar"] * {
    color: #D1FAE5 !important;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    background: linear-gradient(135deg, #34D399, #A3E635) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-size: 1.3rem;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown span {
    color: #D1FAE5 !important;
}
section[data-testid="stSidebar"] .stMarkdown code {
    color: #A3E635 !important;
    background: rgba(163, 230, 53, 0.1) !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSlider label {
    color: #A7F3D0 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(52, 211, 153, 0.25) !important;
}

/* Sidebar metrics */
.sidebar-metric {
    background: rgba(6, 95, 70, 0.45);
    border: 1px solid rgba(52, 211, 153, 0.2);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.sidebar-metric-label {
    color: #6EE7B7 !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.2rem;
}
.sidebar-metric-value {
    color: #ECFDF5 !important;
    font-size: 1.1rem;
    font-weight: 600;
}

/* Ingest results */
.ingest-result {
    background: rgba(20, 40, 30, 0.4);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin: 0.3rem 0;
    border-left: 3px solid #10B981;
}

/* Button styling */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
}

/* Chat input styling */
.stChatInput > div {
    border-radius: 14px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────── Hero Header ────────────────────
st.markdown(
    """
<div class="hero-header">
    <h1>🧠 DocMind AI</h1>
    <p>Upload documents, ask questions, get precise AI-powered answers grounded in your data</p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str) -> LocalEmbeddingModel:
    """Load the local embedding model once and reuse it across reruns."""
    return LocalEmbeddingModel(model_name)


@st.cache_resource(show_spinner=False)
def get_vector_store(chroma_dir: str, collection_name: str) -> ChromaVectorStore:
    """Open a persistent local vector store so vectors survive restarts."""
    return ChromaVectorStore(Path(chroma_dir), collection_name)


def get_messages() -> List[ChatMessage]:
    """Read or initialize chat transcript from session state."""
    raw_messages = st.session_state.get("messages")
    if not isinstance(raw_messages, list):
        st.session_state["messages"] = []
    return cast(List[ChatMessage], st.session_state["messages"])


def get_last_ingest_issue() -> str:
    """Read or initialize ingest issue state."""
    raw_issue = st.session_state.get("last_ingest_issue")
    if not isinstance(raw_issue, str):
        st.session_state["last_ingest_issue"] = ""
    return cast(str, st.session_state["last_ingest_issue"])


def set_last_ingest_issue(value: str) -> None:
    """Persist latest ingestion issue message."""
    st.session_state["last_ingest_issue"] = value


def _similarity_badge(distance: object) -> str:
    """Generate an HTML similarity badge based on distance score."""
    if not isinstance(distance, (int, float)):
        return ""
    similarity = 1 - float(distance)
    pct = f"{similarity * 100:.1f}%"
    if similarity >= 0.7:
        return f'<span class="sim-badge sim-high">✦ {pct}</span>'
    elif similarity >= 0.4:
        return f'<span class="sim-badge sim-medium">● {pct}</span>'
    else:
        return f'<span class="sim-badge sim-low">○ {pct}</span>'


def render_context_snippets(contexts: List[SourceContext]) -> None:
    """Show retrieved snippets used to ground the answer."""
    for idx, item in enumerate(contexts, start=1):
        distance = item.get("distance")
        sim_badge = _similarity_badge(distance)

        st.markdown(
            f'<div class="context-card">'
            f'<div class="context-card-header">'
            f"<span><strong>[{idx}]</strong> 📄 <code>{item['source']}</code> · chunk <code>{item['chunk_index']}</code></span>"
            f"{sim_badge}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        snippet = item["text"]
        if len(snippet) > 900:
            snippet = snippet[:900] + "..."
        st.code(snippet, language="text")


def process_ingestion(
    uploaded_files: List[Any],
    rag_pipeline: RAGPipeline,
    stored_chunks_placeholder: Any,
    vector_store: ChromaVectorStore,
) -> None:
    """Ingest uploaded files and render results."""
    total_chunks = 0
    failed_files: List[str] = []
    zero_chunk_files: List[str] = []
    successful_files: List[str] = []

    with st.spinner("⏳ Parsing, chunking, embedding, and storing..."):
        for uploaded in uploaded_files:
            suffix = Path(str(uploaded.name)).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded.getbuffer())
                tmp_path = Path(tmp_file.name)

            try:
                chunk_count = rag_pipeline.ingest_file(tmp_path, source_name=str(uploaded.name))
                if chunk_count > 0:
                    total_chunks += chunk_count
                    successful_files.append(f"{uploaded.name}: {chunk_count} chunks")
                else:
                    zero_chunk_files.append(str(uploaded.name))
            except Exception as exc:
                failed_files.append(f"{uploaded.name}: {exc}")
            finally:
                tmp_path.unlink(missing_ok=True)

    current_count = vector_store.count()
    stored_chunks_placeholder.markdown(
        f'<div class="sidebar-metric">'
        f'<div class="sidebar-metric-label">Stored Chunks</div>'
        f'<div class="sidebar-metric-value">{current_count}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    if total_chunks > 0:
        set_last_ingest_issue("")
        st.success(f"✅ Ingestion complete — **{total_chunks}** chunks stored")
    else:
        issue_parts: List[str] = []
        if zero_chunk_files:
            issue_parts.append(
                "No extractable text found in: "
                + ", ".join(zero_chunk_files)
                + ". If these are scanned PDFs, run OCR first or try a text-based PDF."
            )
        if failed_files:
            issue_parts.append(" ; ".join(failed_files))

        set_last_ingest_issue(" | ".join(issue_parts) if issue_parts else "No chunks were stored from uploaded files.")

    if successful_files:
        for item in successful_files:
            st.markdown(f'<div class="ingest-result">📄 {item}</div>', unsafe_allow_html=True)

    if failed_files:
        st.error("Some files failed during ingestion:")
        for item in failed_files:
            st.write(f"- {item}")


# ──────────────────── Initialize ────────────────────
try:
    embedding_model = get_embedding_model(settings.embedding_model_name)
    vector_store = get_vector_store(str(settings.chroma_dir), settings.collection_name)
except Exception as exc:
    st.error(f"Failed to initialize embedding model or vector store: {exc}")
    st.stop()

messages = get_messages()
_ = get_last_ingest_issue()

# ──────────────────── Sidebar ────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_api_key:
        st.markdown(
            '<span class="status-badge status-ready">🔑 API Key Active</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-error">🔑 API Key Missing</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    top_k = st.slider("🔍 Top-k retrieval", min_value=1, max_value=10, value=settings.top_k)

    stored_chunks_placeholder = st.empty()
    chunk_count = vector_store.count()
    stored_chunks_placeholder.markdown(
        f'<div class="sidebar-metric">'
        f'<div class="sidebar-metric-label">Stored Chunks</div>'
        f'<div class="sidebar-metric-value">{chunk_count}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="sidebar-metric">'
        f'<div class="sidebar-metric-label">Vector Backend</div>'
        f'<div class="sidebar-metric-value">{vector_store.backend_name}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="sidebar-metric">'
        f'<div class="sidebar-metric-label">Min Similarity</div>'
        f'<div class="sidebar-metric-value">{settings.retrieval_min_similarity:.0%}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    indexed_sources = vector_store.list_sources(limit=8)
    if indexed_sources:
        st.markdown("---")
        st.markdown("**📚 Indexed Sources**")
        for src in indexed_sources:
            st.markdown(f"- `{src}`")

    if vector_store.backend_name == "fallback" and vector_store.backend_error:
        st.warning("Chroma could not be loaded. Using local fallback store.")

    st.markdown("---")
    if st.button("🗑️ Clear Vector Store", use_container_width=True):
        vector_store.clear()
        st.session_state["messages"] = []
        set_last_ingest_issue("")
        st.success("Vector store cleared.")

# ──────────────────── Upload Section ────────────────────
st.markdown('<div class="section-header">📤 Upload & Ingest Documents</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="upload-zone">',
    unsafe_allow_html=True,
)
uploaded_files_value = st.file_uploader(
    "Drop your PDF or TXT files here",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)
uploaded_files = cast(List[Any], uploaded_files_value or [])

if st.button("⚡ Ingest Documents", disabled=not uploaded_files, use_container_width=True):
    rag_pipeline = RAGPipeline(settings=settings, embedding_model=embedding_model, vector_store=vector_store)
    process_ingestion(uploaded_files, rag_pipeline, stored_chunks_placeholder, vector_store)

# Status badge
indexed_chunks = vector_store.count()
if indexed_chunks > 0:
    st.markdown(
        f'<span class="status-badge status-ready">✅ Ready — {indexed_chunks} chunks indexed</span>',
        unsafe_allow_html=True,
    )
elif get_last_ingest_issue():
    st.markdown(
        '<span class="status-badge status-error">⚠️ Ingestion issue</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="status-badge status-empty">📭 No documents ingested yet</span>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ──────────────────── Chat Section ────────────────────
st.markdown('<div class="section-header">💬 Ask Questions</div>', unsafe_allow_html=True)

for message in messages:
    raw_role = message.get("role", "assistant")
    role: Literal["user", "assistant"]
    if raw_role in {"user", "assistant"}:
        role = cast(Literal["user", "assistant"], raw_role)
    else:
        role = "assistant"
    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🧠"):
        st.markdown(message.get("content", ""))
        sources = message.get("sources")
        if role == "assistant" and isinstance(sources, list) and sources:
            with st.expander("📎 Retrieved context snippets", expanded=False):
                render_context_snippets(cast(List[SourceContext], sources))

current_chunk_count = vector_store.count()
question = st.chat_input("Ask a question about your documents...")
if question:
    user_message: ChatMessage = {"role": "user", "content": question}
    messages.append(user_message)
    st.session_state["messages"] = messages

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🧠"):
        if current_chunk_count == 0:
            answer = "📭 No ingested text is available yet. Upload and ingest a text-based PDF or TXT file first."
            contexts: List[SourceContext] = []
        elif not gemini_api_key:
            answer = "🔑 GEMINI_API_KEY is not configured. Set it in your environment or .env file."
            contexts = []
        else:
            try:
                answer_generator = GeminiAnswerGenerator(
                    api_key=gemini_api_key,
                    model_name=settings.gemini_model_name,
                    max_retries=settings.gemini_max_retries,
                    retry_wait_seconds=settings.gemini_retry_wait_seconds,
                    temperature=settings.gemini_temperature,
                )
                rag_pipeline = RAGPipeline(
                    settings=settings,
                    embedding_model=embedding_model,
                    vector_store=vector_store,
                    answer_generator=answer_generator,
                )

                with st.spinner("🔍 Retrieving context and generating answer..."):
                    answer, contexts = rag_pipeline.answer_question(question, top_k=top_k)
            except Exception as exc:
                answer = f"❌ Failed to generate answer: {exc}"
                contexts = []

        st.markdown(answer)
        if contexts:
            with st.expander("📎 Retrieved context snippets", expanded=False):
                render_context_snippets(contexts)

    assistant_message: ChatMessage = {
        "role": "assistant",
        "content": answer,
        "sources": contexts,
    }
    messages.append(assistant_message)
    st.session_state["messages"] = messages
