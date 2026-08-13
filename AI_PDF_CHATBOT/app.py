"""
DocuMind AI - "Talk to your documents."

Multi-PDF RAG chatbot with document statistics, source citations, and
Gemini generation fallback, presented through a tab-based "knowledge
console" UI.

Architecture (unchanged):
pypdf extraction -> page-aware Document chunks -> RecursiveCharacterTextSplitter
-> GoogleGenerativeAIEmbeddings (models/gemini-embedding-001) -> FAISS
-> similarity_search() -> official google-genai SDK -> Gemini generation
-> answer with sources
"""

import time
import traceback
from dataclasses import dataclass, field

import streamlit as st
from dotenv import load_dotenv
import os

from pypdf import PdfReader

from google import genai
from google.genai import types as genai_types

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

APP_NAME = "DocuMind AI"
APP_TAGLINE = "Talk to your documents"

EMBEDDING_MODEL = "models/gemini-embedding-001"

# Generation fallback chain - both are real, currently available models
# for the google-genai SDK. Only one attempt per model, no infinite retries.
GENERATION_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 3
CHAT_HISTORY_WINDOW = 6
SUMMARY_MAX_CHUNKS = 40  # safety cap - never blindly send the whole document

SYSTEM_INSTRUCTION = (
    "You are a document question-answering assistant.\n"
    "Use ONLY the supplied retrieved context.\n"
    "If the answer is not present in the context, say that it could not be found.\n"
    "Never use outside knowledge.\n"
    "Never invent facts.\n"
    "You may use the recent conversation history only to resolve references "
    "such as pronouns (e.g. 'it', 'that') - never as a source of facts.\n"
    "Be concise and precise."
)

ACCENT_COLORS = ["#8b5cf6", "#06b6d4", "#f59e0b", "#ec4899", "#22c55e", "#3b82f6"]

# =========================================================
# STYLE - a distinct "knowledge console" look, Streamlit-native only
# =========================================================

CUSTOM_CSS = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');

.dm-hero {
    text-align: center;
    padding: 1.6rem 1rem 0.4rem 1rem;
}
.dm-hero-mark {
    display: inline-block;
    font-size: 1.9rem;
    padding: 0.35rem 0.7rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%);
    box-shadow: 0 6px 22px rgba(139, 92, 246, 0.35);
}
.dm-hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    margin-top: 0.5rem;
    letter-spacing: 0.01em;
}
.dm-hero-sub {
    opacity: 0.65;
    font-size: 0.95rem;
    margin-top: 0.1rem;
}
.dm-stat-strip {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    justify-content: center;
    margin: 1rem 0 0.4rem 0;
}
.dm-stat-chip {
    padding: 0.5rem 1rem;
    border-radius: 999px;
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.28);
    font-size: 0.85rem;
    font-weight: 600;
}
.dm-doc-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.7rem;
    border-radius: 10px;
    margin-bottom: 0.35rem;
    background: rgba(127,127,127,0.05);
    border-left: 4px solid var(--dm-accent, #8b5cf6);
}
.dm-doc-name {
    font-weight: 600;
    font-size: 0.9rem;
}
.dm-doc-meta {
    font-size: 0.78rem;
    opacity: 0.6;
}
.dm-suggestion-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.55;
    margin: 0.6rem 0 0.3rem 0;
}
.dm-evidence-card {
    border-left: 3px solid var(--dm-accent, #06b6d4);
    background: rgba(127,127,127,0.05);
    border-radius: 0 10px 10px 0;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.55rem;
}
.dm-evidence-tag {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 0.1rem 0.5rem;
    border-radius: 999px;
    background: var(--dm-accent, #06b6d4);
    color: #0b0e17;
    margin-bottom: 0.3rem;
}
.dm-model-badge {
    display: inline-block;
    font-size: 0.72rem;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.3);
    opacity: 0.85;
}
.dm-model-badge.backup {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.dm-empty-state {
    text-align: center;
    padding: 2.4rem 1.2rem;
    border-radius: 18px;
    background: rgba(139, 92, 246, 0.06);
    border: 1px dashed rgba(139, 92, 246, 0.3);
}
.dm-empty-icon {
    font-size: 2.2rem;
}
.dm-empty-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    margin-top: 0.3rem;
}
.dm-empty-sub {
    opacity: 0.6;
    font-size: 0.88rem;
    margin-top: 0.2rem;
}
.dm-process-stage {
    text-align: center;
    padding: 1.1rem 1rem;
    border-radius: 14px;
    background: rgba(139, 92, 246, 0.08);
    border: 1px solid rgba(139, 92, 246, 0.22);
    font-size: 1rem;
    animation: dm-pulse 1.6s ease-in-out infinite;
}
@keyframes dm-pulse {
    0% { opacity: 0.5; }
    50% { opacity: 1; }
    100% { opacity: 0.5; }
}
.dm-process-done {
    text-align: center;
    padding: 1.1rem 1rem;
    border-radius: 14px;
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(74, 222, 128, 0.3);
    font-size: 1rem;
}
</style>
'''


# =========================================================
# SESSION STATE
# =========================================================

def init_session_state():
    defaults = {
        "vector_store": None,
        "chat_history": [],
        "doc_stats": [],
        "all_chunks": [],
        "processed": False,
        "summary": None,
        "client": None,
        "developer_mode": False,
        "manual_api_key": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =========================================================
# API KEY / CLIENT
# =========================================================

def get_api_key():
    env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key
    return st.session_state.get("manual_api_key") or None


def get_genai_client(api_key: str) -> genai.Client:
    if st.session_state.client is None:
        st.session_state.client = genai.Client(api_key=api_key)
    return st.session_state.client


# =========================================================
# PDF EXTRACTION + CHUNKING (unchanged pipeline)
# =========================================================

@dataclass
class PdfExtractionResult:
    filename: str
    num_pages: int
    page_documents: list = field(default_factory=list)
    had_text: bool = True


def extract_text_from_pdfs(uploaded_files):
    """Extract page-level text with filename/page metadata for every uploaded PDF."""
    all_page_documents = []
    doc_stats = []
    skipped = []

    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(uploaded_file)
        except Exception as e:
            skipped.append((uploaded_file.name, f"could not be read ({e})"))
            continue

        page_docs = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                page_docs.append(
                    Document(page_content=text, metadata={"source": uploaded_file.name, "page": i + 1})
                )

        if not page_docs:
            skipped.append((uploaded_file.name, "no extractable text (likely scanned/image-only)"))
            continue

        all_page_documents.extend(page_docs)
        doc_stats.append({"filename": uploaded_file.name, "pages": len(reader.pages), "chunks": 0})

    return all_page_documents, doc_stats, skipped


def get_text_chunks(page_documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(page_documents)


def build_vector_store(chunks, api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    return FAISS.from_documents(chunks, embeddings)


# =========================================================
# TEXT GENERATION (with controlled fallback) - unchanged behavior
# =========================================================

def generate_with_fallback(client: genai.Client, system_instruction: str, contents):
    """Try each model in GENERATION_MODELS once, in order. Returns (text, model_used)."""
    last_error = None
    for idx, model_name in enumerate(GENERATION_MODELS):
        try:
            if idx > 0:
                st.toast("🟡 Primary model unavailable — trying backup model...")
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                ),
            )
            text = getattr(response, "text", None)
            if text and text.strip():
                if idx > 0:
                    st.toast("🟢 Backup model active")
                return text.strip(), model_name
            last_error = RuntimeError(f"{model_name} returned an empty response")
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All generation models failed. Last error: {last_error}")


# =========================================================
# RETRIEVAL + ANSWERING
# =========================================================

def retrieve_context(question: str, vector_store, k: int = RETRIEVAL_K):
    return vector_store.similarity_search(question, k=k)


def format_context(retrieved_docs) -> str:
    parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source {i}: {src} - Page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def format_recent_history() -> str:
    recent = st.session_state.chat_history[-CHAT_HISTORY_WINDOW:]
    if not recent:
        return ""
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def generate_answer(client: genai.Client, question: str, retrieved_docs):
    context = format_context(retrieved_docs)
    history_text = format_recent_history()
    prompt = (
        f"Conversation so far (for resolving references only):\n{history_text or '(none)'}\n\n"
        f"Document context (the ONLY source of facts you may use):\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the document context above."
    )
    return generate_with_fallback(client, SYSTEM_INSTRUCTION, prompt)


def process_question(question: str, api_key: str, status_placeholder=None):
    """Single shared entry point for questions: retrieve_context() -> generate_answer()."""
    if not api_key:
        st.error("Please provide an API key in the sidebar first.")
        return False

    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and process a PDF first.")
        return False

    client = get_genai_client(api_key)
    st.session_state.chat_history.append({"role": "user", "content": question})

    def set_stage(text):
        if status_placeholder is not None:
            status_placeholder.markdown(f"*{text}*")

    set_stage("🔎 Searching documents...")
    try:
        retrieved_docs = retrieve_context(question, st.session_state.vector_store)
    except Exception as e:
        if status_placeholder is not None:
            status_placeholder.empty()
        st.error("❌ I couldn't search the documents right now.")
        if st.session_state.developer_mode:
            with st.expander("Technical details"):
                st.code(str(e))
        return False

    set_stage("🧠 Generating answer...")
    try:
        answer, model_used = generate_answer(client, question, retrieved_docs)
    except Exception as e:
        if status_placeholder is not None:
            status_placeholder.empty()
        st.error("🟡 Gemini is busy right now and the backup model also failed. Please try again shortly.")
        if st.session_state.developer_mode:
            with st.expander("Technical details"):
                st.code(str(e))
        return False

    sources = [
        {"source": d.metadata.get("source", "Unknown"), "page": d.metadata.get("page", "?"), "text": d.page_content}
        for d in retrieved_docs
    ]
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources, "model_used": model_used}
    )
    if status_placeholder is not None:
        status_placeholder.empty()
    return True


# =========================================================
# SUMMARY (simple chunk-based map-reduce) - unchanged
# =========================================================

def generate_summary(client: genai.Client) -> str:
    chunks = st.session_state.all_chunks[:SUMMARY_MAX_CHUNKS]
    if not chunks:
        raise RuntimeError("No document content available to summarize.")

    batches, current_batch, current_len = [], [], 0
    for chunk in chunks:
        if current_len + len(chunk.page_content) > 8000 and current_batch:
            batches.append(current_batch)
            current_batch, current_len = [], 0
        current_batch.append(chunk)
        current_len += len(chunk.page_content)
    if current_batch:
        batches.append(current_batch)

    summary_instruction = (
        "You are a document summarization assistant. Summarize ONLY the text "
        "provided to you. Do not add outside knowledge or invent details."
    )
    partial_summaries = []
    for batch in batches:
        text = "\n\n".join(c.page_content for c in batch)
        prompt = f"Summarize the key points of this excerpt in 4-6 bullet points:\n\n{text}"
        partial_summaries.append(generate_with_fallback(client, summary_instruction, prompt)[0])

    combined = "\n\n".join(partial_summaries)
    final_prompt = (
        "Using ONLY the partial summaries below (derived from the uploaded documents), "
        "produce a final structured summary in this exact format:\n\n"
        "### Overview\n...\n\n### Key Concepts\n- ...\n- ...\n- ...\n\n"
        "### Important Points\n- ...\n- ...\n- ...\n\n### Conclusion\n...\n\n"
        f"Partial summaries:\n{combined}"
    )
    return generate_with_fallback(client, summary_instruction, final_prompt)[0]


# =========================================================
# PROCESSING PIPELINE (drives the real-stage animation)
# =========================================================

def render_processing_animation(placeholder, stage_text: str, done: bool = False):
    css_class = "dm-process-done" if done else "dm-process-stage"
    placeholder.markdown(f'<div class="{css_class}">{stage_text}</div>', unsafe_allow_html=True)


def process_uploaded_files(uploaded_files, api_key: str, placeholder):
    render_processing_animation(placeholder, "📄 Reading documents...")
    page_documents, doc_stats, skipped = extract_text_from_pdfs(uploaded_files)

    if not page_documents:
        return None, doc_stats, skipped

    render_processing_animation(placeholder, "🔍 Extracting text...")
    time.sleep(0.15)  # brief pause so the real stage is visible - not a fake progress bar

    render_processing_animation(placeholder, "✂️ Creating knowledge chunks...")
    chunks = get_text_chunks(page_documents)

    counts = {}
    for c in chunks:
        src = c.metadata.get("source", "Unknown")
        counts[src] = counts.get(src, 0) + 1
    for d in doc_stats:
        d["chunks"] = counts.get(d["filename"], 0)

    render_processing_animation(placeholder, "🧠 Generating embeddings...")
    render_processing_animation(placeholder, "🔗 Building FAISS knowledge network...")
    try:
        vector_store = build_vector_store(chunks, api_key)
    except Exception as e:
        raise RuntimeError(f"Embedding/FAISS build failed: {e}") from e

    st.session_state.all_chunks = chunks
    render_processing_animation(placeholder, "✨ Knowledge base ready", done=True)
    time.sleep(0.4)
    return vector_store, doc_stats, skipped


# =========================================================
# UI HELPERS
# =========================================================

def accent_for(index: int) -> str:
    return ACCENT_COLORS[index % len(ACCENT_COLORS)]


def render_stat_strip():
    total_docs = len(st.session_state.doc_stats)
    total_pages = sum(d["pages"] for d in st.session_state.doc_stats)
    total_chunks = sum(d["chunks"] for d in st.session_state.doc_stats)
    st.markdown(
        f'''
        <div class="dm-stat-strip">
            <div class="dm-stat-chip">📄 {total_docs} Documents</div>
            <div class="dm-stat-chip">📑 {total_pages} Pages</div>
            <div class="dm-stat-chip">🧩 {total_chunks} Chunks</div>
            <div class="dm-stat-chip">🟢 Ready</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_document_rows():
    for i, d in enumerate(st.session_state.doc_stats):
        color = accent_for(i)
        st.markdown(
            f'''
            <div class="dm-doc-row" style="--dm-accent: {color};">
                <div>📄</div>
                <div>
                    <div class="dm-doc-name">{d['filename']}</div>
                    <div class="dm-doc-meta">{d['pages']} pages · {d['chunks']} chunks</div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )


def render_evidence(sources: list):
    if not sources:
        return
    with st.expander(f"🔎 View retrieved context ({len(sources)} sources)"):
        for i, s in enumerate(sources, start=1):
            color = accent_for(i - 1)
            st.markdown(
                f'''
                <div class="dm-evidence-card" style="--dm-accent: {color};">
                    <div class="dm-evidence-tag">SOURCE {i:02d}</div>
                    <div style="font-size:0.85rem; opacity:0.75; margin-bottom:0.3rem;">📄 {s['source']} — Page {s['page']}</div>
                    <div style="font-size:0.88rem;">{s['text'][:400]}{'...' if len(s['text']) > 400 else ''}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )


def render_model_badge(model_used: str):
    is_backup = model_used != GENERATION_MODELS[0]
    css_class = "dm-model-badge backup" if is_backup else "dm-model-badge"
    label = "🧠 Backup model" if is_backup else "🧠 Gemini"
    st.markdown(f'<span class="{css_class}">{label}</span>', unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

def render_sidebar():
    st.sidebar.markdown(f"### 📚 {APP_NAME}")
    st.sidebar.caption(APP_TAGLINE)

    api_key = get_api_key()
    if not api_key:
        st.sidebar.warning("No API key found in environment.")
        manual_key = st.sidebar.text_input("Enter GOOGLE_API_KEY / GEMINI_API_KEY", type="password")
        if manual_key:
            st.session_state.manual_api_key = manual_key
            api_key = manual_key

    st.sidebar.divider()
    st.sidebar.subheader("Documents")
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    process_clicked = st.sidebar.button("⚡ Process Documents", type="primary", use_container_width=True)

    animation_placeholder = st.sidebar.empty()

    if process_clicked:
        if not api_key:
            st.sidebar.error("Please provide an API key before processing.")
        elif not uploaded_files:
            st.sidebar.error("❌ Please upload at least one PDF.")
        else:
            try:
                vector_store, doc_stats, skipped = process_uploaded_files(uploaded_files, api_key, animation_placeholder)
                if vector_store is None:
                    animation_placeholder.empty()
                    st.sidebar.error("❌ I couldn't process this document. Try uploading a text-based PDF.")
                else:
                    st.session_state.vector_store = vector_store
                    st.session_state.doc_stats = doc_stats
                    st.session_state.processed = True
                    st.session_state.summary = None
                    st.session_state.chat_history = []
                    if skipped:
                        for name, reason in skipped:
                            st.sidebar.warning(f"Skipped {name}: {reason}")
            except Exception as e:
                animation_placeholder.empty()
                st.sidebar.error("❌ Something went wrong while processing your documents.")
                if st.session_state.developer_mode:
                    with st.sidebar.expander("Technical details"):
                        st.code(f"{e}\n\n{traceback.format_exc()}")

    if st.session_state.processed and st.session_state.doc_stats:
        st.sidebar.divider()
        if st.sidebar.button("📋 Generate Summary", use_container_width=True):
            if not api_key:
                st.sidebar.error("Please provide an API key.")
            else:
                with st.sidebar.status("Generating summary...", expanded=False) as status:
                    try:
                        client = get_genai_client(api_key)
                        st.session_state.summary = generate_summary(client)
                        status.update(label="Summary ready", state="complete")
                    except Exception as e:
                        status.update(label="Summary generation failed", state="error")
                        st.sidebar.error("🟡 Couldn't generate a summary right now.")
                        if st.session_state.developer_mode:
                            with st.sidebar.expander("Technical details"):
                                st.code(str(e))

        st.sidebar.divider()
        col_a, col_b = st.sidebar.columns(2)
        if col_a.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        if col_b.button("↻ Clear all", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.doc_stats = []
            st.session_state.all_chunks = []
            st.session_state.processed = False
            st.session_state.summary = None
            st.rerun()

    st.sidebar.divider()
    st.session_state.developer_mode = st.sidebar.checkbox("🐞 Developer mode", value=st.session_state.developer_mode)

    return api_key


# =========================================================
# MAIN PANEL
# =========================================================

def render_hero():
    st.markdown(
        f'''
        <div class="dm-hero">
            <span class="dm-hero-mark">📚</span>
            <div class="dm-hero-title">{APP_NAME}</div>
            <div class="dm-hero-sub">{APP_TAGLINE}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_empty_state():
    st.markdown(
        '''
        <div class="dm-empty-state">
            <div class="dm-empty-icon">🗂️</div>
            <div class="dm-empty-title">No documents yet</div>
            <div class="dm-empty-sub">Upload one or more PDFs from the sidebar, then click "Process Documents" to build your knowledge base.</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_chat_tab(api_key):
    st.markdown('<div class="dm-suggestion-label">Suggested questions</div>', unsafe_allow_html=True)
    suggestions = [
        "What are the main topics discussed?",
        "What are the most important concepts?",
        "Explain the key definitions.",
        "Summarize the main conclusions.",
    ]
    suggested_click = None
    cols = st.columns(len(suggestions))
    for col, question in zip(cols, suggestions):
        if col.button(question, use_container_width=True):
            suggested_click = question

    st.divider()

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                render_evidence(msg.get("sources", []))
                render_model_badge(msg.get("model_used", GENERATION_MODELS[0]))

    typed_question = st.chat_input("Ask anything about your PDFs...")
    question_to_ask = suggested_click or typed_question

    if question_to_ask:
        status_placeholder = st.empty()
        if process_question(question_to_ask, api_key, status_placeholder=status_placeholder):
            st.rerun()


def render_dashboard_tab():
    render_stat_strip()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Your knowledge base**")
    render_document_rows()


def render_summary_tab(api_key):
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.info('No summary yet — click "📋 Generate Summary" in the sidebar to create one from your uploaded PDFs.')


def render_main(api_key):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_hero()

    if not st.session_state.processed:
        render_empty_state()
        return

    tab_chat, tab_dashboard, tab_summary = st.tabs(["💬 Chat", "📊 Dashboard", "📋 Summary"])
    with tab_chat:
        render_chat_tab(api_key)
    with tab_dashboard:
        render_dashboard_tab()
    with tab_summary:
        render_summary_tab(api_key)


# =========================================================
# MAIN
# =========================================================

def main():
    st.set_page_config(page_title=APP_NAME, page_icon="📚", layout="wide")
    init_session_state()
    api_key = render_sidebar()
    render_main(api_key)


if __name__ == "__main__":
    main()