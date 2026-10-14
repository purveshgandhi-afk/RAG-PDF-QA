import os
import re
import uuid
import html
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from backend import setup_qa_system

load_dotenv()

# -- Config --------------------------------------------------------------------
MAX_FILE_SIZE_MB = 50

st.set_page_config(
    page_title="PDF Q&A with Gemini",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Styles --------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    color: #e2e8f0 !important;
    transition: all 0.2s;
    width: 100%;
    margin-bottom: 6px;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.12);
    border-color: rgba(255,255,255,0.2);
}

.main .block-container { padding-top: 1.5rem; max-width: 900px; }
.stChatMessage { border-radius: 12px; margin-bottom: 0.5rem; }

.source-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.5;
}
.source-card .page-badge {
    display: inline-block;
    background: #6366f1;
    color: white !important;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 6px;
}

.app-header { text-align: center; padding: 1.2rem 0 0.5rem; }
.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.app-header p { color: #64748b; font-size: 0.95rem; margin: 0; }
</style>
""", unsafe_allow_html=True)


# -- Helpers -------------------------------------------------------------------

def sanitize_filename(filename: str) -> str:
    filename = filename.replace("\\", "").replace("/", "")
    return re.sub(r"[^a-zA-Z0-9._-]", "_", filename)


def validate_pdf(uploaded_file) -> tuple:
    """Returns (is_valid, error_message)."""
    if not uploaded_file.name.lower().endswith(".pdf"):
        return False, "Only PDF files are allowed."
    file_bytes = uploaded_file.getvalue()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB."
    if not file_bytes.startswith(b"%PDF"):
        return False, "Invalid PDF: magic bytes check failed."
    return True, ""


def format_sources(source_docs: list) -> list:
    seen = set()
    sources = []
    for doc in source_docs:
        page = doc.metadata.get("page", None)
        page_num = (page + 1) if page is not None else "?"
        key = (page_num, doc.page_content[:80])
        if key not in seen:
            seen.add(key)
            sources.append({
                "page": page_num,
                "excerpt": doc.page_content.strip()[:300],
            })
    return sorted(sources, key=lambda x: (x["page"] if isinstance(x["page"], int) else 999))


# -- Session state init --------------------------------------------------------
defaults = {
    "qa_chain": None,
    "current_pdf_name": None,   # track by name only -- no file path needed
    "session_id": str(uuid.uuid4()),
    "messages": [],             # display history: [{role, content, sources}]
    "chat_history": [],         # LangChain memory: [HumanMessage, AIMessage, ...]
    "chunk_count": 0,
    "uploader_key": 0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📄 PDF Q&A with Gemini")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Max 50 MB. Text-based PDFs only.",
        key=f"pdf_uploader_{st.session_state.uploader_key}",
    )

    if uploaded_file:
        is_valid, err = validate_pdf(uploaded_file)
        if not is_valid:
            st.error(f"❌ {err}")
        else:
            safe_name = sanitize_filename(uploaded_file.name)

            # Only reprocess if it is a different file than what is already loaded
            if st.session_state.current_pdf_name != safe_name:
                with st.spinner("Indexing PDF... (first run downloads embedding model ~90 MB)"):
                    try:
                        chain, chunk_count = setup_qa_system(
                            pdf_bytes=uploaded_file.getvalue(),
                            session_id=st.session_state.session_id,
                        )
                        st.session_state.qa_chain = chain
                        st.session_state.current_pdf_name = safe_name
                        st.session_state.chunk_count = chunk_count
                        # Clear chat when a new PDF is loaded
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.success(f"✅ Ready! Indexed {chunk_count} chunks.")
                    except FileNotFoundError as e:
                        st.error(f"❌ File error: {e}")
                    except ValueError as e:
                        st.error(f"❌ PDF error: {e}")
                    except RuntimeError as e:
                        st.error(f"❌ Config error: {e}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")
            else:
                st.info(f"ℹ️ {safe_name} is already loaded.")

    # Current document status
    if st.session_state.current_pdf_name:
        st.markdown("**Current document:**")
        st.markdown(
            f"<div style='background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);"
            f"border-radius:8px;padding:10px 14px;font-size:0.82rem;color:#4338ca;word-break:break-all;'>"
            f"📄 {html.escape(st.session_state.current_pdf_name)}<br>"
            f"<span style='opacity:0.7'>{st.session_state.chunk_count} chunks indexed</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")

    # Session controls
    st.markdown("**Session controls:**")

    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    if st.button("📂 Load New PDF"):
        st.session_state.qa_chain = None
        st.session_state.current_pdf_name = None
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.chunk_count = 0
        st.session_state.uploader_key += 1
        st.rerun()

    st.markdown("---")
    active_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    st.markdown(
        f"<div style='font-size:0.75rem;color:#94a3b8;text-align:center'>"
        f"Embeddings: local all-MiniLM-L6-v2<br>LLM: {active_model}"
        f"</div>",
        unsafe_allow_html=True,
    )


# -- Main chat area ------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>📄 PDF Question Answering</h1>
    <p>Upload a PDF in the sidebar, then ask anything about its content.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.qa_chain:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem;background:#f8fafc;
        border:2px dashed #cbd5e1;border-radius:16px;color:#94a3b8;">
            <div style="font-size:3rem;margin-bottom:1rem">📤</div>
            <div style="font-size:1.1rem;font-weight:600;color:#64748b;margin-bottom:0.5rem">
                No PDF loaded yet
            </div>
            <div style="font-size:0.9rem">
                Upload a PDF using the sidebar to get started.<br>
                Supports text-based PDFs up to 50 MB.
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    # Chat history display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])} passages)", expanded=False):
                    for src in msg["sources"]:
                        page_label = f"Page {src['page']}" if isinstance(src["page"], int) else "Unknown page"
                        st.markdown(
                            f'<div class="source-card">'
                            f'<div class="page-badge">{page_label}</div><br>'
                            f'{html.escape(src["excerpt"])}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Chat input
    if prompt := st.chat_input("Ask a question about the PDF..."):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history,
                    })
                    answer = result.get("answer", "No answer returned.")
                    sources = format_sources(result.get("context", []))

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📚 Sources ({len(sources)} passages)", expanded=False):
                            for src in sources:
                                page_label = f"Page {src['page']}" if isinstance(src["page"], int) else "Unknown page"
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<div class="page-badge">{page_label}</div><br>'
                                    f'{html.escape(src["excerpt"])}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    err_msg = f"❌ Error: {e}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "sources": [],
                    })
