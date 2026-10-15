"""
backend.py -- RAG pipeline for PDF Question Answering

Components:
  - PDF loading         : PyPDFLoader via a short-lived temp file (deleted after chunking)
  - Chunking            : RecursiveCharacterTextSplitter (sentence-aware)
  - Embeddings          : HuggingFace all-MiniLM-L6-v2 (local, free, no API cost)
  - Vector store        : FAISS in-memory -- nothing written to disk
  - Retrieval           : MMR (Maximum Marginal Relevance) for diverse results
  - Conversation memory : create_history_aware_retriever (reformulates follow-ups)
  - LLM                 : Groq (llama-3.3-70b-versatile, free tier, ultra-fast)
  - Chain               : create_retrieval_chain (returns answer + source docs)
"""

import os
import tempfile
import functools

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
try:
    from langchain.chains import (
        create_retrieval_chain,
        create_history_aware_retriever,
    )
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import (
        create_retrieval_chain,
        create_history_aware_retriever,
    )
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv(override=True)

# -- Constants -----------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_CHUNK_LENGTH = 50  # chars -- below this a chunk carries no meaning


# -- Private helpers -----------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_embeddings():
    """Load FastEmbed embedding model. Downloaded once, then cached in memory."""
    return FastEmbedEmbeddings(
        model_name=EMBEDDING_MODEL,
    )


def _get_llm():
    """
    Initialize Groq LLM.
    Defaults to llama-3.3-70b-versatile, configurable via GROQ_MODEL env var.
    Get a free API key at https://console.groq.com
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. "
            "Please add your GROQ_API_KEY in the .env file. "
            "Get a free key at https://console.groq.com"
        )

    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0,
        max_tokens=2048,
        timeout=30.0,
        max_retries=1,
    )


def _build_index_from_bytes(pdf_bytes: bytes, embeddings):
    """
    Write PDF bytes to a short-lived temp file, chunk it, build an in-memory
    FAISS index, then immediately delete the temp file.

    Returns (vector_store, chunk_count).
    Raises ValueError if the PDF yields no usable text.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(pdf_bytes)

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if not docs:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned document or entirely image-based."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LENGTH]

    if not chunks:
        raise ValueError(
            "PDF was read but all pages were blank or image-based. "
            "No usable text was found."
        )

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, len(chunks)


def _build_rag_chain(vector_store, llm):
    """
    Assemble the conversational RAG chain.
    Accepts {"input": str, "chat_history": list[BaseMessage]}.
    Returns {"answer": str, "context": list[Document], ...}.
    """
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question which might reference "
            "context in the chat history, formulate a standalone question that can be "
            "understood without the chat history. "
            "Do NOT answer -- only reformulate if needed, otherwise return as is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = (
        "You are a precise document assistant. "
        "Answer the user's question using ONLY the document context provided below.\n\n"
        "RULES:\n"
        "1. Only use information explicitly present in the context.\n"
        "2. If the answer is not in the context, respond with exactly: "
        "I cannot find this information in the uploaded document.\n"
        "3. Do not speculate, infer, or use outside knowledge.\n"
        "4. When useful, mention the page number (e.g., On page 3...).\n"
        "5. Keep answers clear and well-structured. Use bullet points for lists.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# -- Public API ----------------------------------------------------------------

def setup_qa_system(pdf_bytes: bytes, session_id: str) -> tuple:
    """
    Build the RAG system from raw PDF bytes in memory.
    """
    embeddings = _get_embeddings()
    vector_store, chunk_count = _build_index_from_bytes(pdf_bytes, embeddings)
    llm = _get_llm()
    rag_chain = _build_rag_chain(vector_store, llm)
    return rag_chain, chunk_count


def clear_session_index(session_id: str) -> None:
    pass


def clear_old_indices() -> None:
    pass
