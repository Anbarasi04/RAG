import streamlit as st
import os
import tempfile
from pathlib import Path

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot · Groq",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1b2a 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid #2a2a5a;
    }
    .main-header h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #e0e7ff;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #7c83a8;
        margin: 0;
        font-size: 0.95rem;
    }
    .badge {
        display: inline-block;
        background: #22d3ee22;
        color: #22d3ee;
        border: 1px solid #22d3ee55;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        margin-right: 6px;
        margin-top: 8px;
    }
    .stat-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number { font-family: 'Space Mono', monospace; font-size: 1.6rem; color: #22d3ee; font-weight: 700; }
    .stat-label  { font-size: 0.75rem; color: #6b7280; margin-top: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>⚡ RAG Document Chatbot</h1>
        <p>Upload documents, ask questions — powered by Groq ultra-fast inference + free local embeddings</p>
        <span class="badge">Groq LLM</span>
        <span class="badge">HuggingFace Embeddings</span>
        <span class="badge">FAISS</span>
        <span class="badge">LangChain</span>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )
    if not groq_api_key:
        st.info("🔑 Get a **free** key at [console.groq.com](https://console.groq.com)")

    st.markdown("---")
    st.markdown("**LLM Model (Groq)**")
    model_name = st.selectbox(
        "Groq Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("**Embedding Model** *(free, runs locally)*")
    embed_model = st.selectbox(
        "Embedding",
        [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-MiniLM-L3-v2",
        ],
        index=0,
        label_visibility="collapsed",
        help="sentence-transformers models — completely free, no API key needed.",
    )

    st.markdown("---")
    st.markdown("**Chunking & Retrieval**")
    chunk_size    = st.slider("Chunk Size",     256, 2048, 512,  step=128)
    chunk_overlap = st.slider("Chunk Overlap",    0,  512,  64,  step=32)
    top_k         = st.slider("Top-K Retrieval",  1,   10,   4)

    st.markdown("---")
    st.markdown("**Supported formats**")
    st.markdown("`PDF` `DOCX` `TXT` `PPTX` `CSV`")

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
defaults = {
    "chat_history": [],      # list of HumanMessage / AIMessage objects
    "display_history": [],   # list of {"role", "content", "sources"} for rendering
    "rag_chain": None,
    "retriever": None,
    "processed_files": [],
    "doc_count": 0,
    "chunk_count": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# File loader helpers
# ─────────────────────────────────────────────
LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".csv":  CSVLoader,
}

def load_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in LOADER_MAP:
        st.warning(f"Unsupported type: **{uploaded_file.name}**")
        return []
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        docs = LOADER_MAP[suffix](tmp_path).load()
        for d in docs:
            d.metadata["source"] = uploaded_file.name
        return docs
    except Exception as e:
        st.error(f"Error loading **{uploaded_file.name}**: {e}")
        return []
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# Build RAG chain
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Downloading embedding model (first run only)…")
def get_embeddings(model_id: str):
    """Cache the embedding model so it is only downloaded once."""
    return HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{model_id}",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_rag(documents, groq_api_key, model_name, embed_model):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        st.error("No text could be extracted from the uploaded files.")
        return None, None, 0

    embeddings  = get_embeddings(embed_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": top_k})

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0,
    )

    # Prompt that includes conversation history + retrieved context
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer the user's question using ONLY "
         "the context below. If the answer is not in the context, say so.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # LCEL chain: retrieve → format → prompt → LLM → parse
    rag_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["question"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever, len(chunks)


uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt", "pptx", "csv"],
    accept_multiple_files=True,
)

if uploaded_files and groq_api_key:
    current_names = sorted(f.name for f in uploaded_files)
    if current_names != st.session_state.processed_files:
        with st.status("Processing documents…", expanded=True) as status:
            st.write("Loading files…")
            all_docs = []
            for f in uploaded_files:
                docs = load_file(f)
                all_docs.extend(docs)
                st.write(f"**{f.name}** — {len(docs)} page(s)")

            st.write("Building FAISS vector index…")
            rag_chain, retriever, n_chunks = build_rag(all_docs, groq_api_key, model_name, embed_model)

            if rag_chain:
                st.session_state.rag_chain       = rag_chain
                st.session_state.retriever       = retriever
                st.session_state.processed_files = current_names
                st.session_state.chat_history    = []
                st.session_state.display_history = []
                st.session_state.doc_count       = len(all_docs)
                st.session_state.chunk_count     = n_chunks
                status.update(label="Index built — ready to chat!", state="complete")

elif uploaded_files and not groq_api_key:
    st.info("Enter your Groq API key in the sidebar to process the documents.")


if st.session_state.rag_chain:
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (len(st.session_state.processed_files), "Files loaded"),
        (st.session_state.doc_count,             "Pages extracted"),
        (st.session_state.chunk_count,           "Chunks indexed"),
        (len(st.session_state.display_history)//2, "Turns so far"),
    ]
    for col, (num, label) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-number">{num}</div>'
                f'<div class="stat-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    for msg in st.session_state.display_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Retrieved sources"):
                    seen = set()
                    for src in msg["sources"]:
                        key = (src.metadata.get("source", ""), src.metadata.get("page", ""))
                        if key in seen:
                            continue
                        seen.add(key)
                        st.markdown(
                            f"**{src.metadata.get('source', 'Unknown')}** "
                            f"— page {src.metadata.get('page', '?')}"
                        )
                        st.caption(src.page_content[:300] + "…")

    user_query = st.chat_input("Ask anything about your documents…")

    if user_query:
        st.session_state.display_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("⚡ Groq is thinking…"):
                sources = st.session_state.retriever.invoke(user_query)

                answer = st.session_state.rag_chain.invoke({
                    "question": user_query,
                    "chat_history": st.session_state.chat_history,
                })

            st.markdown(answer)

            if sources:
                with st.expander("📚 Retrieved sources"):
                    seen = set()
                    for src in sources:
                        key = (src.metadata.get("source", ""), src.metadata.get("page", ""))
                        if key in seen:
                            continue
                        seen.add(key)
                        st.markdown(
                            f"**{src.metadata.get('source', 'Unknown')}** "
                            f"— page {src.metadata.get('page', '?')}"
                        )
                        st.caption(src.page_content[:300] + "…")

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=answer))
        st.session_state.display_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

    _, right = st.columns([9, 1])
    with right:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history    = []
            st.session_state.display_history = []
            st.rerun()

else:
    if not uploaded_files:
        st.markdown(
            """
            <div style="text-align:center; padding: 3.5rem 1rem; color: #4b5563;">
                <div style="font-size:3.5rem">📂</div>
                <p style="font-size:1.15rem; margin-top:0.75rem; font-weight:600;">
                    Upload a document to get started
                </p>
                <p style="font-size:0.85rem;">Supported: PDF · DOCX · TXT · PPTX · CSV</p>
            </div>
            """,
            unsafe_allow_html=True,
        )