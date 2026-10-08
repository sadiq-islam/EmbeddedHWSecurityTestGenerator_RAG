"""Interactive hardware manual chat with configurable dense retrieval."""
from hashlib import sha256
import os
from pathlib import Path
import re
import streamlit as st
from dotenv import load_dotenv
from src.chatbot import HardwarePlanner
from src.config import RetrievalConfig
from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStore, load_embedder

# Environment secrets should never be logged
load_dotenv(Path(__file__).resolve().parent / ".env")
st.set_page_config(page_title="Hardware Planner Chat", layout="wide")

@st.cache_resource
def cached_embedder():
    # Model caching avoids reloading sentence-transformers on every UI interaction
    return load_embedder()


def reset_document():
    # Ensures no ghost state leaks across distinct uploaded manuals
    st.session_state.vector_store = None
    st.session_state.chat_messages = []
    st.session_state.current_draft = ""
    st.session_state.extraction_stats = None
    if st.session_state.planner:
        st.session_state.planner.init_chat()


def show_sources(hits):
    # Transparency rendering for document sources
    with st.expander("Document Sources & Evidence"):
        for hit in hits:
            pages = ", ".join(map(str, hit["pages"])) or "unknown"
            st.markdown(f"**Evidence `{hit['chunk_id']}` — Pages {pages}**")
            st.caption(f"{hit['score_kind']} score: {hit['score']:.3f}; "
                       "provenance applies to the parent chunk")
            st.code(hit["content"], language=None)
            st.json(hit["provenance"])


# Session State Initialization block
for key, value in {"planner": None, "vector_store": None, "chat_messages": [],
                   "current_draft": "", "upload_id": None, "extraction_stats": None}.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("Embedded Hardware Security Test Planner")

with st.sidebar:
    st.header("Document Setup")
    api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    planner = st.session_state.planner

    # Manage Planner object lifecycle automatically on API key changes
    if api_key and (planner is None or planner.client.api_key != api_key):
        if planner:
            planner.close()
        st.session_state.planner = HardwarePlanner(api_key=api_key)
        st.session_state.chat_messages = []
        st.session_state.current_draft = ""
    elif not api_key and planner:
        planner.close()
        st.session_state.planner = None

    rerank = st.checkbox("Enable reranking", value=True)
    top_k = int(st.number_input("Final top-k", min_value=1, max_value=50, value=4))
    candidate_k = int(st.number_input("Candidate top-k", min_value=top_k,
                                      max_value=200, value=max(top_k, 15)))
    chunk_tokens = int(st.number_input("Embedding chunk tokens", min_value=32, max_value=256, value=240))
    overlap = int(st.number_input("Overlap tokens", min_value=0,
                                 max_value=chunk_tokens - 1, value=min(24, chunk_tokens - 1)))

    config = RetrievalConfig(top_k=top_k, candidate_k=candidate_k, rerank=rerank,
                             chunk_tokens=chunk_tokens, overlap_tokens=overlap)

    uploaded = st.file_uploader("Upload PDF Specification", type=["pdf"])
    data = uploaded.getvalue() if uploaded is not None else None

    # Track file differences using sha256 to automatically reset memory if a new file arrives
    upload_id = sha256(data).hexdigest() if data is not None else None
    if upload_id != st.session_state.upload_id:
        reset_document()
        st.session_state.upload_id = upload_id

    if uploaded is not None and st.button("Process Document"):
        reset_document()
        with st.spinner("Parsing document and building index..."):
            try:
                store = VectorStore(cached_embedder(), config)
                with PDFProcessor(data, source_name=uploaded.name) as processor:
                    doc, chunks, pages = processor.extract_document()
                store.build_index(chunks, processor.document_id, uploaded.name)
                st.session_state.vector_store = store
                st.session_state.extraction_stats = {"Pages represented": f"{len(doc.pages)} / {pages}",
                    "Tables detected": len(doc.tables), "Indexed chunks": len(store.chunks)}
            except Exception as exc:
                st.error(f"Document processing failed: {type(exc).__name__}")

    if st.session_state.extraction_stats:
        for label, value in st.session_state.extraction_stats.items():
            st.metric(label, value)
        st.caption("Page/table counts do not measure extraction accuracy.")

store = st.session_state.vector_store
if store:
    # State validation: Disable chats if chunk configs mutate after processing the doc
    if (store.config.chunk_tokens, store.config.overlap_tokens) != (chunk_tokens, overlap):
        st.warning("Chunk settings changed. Process the document again before chatting.")
        ready = False
    else:
        store.config = config
        ready = True
else:
    ready = False

# Enable exporting validated results directly to markdown
if st.session_state.current_draft:
    st.sidebar.download_button("Download Current Draft (.md)", st.session_state.current_draft,
                               file_name="test_draft.md", mime="text/markdown")

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

if prompt := st.chat_input("Ask about the specs or request a proposed test...",
                           disabled=not (ready and st.session_state.planner)):
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        with st.chat_message("assistant"), st.spinner("Retrieving and generating..."):
            response, hits = st.session_state.planner.chat(prompt, store=store)

            # Post-processing: Handle automated payload tags without rendering them to the user
            response = response.replace("[DOWNLOAD_READY]", "").strip()
            draft = re.search(r"```markdown\s*\n(.*?)```", response, re.DOTALL)
            if draft:
                st.session_state.current_draft = draft.group(1).strip()

            st.markdown(response)
            if hits:
                show_sources(hits)

        st.session_state.chat_messages.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response, "sources": hits}])

        # Rerun makes the newly created download button available immediately.
        st.rerun()
    except Exception as exc:
        st.error(f"Generation failed: {type(exc).__name__}. No draft was updated.")