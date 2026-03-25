from pathlib import Path

import streamlit as st

from rag_service import UPLOAD_DIR, service

st.set_page_config(
    page_title="Document Parser Assistant",
    page_icon="📄",
    layout="wide",
)


def ensure_state() -> None:
    defaults = {
        "session_id": "streamlit-session",
        "messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_uploads(files: list) -> tuple[list[str], list[str]]:
    saved: list[str] = []
    errors: list[str] = []

    for upload in files:
        if not upload.name.lower().endswith(".pdf"):
            errors.append(f"{upload.name}: skipped because it is not a PDF.")
            continue

        destination = UPLOAD_DIR / Path(upload.name).name
        destination.write_bytes(upload.getbuffer())
        saved.append(destination.name)

    return saved, errors


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for source in sources:
            st.markdown(
                f"**{source['id']}** | `{source['source']}` | Page {source['page']}"
            )
            st.caption(source["content"])


ensure_state()
service.ensure_default_index()
dashboard = service.get_dashboard_data()

st.title("Document Parser Assistant")
st.caption("Streamlit UI powered by the same cached RAG service.")

with st.sidebar:
    st.subheader("System")
    st.write(f"Indexed files: {dashboard['indexed_file_count']}")
    st.write(f"Indexed chunks: {dashboard['chunk_count']}")
    st.write(f"Messages stored: {dashboard['messages']}")

    st.divider()
    st.subheader("Upload PDFs")
    uploads = st.file_uploader(
        "Add one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Index Documents", use_container_width=True):
        if not uploads:
            st.warning("Choose at least one PDF first.")
        else:
            with st.spinner("Saving files and rebuilding the index..."):
                saved, errors = save_uploads(uploads)
                if saved:
                    summary = service.rebuild_vectorstore()
                    st.success(
                        f"Indexed {summary['chunk_count']} chunks from {summary['file_count']} PDF files."
                    )
                for error in errors:
                    st.warning(error)
                if saved:
                    st.rerun()

    st.divider()
    st.subheader("Indexed Files")
    for filename in service.list_pdf_files():
        st.caption(filename)

    if st.button("Clear Chat", use_container_width=True):
        from rag_service import clear_session_history

        clear_session_history(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Indexed Files", dashboard["indexed_file_count"])
col2.metric("Indexed Chunks", dashboard["chunk_count"])
col3.metric("Active Sessions", dashboard["sessions"])
col4.metric("Stored Messages", dashboard["messages"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

prompt = st.chat_input("Ask a question about your uploaded documents...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = service.answer(prompt, st.session_state.session_id)
        st.markdown(result["answer"])
        render_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
