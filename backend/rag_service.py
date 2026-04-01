import re
from pathlib import Path
from threading import RLock
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.rag_chains import get_answer_chain, get_embeddings, get_rewriter_chain
from backend.rag_config import (
    FALLBACK_ANSWER,
    FOLLOW_UP_PATTERN,
    INDEX_DIR,
    QUESTION_REWRITE_HINTS,
    UPLOAD_DIR,
    ensure_directories,
    utc_now_iso,
)
from backend.rag_sessions import (
    clear_session_history,
    get_history_messages,
    get_session_history,
    get_session_metrics,
)
from backend.rag_utils import (
    build_context,
    build_index_signature,
    load_manifest,
    normalize_page_number,
    serialize_sources,
    write_manifest,
)


class RAGService:
    """Simple RAG service: load PDFs, chunk them, store them in FAISS, and answer from retrieved chunks."""

    def __init__(self) -> None:
        """Initializes folders, a lock, and the cached in-memory vectorstore."""

        ensure_directories()
        self._lock = RLock()
        self._vectorstore: FAISS | None = None
        self._index_signature: tuple[tuple[str, int, int], ...] | None = None

    def ensure_default_index(self) -> None:
        """Builds the vector index on startup when PDFs already exist."""

        if not self.vectorstore_ready() and any(UPLOAD_DIR.glob("**/*.pdf")):
            self.rebuild_vectorstore()

    def vectorstore_ready(self) -> bool:
        """Checks whether the saved FAISS files exist on disk."""

        return (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists()

    def list_pdf_files(self) -> list[str]:
        """Returns the uploaded PDF filenames."""

        return sorted({path.name for path in UPLOAD_DIR.glob("**/*.pdf")})

    def get_dashboard_data(self) -> dict[str, Any]:
        """Returns counts used by the frontend dashboard."""

        manifest = load_manifest()
        pdf_files = self.list_pdf_files()

        if self.vectorstore_ready() and not manifest:
            chunk_count = 0
            vectorstore = self._load_vectorstore()
            if vectorstore is not None:
                chunk_count = int(getattr(vectorstore.index, "ntotal", 0))
            manifest = {
                "chunk_count": chunk_count,
                "file_count": len(pdf_files),
                "files": pdf_files,
                "built_at": utc_now_iso(),
            }

        return {
            "vectorstore_ready": self.vectorstore_ready(),
            "files": pdf_files,
            **get_session_metrics(),
            "chunk_count": manifest.get("chunk_count", 0),
            "indexed_file_count": manifest.get("file_count", 0),
            "last_indexed_at": manifest.get("built_at"),
        }

    def rebuild_vectorstore(self) -> dict[str, Any]:
        """Loads PDFs, splits them into chunks, embeds them, and saves the FAISS index."""

        pdf_paths: list[Path] = []
        seen = set()
        for path in sorted(UPLOAD_DIR.glob("**/*.pdf")):
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                pdf_paths.append(path)

        if not pdf_paths:
            raise ValueError("No PDF files found to index.")

        documents = []
        for pdf_path in pdf_paths:
            try:
                pages = PyPDFLoader(str(pdf_path)).load()
                for page in pages:
                    page.metadata["source"] = pdf_path.name
                    page.metadata["page_number"] = normalize_page_number(page.metadata)
                documents.extend(pages)
            except Exception as exc:
                print(f"Warning: skipping {pdf_path.name} — {exc}")

        if not documents:
            raise ValueError("No readable PDF content found to index.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_id"] = index
            chunk.metadata["page_number"] = normalize_page_number(chunk.metadata)

        vectorstore = FAISS.from_documents(chunks, get_embeddings())
        vectorstore.save_local(str(INDEX_DIR))
        write_manifest(pdf_paths, len(chunks))

        with self._lock:
            self._vectorstore = vectorstore
            self._index_signature = build_index_signature(INDEX_DIR)

        return {
            "chunk_count": len(chunks),
            "file_count": len(pdf_paths),
            "files": [path.name for path in pdf_paths],
        }

    def answer(self, question: str, session_id: str) -> dict[str, Any]:
        """Answers a question using retrieved PDF chunks and stores the chat turn."""

        history = get_session_history(session_id)
        standalone_question = self._rewrite_question(question, history.messages)
        docs = self._retrieve_docs(standalone_question, limit=6)
        sources = serialize_sources(docs)

        if not sources:
            answer = FALLBACK_ANSWER
        else:
            answer = get_answer_chain().invoke(
                {
                    "input": question,
                    "chat_history": history.messages,
                    "context": build_context(sources),
                }
            ).strip()
            if answer != FALLBACK_ANSWER and not re.search(r"\[S\d+\]", answer):
                answer += "\n\n*(Note: answer could not be verified with inline citations.)*"

        history.add_user_message(question)
        history.add_ai_message(answer)
        return {
            "answer": answer,
            "question": question,
            "standalone_question": standalone_question,
            "sources": sources,
        }

    def _rewrite_question(self, question: str, chat_history: list[Any]) -> str:
        """Rewrites follow-up questions into standalone form when chat history is needed."""

        if not chat_history:
            return question

        lowered = question.lower().strip()
        needs_rewrite = FOLLOW_UP_PATTERN.search(question) or any(
            lowered.startswith(prefix) for prefix in QUESTION_REWRITE_HINTS
        )
        if not needs_rewrite:
            return question

        rewritten = get_rewriter_chain().invoke(
            {"input": question, "chat_history": chat_history}
        ).strip()
        return rewritten or question

    def _retrieve_docs(self, query: str, limit: int) -> list[Any]:
        """Retrieves the most relevant PDF chunks from FAISS."""

        vectorstore = self._load_vectorstore()
        if vectorstore is None:
            return []

        return vectorstore.max_marginal_relevance_search(
            query,
            k=limit,
            fetch_k=max(16, limit * 3),
            lambda_mult=0.35,
        )

    def _load_vectorstore(self) -> FAISS | None:
        """Loads the FAISS index once and reuses it until index files change."""

        if not self.vectorstore_ready():
            return None

        signature = build_index_signature(INDEX_DIR)
        with self._lock:
            if self._vectorstore is not None and self._index_signature == signature:
                return self._vectorstore

            self._vectorstore = FAISS.load_local(
                str(INDEX_DIR),
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )
            self._index_signature = signature
            return self._vectorstore


service = RAGService()
