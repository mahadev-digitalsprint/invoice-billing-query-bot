import re
from pathlib import Path
from threading import RLock
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from backend.rag_chains import (
    get_answer_chain,
    get_embeddings,
    get_extraction_chain,
    get_rewriter_chain,
)
from backend.rag_config import (
    FALLBACK_ANSWER,
    FOLLOW_UP_PATTERN,
    INDEX_DIR,
    QUESTION_REWRITE_HINTS,
    UPLOAD_DIR,
    ensure_directories,
    utc_now_iso,
)
from backend.rag_records import (
    build_structured_record,
    get_structured_invoice,
    list_structured_invoices,
    retrieve_structured_sources,
    save_structured_record,
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
    parse_llm_json,
    serialize_sources,
    structured_json_path_for,
    write_manifest,
)


class RAGService:
    """Coordinates indexing, structured extraction, retrieval, and grounded answering."""

    def __init__(self) -> None:
        """Initializes directories, lock state, and the cached in-memory vectorstore reference."""

        ensure_directories()
        self._lock = RLock()
        self._vectorstore: FAISS | None = None
        self._index_signature: tuple[tuple[str, int, int], ...] | None = None

    def ensure_default_index(self) -> None:
        """Ensures the backend has both a FAISS index and structured JSON for existing PDFs."""

        if not self.vectorstore_ready() and any(UPLOAD_DIR.glob("**/*.pdf")):
            self.rebuild_vectorstore()
        self.ensure_structured_data()

    def vectorstore_ready(self) -> bool:
        """Checks whether the saved FAISS index files already exist on disk."""

        return (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists()

    def list_pdf_files(self) -> list[str]:
        """Returns the uploaded PDF filenames in sorted order."""

        return sorted({path.name for path in UPLOAD_DIR.glob("**/*.pdf")})

    def get_dashboard_data(self) -> dict[str, Any]:
        """Builds the dashboard counters shown by the frontend on page load and refresh."""

        manifest = load_manifest()
        pdf_files = self.list_pdf_files()
        structured_invoices = list_structured_invoices()
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
            "structured_invoice_count": len(structured_invoices),
        }

    def rebuild_vectorstore(self) -> dict[str, Any]:
        """Loads all PDFs, splits them into chunks, embeds them, and saves a fresh FAISS index."""

        unique_paths: list[Path] = []
        seen = set()
        for path in sorted(UPLOAD_DIR.glob("**/*.pdf")):
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                unique_paths.append(path)

        if not unique_paths:
            raise ValueError("No PDF files found to index.")

        documents = []
        for pdf_path in unique_paths:
            try:
                pages = PyPDFLoader(str(pdf_path)).load()
                for page in pages:
                    page.metadata["source"] = pdf_path.name
                    page.metadata["page_number"] = normalize_page_number(page.metadata)
                documents.extend(pages)
            except Exception as exc:
                print(f"Warning: skipping {pdf_path.name} — {exc}")

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""],
        ).split_documents(documents)
        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_id"] = index
            chunk.metadata["page_number"] = normalize_page_number(chunk.metadata)

        vectorstore = FAISS.from_documents(chunks, get_embeddings())
        vectorstore.save_local(str(INDEX_DIR))
        write_manifest(unique_paths, len(chunks))

        with self._lock:
            self._vectorstore = vectorstore
            self._index_signature = build_index_signature(INDEX_DIR)

        return {
            "chunk_count": len(chunks),
            "file_count": len(unique_paths),
            "files": [path.name for path in unique_paths],
        }

    def ensure_structured_data(self) -> dict[str, Any]:
        """Extracts structured JSON only for PDFs that do not already have one."""

        missing_files = [
            pdf_path.name
            for pdf_path in sorted(UPLOAD_DIR.glob("**/*.pdf"))
            if not structured_json_path_for(pdf_path).exists()
        ]
        if not missing_files:
            return {"extracted": [], "errors": []}
        return self.extract_structured_invoices(missing_files)

    def extract_structured_invoices(self, filenames: list[str] | None = None) -> dict[str, Any]:
        """Extracts structured JSON for one set of PDFs and returns success/error summaries."""

        targets = (
            sorted(UPLOAD_DIR.glob("**/*.pdf"))
            if filenames is None
            else [
                candidate
                for filename in filenames
                if (candidate := UPLOAD_DIR / Path(filename).name).exists()
            ]
        )
        extracted: list[str] = []
        errors: list[str] = []
        for pdf_path in targets:
            try:
                extracted.append(str(self.extract_structured_invoice(pdf_path)["json_file"]))
            except Exception as exc:
                errors.append(f"{pdf_path.name}: {exc}")
        return {"extracted": extracted, "errors": errors}

    def extract_structured_invoice(self, pdf_path: Path) -> dict[str, Any]:
        """Reads one PDF, sends its text to the extraction chain, and saves the structured result."""

        reader = PdfReader(str(pdf_path))
        extracted_pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                extracted_pages.append(f"Page {index}\n{text.strip()}")

        invoice_text = "\n\n".join(extracted_pages).strip()
        if not invoice_text:
            raise ValueError("No extractable text found in the PDF.")

        payload = parse_llm_json(
            get_extraction_chain().invoke(
                {
                    "file_name": pdf_path.name,
                    "page_count": len(reader.pages),
                    "text": invoice_text,
                }
            )
        )
        record = build_structured_record(pdf_path, payload, len(reader.pages))
        save_structured_record(pdf_path, record)
        return record

    def list_structured_invoices(self) -> list[dict[str, Any]]:
        """Delegates to the structured-record helper to build invoice summary rows."""

        return list_structured_invoices()

    def get_structured_invoice(self, json_name: str) -> dict[str, Any]:
        """Delegates to the structured-record helper to load one JSON invoice."""

        return get_structured_invoice(json_name)

    def answer(self, question: str, session_id: str) -> dict[str, Any]:
        """Runs the full RAG flow for one user question and stores the result in session history."""

        history = get_session_history(session_id)
        standalone_question = self._rewrite_question(question, history.messages)
        structured_sources = retrieve_structured_sources(standalone_question, limit=3)
        preferred_source = None
        if len(structured_sources) == 1:
            preferred_source = str(structured_sources[0]["source"]).replace(
                " (structured)",
                "",
            )

        docs = self._retrieve_docs(
            standalone_question,
            limit=2 if structured_sources else 6,
            preferred_source=preferred_source,
        )
        sources = structured_sources + serialize_sources(
            docs,
            start_index=len(structured_sources) + 1,
        )

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
        """Rewrites follow-up questions into standalone form when history is needed."""

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

    def _retrieve_docs(
        self,
        query: str,
        limit: int,
        preferred_source: str | None = None,
    ) -> list[Any]:
        """Retrieves the most relevant PDF chunks from FAISS, optionally preferring one file."""

        vectorstore = self._load_vectorstore()
        if vectorstore is None:
            return []

        docs = vectorstore.max_marginal_relevance_search(
            query,
            k=limit,
            fetch_k=max(16, limit * 3),
            lambda_mult=0.35,
        )
        if preferred_source:
            matching_docs = [
                doc for doc in docs if doc.metadata.get("source") == preferred_source
            ]
            if matching_docs:
                return matching_docs[:limit]
        return docs

    def _load_vectorstore(self) -> FAISS | None:
        """Loads the saved FAISS index once and reuses it until the index files change."""

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
