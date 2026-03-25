import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_DIR = BASE_DIR / "faiss_index"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
DEFAULT_PDF = BASE_DIR / "Invoice And Billing_RAG.pdf"

FALLBACK_ANSWER = "I couldn't verify that from the uploaded documents."
FOLLOW_UP_PATTERN = re.compile(
    r"\b("
    r"it|its|they|them|their|that|those|these|this|former|latter|same|previous|above|below|"
    r"invoice|project|document|page|policy|payment|amount|status|due date"
    r")\b",
    re.IGNORECASE,
)
QUESTION_REWRITE_HINTS = ("what about", "how about", "and ", "why", "when", "which one")

SYSTEM_PROMPT = """You are a document-grounded invoice and billing assistant.

Use only the evidence in <CONTEXT>. If the answer is not fully supported there, reply exactly:
"I couldn't verify that from the uploaded documents."

Rules:
- Never invent invoice IDs, dates, totals, statuses, customers, or policy text.
- Every factual sentence or bullet must include bracket citations like [S1] or [S1][S2].
- Use conversation history only to resolve references such as "that invoice". Treat the context snippets as the only evidence.
- When a question matches multiple records, include every supported match.
- If you compute a total, only use values that appear in the context and show the result with citations.
- Keep answers concise and easy to scan.

<CONTEXT>
{context}
</CONTEXT>
"""

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the latest user question only when conversation history is needed to resolve references. "
            "Keep the original meaning, invoice IDs, dates, and amounts. "
            "Return only the standalone question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

_session_store: dict[str, ChatMessageHistory] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_directories() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)


def _normalize_page_number(metadata: dict[str, Any]) -> int:
    raw_page = metadata.get("page", 0)
    try:
        return int(raw_page) + 1
    except (TypeError, ValueError):
        return 1


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def clear_session_history(session_id: str) -> None:
    if session_id in _session_store:
        _session_store[session_id].clear()


def get_history_messages(session_id: str) -> list[dict[str, str]]:
    history = get_session_history(session_id)
    messages: list[dict[str, str]] = []
    for item in history.messages:
        role = "user" if item.__class__.__name__ == "HumanMessage" else "assistant"
        messages.append({"role": role, "content": item.content})
    return messages


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_rewriter_chain():
    return REWRITE_PROMPT | get_llm() | StrOutputParser()


@lru_cache(maxsize=1)
def get_answer_chain():
    return ANSWER_PROMPT | get_llm() | StrOutputParser()


class RAGService:
    def __init__(self) -> None:
        _ensure_directories()
        self._lock = RLock()
        self._vectorstore: FAISS | None = None
        self._index_signature: tuple[tuple[str, int, int], ...] | None = None

    def ensure_default_index(self) -> None:
        if self.vectorstore_ready():
            return
        if DEFAULT_PDF.exists():
            self.rebuild_vectorstore()

    def vectorstore_ready(self) -> bool:
        return (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists()

    def list_pdf_files(self) -> list[str]:
        files = sorted({path.name for path in UPLOAD_DIR.glob("**/*.pdf")})
        if DEFAULT_PDF.exists():
            files.insert(0, f"{DEFAULT_PDF.name} (default)")
        return files

    def get_dashboard_data(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        if self.vectorstore_ready() and not manifest:
            chunk_count = 0
            vectorstore = self._load_vectorstore()
            if vectorstore is not None:
                chunk_count = int(getattr(vectorstore.index, "ntotal", 0))
            built_at = datetime.fromtimestamp(
                (INDEX_DIR / "index.faiss").stat().st_mtime,
                tz=timezone.utc,
            ).isoformat()
            manifest = {
                "chunk_count": chunk_count,
                "file_count": len(self.list_pdf_files()),
                "files": self.list_pdf_files(),
                "built_at": built_at,
            }
        return {
            "vectorstore_ready": self.vectorstore_ready(),
            "files": self.list_pdf_files(),
            "sessions": len(_session_store),
            "messages": sum(len(history.messages) for history in _session_store.values()),
            "chunk_count": manifest.get("chunk_count", 0),
            "indexed_file_count": manifest.get("file_count", 0),
            "last_indexed_at": manifest.get("built_at"),
        }

    def rebuild_vectorstore(self) -> dict[str, Any]:
        pdf_files = sorted(UPLOAD_DIR.glob("**/*.pdf"))
        if DEFAULT_PDF.exists():
            pdf_files.insert(0, DEFAULT_PDF)

        unique_paths: list[Path] = []
        seen = set()
        for path in pdf_files:
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
                    page.metadata["page_number"] = _normalize_page_number(page.metadata)
                documents.extend(pages)
            except Exception as exc:
                print(f"Warning: skipping {pdf_path.name} — {exc}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_id"] = index
            chunk.metadata["page_number"] = _normalize_page_number(chunk.metadata)

        vectorstore = FAISS.from_documents(chunks, get_embeddings())
        vectorstore.save_local(str(INDEX_DIR))
        self._write_manifest(unique_paths, len(chunks))

        with self._lock:
            self._vectorstore = vectorstore
            self._index_signature = self._build_index_signature()

        return {
            "chunk_count": len(chunks),
            "file_count": len(unique_paths),
            "files": [path.name for path in unique_paths],
        }

    def answer(self, question: str, session_id: str) -> dict[str, Any]:
        history = get_session_history(session_id)
        standalone_question = self._rewrite_question(question, history.messages)
        docs = self._retrieve_docs(standalone_question, limit=6)
        sources = self._serialize_sources(docs)

        if not sources:
            answer = FALLBACK_ANSWER
        else:
            context = self._build_context(sources)
            answer = get_answer_chain().invoke(
                {
                    "input": question,
                    "chat_history": history.messages,
                    "context": context,
                }
            ).strip()
            if answer == FALLBACK_ANSWER:
                pass  # LLM explicitly said it couldn't answer
            elif not re.search(r"\[S\d+\]", answer):
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
        if not self.vectorstore_ready():
            return None

        signature = self._build_index_signature()
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

    def _build_index_signature(self) -> tuple[tuple[str, int, int], ...]:
        signature_items: list[tuple[str, int, int]] = []
        for filename in ("index.faiss", "index.pkl", "manifest.json"):
            path = INDEX_DIR / filename
            if path.exists():
                stat = path.stat()
                signature_items.append((filename, stat.st_mtime_ns, stat.st_size))
        return tuple(signature_items)

    def _serialize_sources(self, docs: list[Any]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for index, doc in enumerate(docs, start=1):
            sources.append(
                {
                    "id": f"S{index}",
                    "source": doc.metadata.get("source", "document"),
                    "page": doc.metadata.get("page_number")
                    or _normalize_page_number(doc.metadata),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "content": doc.page_content.strip(),
                }
            )
        return sources

    def _build_context(self, sources: list[dict[str, Any]]) -> str:
        blocks = []
        for source in sources:
            blocks.append(
                "\n".join(
                    [
                        f"[{source['id']}] File: {source['source']}",
                        f"[{source['id']}] Page: {source['page']}",
                        f"[{source['id']}] Content: {source['content']}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _write_manifest(self, files: list[Path], chunk_count: int) -> None:
        payload = {
            "built_at": _utc_now_iso(),
            "chunk_count": chunk_count,
            "file_count": len(files),
            "files": [path.name for path in files],
        }
        MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_manifest(self) -> dict[str, Any]:
        if not MANIFEST_PATH.exists():
            return {}
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}


service = RAGService()
