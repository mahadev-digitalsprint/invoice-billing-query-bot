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
from pypdf import PdfReader

load_dotenv()

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_DIR = BASE_DIR / "faiss_index"
STRUCTURED_DIR = BASE_DIR / "structured_data"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

FALLBACK_ANSWER = "I couldn't verify that from the uploaded documents."
FOLLOW_UP_PATTERN = re.compile(
    r"\b("
    r"it|its|they|them|their|that|those|these|this|former|latter|same|previous|above|below|"
    r"invoice|project|document|page|policy|payment|amount|status|due date"
    r")\b",
    re.IGNORECASE,
)
QUESTION_REWRITE_HINTS = ("what about", "how about", "and ", "why", "when", "which one")
STRUCTURED_QUERY_STOPWORDS = {
    "about",
    "after",
    "amount",
    "answer",
    "billing",
    "client",
    "data",
    "details",
    "document",
    "file",
    "from",
    "give",
    "have",
    "invoice",
    "invoices",
    "json",
    "page",
    "please",
    "record",
    "seller",
    "show",
    "tell",
    "that",
    "the",
    "this",
    "total",
    "totals",
    "what",
    "when",
    "which",
    "with",
}

SYSTEM_PROMPT = """You are a document-grounded invoice and billing assistant.

Use only the evidence in <CONTEXT>. If the answer is not fully supported there, reply exactly:
"I couldn't verify that from the uploaded documents."

Rules:
- Never invent invoice IDs, dates, totals, statuses, customers, or policy text.
- Every factual sentence or bullet must include bracket citations like [S1] or [S1][S2].
- Use conversation history only to resolve references such as "that invoice". Treat the context snippets as the only evidence.
- The context may include structured invoice JSON summaries and raw PDF snippets. You may answer from either, but only if the answer is directly supported by the provided context.
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

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You extract structured invoice data from PDF text.

Return only valid JSON. Do not wrap it in markdown fences.

Use this exact shape:
{{
  "invoice_number": string | null,
  "issue_date": string | null,
  "currency": string | null,
  "seller": {{
    "name": string | null,
    "address": string | null,
    "tax_id": string | null,
    "iban": string | null
  }},
  "client": {{
    "name": string | null,
    "address": string | null,
    "tax_id": string | null,
    "iban": string | null
  }},
  "items": [
    {{
      "line_number": string | null,
      "description": string | null,
      "quantity": string | null,
      "unit": string | null,
      "net_price": string | null,
      "net_amount": string | null,
      "vat_rate": string | null,
      "gross_amount": string | null
    }}
  ],
  "tax_breakdown": [
    {{
      "vat_rate": string | null,
      "net_amount": string | null,
      "vat_amount": string | null,
      "gross_amount": string | null
    }}
  ],
  "totals": {{
    "net_total": string | null,
    "vat_total": string | null,
    "gross_total": string | null
  }},
  "notes": [string]
}}

Rules:
- Use null for missing scalar fields and [] for missing arrays.
- Normalize IBAN by removing internal spaces, for example "G B 7 8 ..." becomes "GB78...".
- Preserve invoice identifiers exactly when clear.
- Keep dates as they appear in the invoice when clear.
- Keep amounts as strings.
- Put any uncertainty or missing-field explanation into "notes".
""",
        ),
        (
            "human",
            "File name: {file_name}\nPage count: {page_count}\n\nInvoice text:\n{text}",
        ),
    ]
)

_session_store: dict[str, ChatMessageHistory] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_directories() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)
    STRUCTURED_DIR.mkdir(exist_ok=True)


def _normalize_whitespace(value: Any) -> str | None:
    if value is None:
        return None
    collapsed = re.sub(r"\s+", " ", str(value)).strip()
    return collapsed or None


def _clean_llm_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned


def _parse_llm_json(text: str) -> dict[str, Any]:
    payload = json.loads(_clean_llm_json(text))
    if not isinstance(payload, dict):
        raise ValueError("Structured extraction did not return a JSON object.")
    return payload


def _coerce_party(value: Any) -> dict[str, Any]:
    party = value if isinstance(value, dict) else {}
    return {
        "name": _normalize_whitespace(party.get("name")),
        "address": _normalize_whitespace(party.get("address")),
        "tax_id": _normalize_whitespace(party.get("tax_id")),
        "iban": _normalize_whitespace(party.get("iban")),
    }


def _coerce_line_items(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "line_number": _normalize_whitespace(item.get("line_number")),
                "description": _normalize_whitespace(item.get("description")),
                "quantity": _normalize_whitespace(item.get("quantity")),
                "unit": _normalize_whitespace(item.get("unit")),
                "net_price": _normalize_whitespace(item.get("net_price")),
                "net_amount": _normalize_whitespace(item.get("net_amount")),
                "vat_rate": _normalize_whitespace(item.get("vat_rate")),
                "gross_amount": _normalize_whitespace(item.get("gross_amount")),
            }
        )
    return normalized


def _coerce_tax_breakdown(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "vat_rate": _normalize_whitespace(item.get("vat_rate")),
                "net_amount": _normalize_whitespace(item.get("net_amount")),
                "vat_amount": _normalize_whitespace(item.get("vat_amount")),
                "gross_amount": _normalize_whitespace(item.get("gross_amount")),
            }
        )
    return normalized


def _coerce_totals(value: Any) -> dict[str, Any]:
    totals = value if isinstance(value, dict) else {}
    return {
        "net_total": _normalize_whitespace(totals.get("net_total")),
        "vat_total": _normalize_whitespace(totals.get("vat_total")),
        "gross_total": _normalize_whitespace(totals.get("gross_total")),
    }


def _coerce_notes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        note = _normalize_whitespace(item)
        if note:
            normalized.append(note)
    return normalized


def _flatten_for_search(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(_flatten_for_search(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_for_search(item) for item in value)
    return str(value)


def _structured_json_path_for(pdf_path: Path) -> Path:
    return STRUCTURED_DIR / f"{pdf_path.stem}.json"


def _tokenize_query(query: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
    return {token for token in tokens if token not in STRUCTURED_QUERY_STOPWORDS}


def _score_structured_record(query: str, record: dict[str, Any]) -> int:
    lowered_query = query.lower()
    search_blob = _flatten_for_search(record).lower()
    score = 0

    for token in _tokenize_query(query):
        if token in search_blob:
            score += 1

    invoice_number = str(record.get("invoice_number") or "").strip().lower()
    source_file = str(record.get("source_file") or "").strip().lower()
    if invoice_number and invoice_number in lowered_query:
        score += 5
    if source_file and source_file in lowered_query:
        score += 4
    if "iban" in lowered_query and "iban" in search_blob:
        score += 2
    if "tax" in lowered_query and "tax" in search_blob:
        score += 2

    return score


def _build_structured_source_content(record: dict[str, Any]) -> str:
    seller = record.get("seller") or {}
    client = record.get("client") or {}
    totals = record.get("totals") or {}
    lines = [
        "Structured invoice data",
        f"invoice_number: {record.get('invoice_number') or 'unknown'}",
        f"issue_date: {record.get('issue_date') or 'unknown'}",
        f"currency: {record.get('currency') or 'unknown'}",
        f"seller_name: {seller.get('name') or 'unknown'}",
        f"seller_tax_id: {seller.get('tax_id') or 'unknown'}",
        f"seller_iban: {seller.get('iban') or 'unknown'}",
        f"client_name: {client.get('name') or 'unknown'}",
        f"client_tax_id: {client.get('tax_id') or 'unknown'}",
        f"net_total: {totals.get('net_total') or 'unknown'}",
        f"vat_total: {totals.get('vat_total') or 'unknown'}",
        f"gross_total: {totals.get('gross_total') or 'unknown'}",
    ]

    items = record.get("items") or []
    if items:
        item_summaries = []
        for item in items[:8]:
            item_summaries.append(
                ", ".join(
                    part
                    for part in (
                        f"line {item.get('line_number')}" if item.get("line_number") else None,
                        item.get("description"),
                        f"qty {item.get('quantity')}" if item.get("quantity") else None,
                        f"net {item.get('net_amount')}" if item.get("net_amount") else None,
                        f"gross {item.get('gross_amount')}" if item.get("gross_amount") else None,
                    )
                    if part
                )
            )
        lines.append("items: " + " | ".join(item_summaries))

    notes = record.get("notes") or []
    if notes:
        lines.append("notes: " + " | ".join(notes[:3]))

    return "\n".join(lines)


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
        model="gemini-3.1-pro-preview",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_rewriter_chain():
    return REWRITE_PROMPT | get_llm() | StrOutputParser()


@lru_cache(maxsize=1)
def get_answer_chain():
    return ANSWER_PROMPT | get_llm() | StrOutputParser()


@lru_cache(maxsize=1)
def get_extraction_chain():
    return EXTRACTION_PROMPT | get_llm() | StrOutputParser()


class RAGService:
    def __init__(self) -> None:
        _ensure_directories()
        self._lock = RLock()
        self._vectorstore: FAISS | None = None
        self._index_signature: tuple[tuple[str, int, int], ...] | None = None

    def ensure_default_index(self) -> None:
        if not self.vectorstore_ready() and any(UPLOAD_DIR.glob("**/*.pdf")):
            self.rebuild_vectorstore()
        self.ensure_structured_data()

    def vectorstore_ready(self) -> bool:
        return (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists()

    def list_pdf_files(self) -> list[str]:
        return sorted({path.name for path in UPLOAD_DIR.glob("**/*.pdf")})

    def get_dashboard_data(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        structured_invoices = self.list_structured_invoices()
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
            "structured_invoice_count": len(structured_invoices),
        }

    def rebuild_vectorstore(self) -> dict[str, Any]:
        pdf_files = sorted(UPLOAD_DIR.glob("**/*.pdf"))

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

    def ensure_structured_data(self) -> dict[str, Any]:
        missing_files: list[str] = []
        for pdf_path in sorted(UPLOAD_DIR.glob("**/*.pdf")):
            if not _structured_json_path_for(pdf_path).exists():
                missing_files.append(pdf_path.name)
        if not missing_files:
            return {"extracted": [], "errors": []}
        return self.extract_structured_invoices(missing_files)

    def extract_structured_invoices(self, filenames: list[str] | None = None) -> dict[str, Any]:
        targets: list[Path]
        if filenames is None:
            targets = sorted(UPLOAD_DIR.glob("**/*.pdf"))
        else:
            targets = []
            for filename in filenames:
                candidate = UPLOAD_DIR / Path(filename).name
                if candidate.exists():
                    targets.append(candidate)

        extracted: list[str] = []
        errors: list[str] = []
        for pdf_path in targets:
            try:
                record = self.extract_structured_invoice(pdf_path)
                extracted.append(str(record["json_file"]))
            except Exception as exc:
                errors.append(f"{pdf_path.name}: {exc}")

        return {
            "extracted": extracted,
            "errors": errors,
        }

    def extract_structured_invoice(self, pdf_path: Path) -> dict[str, Any]:
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        extracted_pages: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                extracted_pages.append(f"Page {index}\n{text.strip()}")

        invoice_text = "\n\n".join(extracted_pages).strip()
        if not invoice_text:
            raise ValueError("No extractable text found in the PDF.")

        raw_payload = get_extraction_chain().invoke(
            {
                "file_name": pdf_path.name,
                "page_count": page_count,
                "text": invoice_text,
            }
        )
        payload = _parse_llm_json(raw_payload)
        record = {
            "source_file": pdf_path.name,
            "json_file": _structured_json_path_for(pdf_path).name,
            "page_count": page_count,
            "extracted_at": _utc_now_iso(),
            "invoice_number": _normalize_whitespace(payload.get("invoice_number")),
            "issue_date": _normalize_whitespace(payload.get("issue_date")),
            "currency": _normalize_whitespace(payload.get("currency")),
            "seller": _coerce_party(payload.get("seller")),
            "client": _coerce_party(payload.get("client")),
            "items": _coerce_line_items(payload.get("items")),
            "tax_breakdown": _coerce_tax_breakdown(payload.get("tax_breakdown")),
            "totals": _coerce_totals(payload.get("totals")),
            "notes": _coerce_notes(payload.get("notes")),
        }

        output_path = _structured_json_path_for(pdf_path)
        output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    def list_structured_invoices(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for record in self._load_structured_records():
            totals = record.get("totals") or {}
            summaries.append(
                {
                    "source_file": record.get("source_file"),
                    "json_file": record.get("json_file"),
                    "invoice_number": record.get("invoice_number"),
                    "issue_date": record.get("issue_date"),
                    "currency": record.get("currency"),
                    "gross_total": totals.get("gross_total"),
                }
            )
        return summaries

    def get_structured_invoice(self, json_name: str) -> dict[str, Any]:
        safe_name = Path(json_name).name
        json_path = STRUCTURED_DIR / safe_name
        if not json_path.exists():
            raise FileNotFoundError(safe_name)
        return json.loads(json_path.read_text(encoding="utf-8"))

    def answer(self, question: str, session_id: str) -> dict[str, Any]:
        history = get_session_history(session_id)
        standalone_question = self._rewrite_question(question, history.messages)
        structured_sources = self._retrieve_structured_sources(standalone_question, limit=3)
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
        rag_sources = self._serialize_sources(
            docs,
            start_index=len(structured_sources) + 1,
        )
        sources = structured_sources + rag_sources

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

    def _retrieve_docs(
        self,
        query: str,
        limit: int,
        preferred_source: str | None = None,
    ) -> list[Any]:
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

    def _load_structured_records(self) -> list[dict[str, Any]]:
        valid_source_files = {path.name for path in UPLOAD_DIR.glob("**/*.pdf")}
        records: list[dict[str, Any]] = []
        for json_path in sorted(STRUCTURED_DIR.glob("*.json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("source_file") in valid_source_files:
                records.append(payload)
        return records

    def _retrieve_structured_sources(self, query: str, limit: int) -> list[dict[str, Any]]:
        records = self._load_structured_records()
        if not records:
            return []

        scored_records = [
            (_score_structured_record(query, record), record) for record in records
        ]
        scored_records.sort(
            key=lambda item: (
                item[0],
                str(item[1].get("invoice_number") or ""),
                str(item[1].get("source_file") or ""),
            ),
            reverse=True,
        )

        top_score = scored_records[0][0] if scored_records else 0
        if top_score >= 4:
            matching_records = [
                record for score, record in scored_records if score == top_score
            ][:limit]
        else:
            matching_records = [
                record for score, record in scored_records if score > 0
            ][:limit]
        if not matching_records and len(records) == 1:
            matching_records = records[:1]

        sources: list[dict[str, Any]] = []
        for index, record in enumerate(matching_records, start=1):
            sources.append(
                {
                    "id": f"S{index}",
                    "source": f"{record.get('source_file', 'invoice')} (structured)",
                    "page": 1,
                    "chunk_id": "structured",
                    "content": _build_structured_source_content(record),
                }
            )
        return sources

    def _build_index_signature(self) -> tuple[tuple[str, int, int], ...]:
        signature_items: list[tuple[str, int, int]] = []
        for filename in ("index.faiss", "index.pkl", "manifest.json"):
            path = INDEX_DIR / filename
            if path.exists():
                stat = path.stat()
                signature_items.append((filename, stat.st_mtime_ns, stat.st_size))
        return tuple(signature_items)

    def _serialize_sources(self, docs: list[Any], start_index: int = 1) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for index, doc in enumerate(docs, start=start_index):
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
