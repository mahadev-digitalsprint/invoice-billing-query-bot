import json
import re
from pathlib import Path
from typing import Any

from backend.rag_config import MANIFEST_PATH, STRUCTURED_DIR, STRUCTURED_QUERY_STOPWORDS, utc_now_iso


def normalize_whitespace(value: Any) -> str | None:
    """Collapses repeated whitespace and returns None for empty values."""

    if value is None:
        return None
    collapsed = re.sub(r"\s+", " ", str(value)).strip()
    return collapsed or None


def clean_llm_json(text: str) -> str:
    """Strips markdown fences and extra text so the LLM response can be parsed as JSON."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned


def parse_llm_json(text: str) -> dict[str, Any]:
    """Parses cleaned LLM output and ensures the result is a JSON object."""

    payload = json.loads(clean_llm_json(text))
    if not isinstance(payload, dict):
        raise ValueError("Structured extraction did not return a JSON object.")
    return payload


def coerce_party(value: Any) -> dict[str, Any]:
    """Normalizes a seller or client object into the expected invoice shape."""

    party = value if isinstance(value, dict) else {}
    return {
        "name": normalize_whitespace(party.get("name")),
        "address": normalize_whitespace(party.get("address")),
        "tax_id": normalize_whitespace(party.get("tax_id")),
        "iban": normalize_whitespace(party.get("iban")),
    }


def coerce_line_items(items: Any) -> list[dict[str, Any]]:
    """Normalizes invoice line items into clean string-based records."""

    if not isinstance(items, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "line_number": normalize_whitespace(item.get("line_number")),
                "description": normalize_whitespace(item.get("description")),
                "quantity": normalize_whitespace(item.get("quantity")),
                "unit": normalize_whitespace(item.get("unit")),
                "net_price": normalize_whitespace(item.get("net_price")),
                "net_amount": normalize_whitespace(item.get("net_amount")),
                "vat_rate": normalize_whitespace(item.get("vat_rate")),
                "gross_amount": normalize_whitespace(item.get("gross_amount")),
            }
        )
    return normalized


def coerce_tax_breakdown(items: Any) -> list[dict[str, Any]]:
    """Normalizes VAT or tax rows into a consistent list of dictionaries."""

    if not isinstance(items, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "vat_rate": normalize_whitespace(item.get("vat_rate")),
                "net_amount": normalize_whitespace(item.get("net_amount")),
                "vat_amount": normalize_whitespace(item.get("vat_amount")),
                "gross_amount": normalize_whitespace(item.get("gross_amount")),
            }
        )
    return normalized


def coerce_totals(value: Any) -> dict[str, Any]:
    """Normalizes top-level invoice total fields."""

    totals = value if isinstance(value, dict) else {}
    return {
        "net_total": normalize_whitespace(totals.get("net_total")),
        "vat_total": normalize_whitespace(totals.get("vat_total")),
        "gross_total": normalize_whitespace(totals.get("gross_total")),
    }


def coerce_project_billing_summary(value: Any) -> dict[str, Any]:
    """Normalizes aggregated project billing totals for knowledge-base style invoices."""

    summary = value if isinstance(value, dict) else {}
    return {
        "total_invoiced": normalize_whitespace(summary.get("total_invoiced")),
        "total_paid": normalize_whitespace(summary.get("total_paid")),
        "outstanding": normalize_whitespace(summary.get("outstanding")),
    }


def coerce_notes(value: Any) -> list[str]:
    """Normalizes a list of notes or follow-up strings and removes empty values."""

    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        note = normalize_whitespace(item)
        if note:
            normalized.append(note)
    return normalized


def coerce_project_invoices(items: Any) -> list[dict[str, Any]]:
    """Normalizes invoice records nested under a project section."""

    if not isinstance(items, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "invoice_number": normalize_whitespace(item.get("invoice_number")),
                "invoice_date": normalize_whitespace(item.get("invoice_date")),
                "billing_period": normalize_whitespace(item.get("billing_period")),
                "amount": normalize_whitespace(item.get("amount")),
                "description": normalize_whitespace(item.get("description")),
                "payment_terms": normalize_whitespace(item.get("payment_terms")),
                "status": normalize_whitespace(item.get("status")),
                "payment_date": normalize_whitespace(item.get("payment_date")),
                "due_date": normalize_whitespace(item.get("due_date")),
                "overdue_details": normalize_whitespace(item.get("overdue_details")),
                "follow_up_actions": coerce_notes(item.get("follow_up_actions")),
                "notes": coerce_notes(item.get("notes")),
            }
        )
    return normalized


def coerce_projects(items: Any) -> list[dict[str, Any]]:
    """Normalizes project-level rollups for multi-invoice billing documents."""

    if not isinstance(items, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "project_name": normalize_whitespace(item.get("project_name")),
                "client_name": normalize_whitespace(item.get("client_name")),
                "invoices": coerce_project_invoices(item.get("invoices")),
                "billing_summary": coerce_project_billing_summary(
                    item.get("billing_summary")
                ),
                "observations": coerce_notes(item.get("observations")),
            }
        )
    return normalized


def coerce_common_billing_policies(value: Any) -> dict[str, Any]:
    """Normalizes common billing policy fields extracted from reference PDFs."""

    policies = value if isinstance(value, dict) else {}
    return {
        "payment_terms": normalize_whitespace(policies.get("payment_terms")),
        "overdue_definition": normalize_whitespace(
            policies.get("overdue_definition")
        ),
        "follow_up_process": coerce_notes(policies.get("follow_up_process")),
        "dispute_handling": normalize_whitespace(policies.get("dispute_handling")),
    }


def coerce_sample_questions(items: Any) -> list[dict[str, Any]]:
    """Normalizes example question-answer pairs extracted from billing reference files."""

    if not isinstance(items, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = normalize_whitespace(item.get("question"))
        answer = normalize_whitespace(item.get("answer"))
        if question or answer:
            normalized.append({"question": question, "answer": answer})
    return normalized


def flatten_for_search(value: Any) -> str:
    """Flattens nested structured data into one searchable text blob."""

    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(flatten_for_search(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(flatten_for_search(item) for item in value)
    return str(value)


def structured_json_path_for(pdf_path: Path) -> Path:
    """Maps an uploaded PDF path to the matching structured JSON output path."""

    return STRUCTURED_DIR / f"{pdf_path.stem}.json"


def tokenize_query(query: str) -> set[str]:
    """Extracts useful search tokens from a user question while skipping common stopwords."""

    tokens = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
    return {token for token in tokens if token not in STRUCTURED_QUERY_STOPWORDS}


def score_structured_record(query: str, record: dict[str, Any]) -> int:
    """Scores how well one structured invoice record matches a user question."""

    lowered_query = query.lower()
    search_blob = flatten_for_search(record).lower()
    score = 0

    for token in tokenize_query(query):
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


def build_structured_source_content(record: dict[str, Any]) -> str:
    """Builds a readable text summary of a structured record for LLM context."""

    seller = record.get("seller") or {}
    client = record.get("client") or {}
    totals = record.get("totals") or {}
    lines = [
        "Structured invoice data",
        f"document_type: {record.get('document_type') or 'unknown'}",
        f"project_name: {record.get('project_name') or 'unknown'}",
        f"invoice_number: {record.get('invoice_number') or 'unknown'}",
        f"issue_date: {record.get('issue_date') or 'unknown'}",
        f"billing_period: {record.get('billing_period') or 'unknown'}",
        f"currency: {record.get('currency') or 'unknown'}",
        f"payment_terms: {record.get('payment_terms') or 'unknown'}",
        f"status: {record.get('status') or 'unknown'}",
        f"payment_date: {record.get('payment_date') or 'unknown'}",
        f"due_date: {record.get('due_date') or 'unknown'}",
        f"overdue_details: {record.get('overdue_details') or 'unknown'}",
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

    follow_up_actions = record.get("follow_up_actions") or []
    if follow_up_actions:
        lines.append("follow_up_actions: " + " | ".join(follow_up_actions[:4]))

    projects = record.get("projects") or []
    if projects:
        project_summaries: list[str] = []
        for project in projects[:6]:
            summary = project.get("billing_summary") or {}
            project_parts = [
                f"project {project.get('project_name') or 'unknown'}",
                f"client {project.get('client_name') or 'unknown'}",
            ]
            if summary.get("total_invoiced"):
                project_parts.append(f"total invoiced {summary['total_invoiced']}")
            if summary.get("total_paid"):
                project_parts.append(f"total paid {summary['total_paid']}")
            if summary.get("outstanding"):
                project_parts.append(f"outstanding {summary['outstanding']}")
            invoices = project.get("invoices") or []
            if invoices:
                invoice_summaries: list[str] = []
                for invoice in invoices[:6]:
                    invoice_parts = [
                        invoice.get("invoice_number"),
                        invoice.get("invoice_date"),
                        invoice.get("billing_period"),
                        invoice.get("amount"),
                        invoice.get("status"),
                    ]
                    if invoice.get("overdue_details"):
                        invoice_parts.append(f"overdue {invoice['overdue_details']}")
                    if invoice.get("notes"):
                        invoice_parts.append(
                            "notes " + "; ".join(invoice["notes"][:2])
                        )
                    if invoice.get("follow_up_actions"):
                        invoice_parts.append(
                            "follow up "
                            + "; ".join(invoice["follow_up_actions"][:2])
                        )
                    invoice_summaries.append(
                        ", ".join(part for part in invoice_parts if part)
                    )
                project_parts.append("invoices: " + " | ".join(invoice_summaries))
            observations = project.get("observations") or []
            if observations:
                project_parts.append("observations: " + " | ".join(observations[:3]))
            project_summaries.append(" ; ".join(project_parts))
        lines.append("projects: " + " || ".join(project_summaries))

    policies = record.get("common_billing_policies") or {}
    policy_parts = []
    if policies.get("payment_terms"):
        policy_parts.append(f"payment terms {policies['payment_terms']}")
    if policies.get("overdue_definition"):
        policy_parts.append(f"overdue definition {policies['overdue_definition']}")
    if policies.get("follow_up_process"):
        policy_parts.append(
            "follow up process " + " | ".join(policies["follow_up_process"][:4])
        )
    if policies.get("dispute_handling"):
        policy_parts.append(f"dispute handling {policies['dispute_handling']}")
    if policy_parts:
        lines.append("common_billing_policies: " + " ; ".join(policy_parts))

    sample_questions = record.get("sample_questions") or []
    if sample_questions:
        qa_summaries = []
        for sample in sample_questions[:5]:
            qa_summaries.append(
                f"Q: {sample.get('question') or 'unknown'} A: {sample.get('answer') or 'unknown'}"
            )
        lines.append("sample_questions: " + " || ".join(qa_summaries))

    return "\n".join(lines)


def normalize_page_number(metadata: dict[str, Any]) -> int:
    """Converts zero-based page metadata into a human-friendly one-based page number."""

    raw_page = metadata.get("page", 0)
    try:
        return int(raw_page) + 1
    except (TypeError, ValueError):
        return 1


def serialize_sources(docs: list[Any], start_index: int = 1) -> list[dict[str, Any]]:
    """Converts retrieved LangChain documents into the source format returned to the UI."""

    sources: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=start_index):
        sources.append(
            {
                "id": f"S{index}",
                "source": doc.metadata.get("source", "document"),
                "page": doc.metadata.get("page_number")
                or normalize_page_number(doc.metadata),
                "chunk_id": doc.metadata.get("chunk_id"),
                "content": doc.page_content.strip(),
            }
        )
    return sources


def build_context(sources: list[dict[str, Any]]) -> str:
    """Formats all retrieved sources into one prompt-ready context block."""

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


def build_index_signature(index_dir: Path) -> tuple[tuple[str, int, int], ...]:
    """Builds a lightweight signature used to detect when the saved FAISS index changed."""

    signature_items: list[tuple[str, int, int]] = []
    for filename in ("index.faiss", "index.pkl", "manifest.json"):
        path = index_dir / filename
        if path.exists():
            stat = path.stat()
            signature_items.append((filename, stat.st_mtime_ns, stat.st_size))
    return tuple(signature_items)


def write_manifest(files: list[Path], chunk_count: int) -> None:
    """Writes index metadata such as file count, chunk count, and build time."""

    payload = {
        "built_at": utc_now_iso(),
        "chunk_count": chunk_count,
        "file_count": len(files),
        "files": [path.name for path in files],
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest() -> dict[str, Any]:
    """Loads saved index metadata, returning an empty object if it is missing or invalid."""

    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
