import json
from pathlib import Path
from typing import Any

from backend.rag_config import STRUCTURED_DIR, UPLOAD_DIR, utc_now_iso
from backend.rag_utils import (
    build_structured_source_content,
    coerce_common_billing_policies,
    coerce_line_items,
    coerce_notes,
    coerce_party,
    coerce_projects,
    coerce_sample_questions,
    coerce_tax_breakdown,
    coerce_totals,
    normalize_whitespace,
    score_structured_record,
    structured_json_path_for,
)


def build_structured_record(
    pdf_path: Path,
    payload: dict[str, Any],
    page_count: int,
) -> dict[str, Any]:
    """Converts raw extraction payload into the final structured invoice JSON shape."""

    return {
        "source_file": pdf_path.name,
        "json_file": structured_json_path_for(pdf_path).name,
        "page_count": page_count,
        "extracted_at": utc_now_iso(),
        "document_type": normalize_whitespace(payload.get("document_type")),
        "project_name": normalize_whitespace(payload.get("project_name")),
        "invoice_number": normalize_whitespace(payload.get("invoice_number")),
        "issue_date": normalize_whitespace(payload.get("issue_date")),
        "billing_period": normalize_whitespace(payload.get("billing_period")),
        "currency": normalize_whitespace(payload.get("currency")),
        "payment_terms": normalize_whitespace(payload.get("payment_terms")),
        "status": normalize_whitespace(payload.get("status")),
        "payment_date": normalize_whitespace(payload.get("payment_date")),
        "due_date": normalize_whitespace(payload.get("due_date")),
        "overdue_details": normalize_whitespace(payload.get("overdue_details")),
        "follow_up_actions": coerce_notes(payload.get("follow_up_actions")),
        "seller": coerce_party(payload.get("seller")),
        "client": coerce_party(payload.get("client")),
        "items": coerce_line_items(payload.get("items")),
        "tax_breakdown": coerce_tax_breakdown(payload.get("tax_breakdown")),
        "totals": coerce_totals(payload.get("totals")),
        "projects": coerce_projects(payload.get("projects")),
        "common_billing_policies": coerce_common_billing_policies(
            payload.get("common_billing_policies")
        ),
        "sample_questions": coerce_sample_questions(payload.get("sample_questions")),
        "notes": coerce_notes(payload.get("notes")),
    }


def save_structured_record(pdf_path: Path, record: dict[str, Any]) -> None:
    """Writes one structured invoice record to disk beside the uploaded PDF set."""

    structured_json_path_for(pdf_path).write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )


def load_structured_records() -> list[dict[str, Any]]:
    """Loads valid structured invoice JSON files that still match existing uploaded PDFs."""

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


def list_structured_invoices() -> list[dict[str, Any]]:
    """Builds lightweight invoice summaries for the frontend extracted-JSON list."""

    summaries: list[dict[str, Any]] = []
    for record in load_structured_records():
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


def get_structured_invoice(json_name: str) -> dict[str, Any]:
    """Returns one structured invoice record from disk by safe filename."""

    json_path = STRUCTURED_DIR / Path(json_name).name
    if not json_path.exists():
        raise FileNotFoundError(json_path.name)
    return json.loads(json_path.read_text(encoding="utf-8"))


def retrieve_structured_sources(query: str, limit: int) -> list[dict[str, Any]]:
    """Finds the best structured records for a question and formats them as RAG sources."""

    records = load_structured_records()
    if not records:
        return []

    scored_records = [(score_structured_record(query, record), record) for record in records]
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
        matching_records = [record for score, record in scored_records if score == top_score][:limit]
    else:
        matching_records = [record for score, record in scored_records if score > 0][:limit]
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
                "content": build_structured_source_content(record),
            }
        )
    return sources
