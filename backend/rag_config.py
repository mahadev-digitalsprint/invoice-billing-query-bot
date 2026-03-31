import re
from datetime import datetime, timezone
from pathlib import Path

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


def ensure_directories() -> None:
    """Creates the upload, index, and structured-data folders if they do not exist."""

    UPLOAD_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)
    STRUCTURED_DIR.mkdir(exist_ok=True)


def utc_now_iso() -> str:
    """Returns the current UTC timestamp in ISO format for metadata fields."""

    return datetime.now(timezone.utc).isoformat()
