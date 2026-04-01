import json
from pathlib import Path
from typing import Any

from backend.rag_config import MANIFEST_PATH, utc_now_iso


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
