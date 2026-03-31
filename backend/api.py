import asyncio
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.rag_service import (
    UPLOAD_DIR,
    clear_session_history,
    get_history_messages,
    service,
)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
FRONTEND_DIST_DIR = BASE_DIR.parent / "frontend" / "dist"
FRONTEND_INDEX_FILE = FRONTEND_DIST_DIR / "index.html"
INDEX_FILE = STATIC_DIR / "index.html"


class ChatRequest(BaseModel):
    """Represents a chat question plus the browser session it belongs to."""

    question: str = Field(..., min_length=1)
    session_id: str = Field(default="default", min_length=1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Runs one-time startup work before the API starts serving requests."""

    await asyncio.to_thread(service.ensure_default_index)
    yield


app = FastAPI(
    title="Document Parser Assistant",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount(
    "/assets",
    StaticFiles(directory=FRONTEND_DIST_DIR / "assets", check_dir=False),
    name="frontend-assets",
)


@app.get("/", include_in_schema=False)
async def index():
    """Serves the built frontend when available, otherwise the static fallback page."""

    if FRONTEND_INDEX_FILE.exists():
        return FileResponse(FRONTEND_INDEX_FILE)
    return FileResponse(INDEX_FILE)


@app.get("/api/health")
async def health():
    """Returns a health response along with current dashboard counters."""

    dashboard = service.get_dashboard_data()
    return {"status": "ok", **dashboard}


@app.get("/api/dashboard")
async def dashboard():
    """Returns backend metrics used by the dashboard cards."""

    return service.get_dashboard_data()


@app.get("/api/files")
async def list_files():
    """Lists PDF files currently available for indexing and retrieval."""

    return {"files": service.list_pdf_files()}


@app.get("/api/invoices")
async def list_invoices():
    """Lists saved structured invoice summaries for the frontend sidebar."""

    return {"invoices": service.list_structured_invoices()}


@app.get("/api/invoices/{json_name}")
async def get_invoice(json_name: str):
    """Returns one structured invoice JSON file by name."""

    try:
        return service.get_structured_invoice(json_name)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Structured invoice JSON not found: {json_name}",
        ) from exc


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Saves uploaded PDFs, extracts structured data, and rebuilds the vector index."""

    saved: list[str] = []
    errors: list[str] = []

    for upload in files:
        filename = upload.filename or "unnamed.pdf"
        if not filename.lower().endswith(".pdf"):
            errors.append(f"{filename}: skipped because it is not a PDF.")
            await upload.close()
            continue

        destination = UPLOAD_DIR / Path(filename).name
        with destination.open("wb") as file_handle:
            shutil.copyfileobj(upload.file, file_handle)
        await upload.close()
        saved.append(destination.name)

    if not saved:
        raise HTTPException(
            status_code=400,
            detail="No valid PDF files were uploaded.",
        )

    try:
        structured_summary = await asyncio.to_thread(
            service.extract_structured_invoices,
            saved,
        )
        summary = await asyncio.to_thread(service.rebuild_vectorstore)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Index rebuild failed: {exc}",
        ) from exc

    return {
        "saved": saved,
        "errors": errors,
        "message": f"Indexed {summary['chunk_count']} chunks from {summary['file_count']} PDF files.",
        "structured_files": structured_summary["extracted"],
        "structured_errors": structured_summary["errors"],
        **summary,
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Answers a user question using the indexed invoices and chat history."""

    if not service.vectorstore_ready():
        raise HTTPException(
            status_code=400,
            detail="No indexed documents found. Upload at least one PDF first.",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = await asyncio.to_thread(service.answer, question, request.session_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat request failed: {exc}",
        ) from exc

    return {"session_id": request.session_id, **result}


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """Returns chat messages stored for a given session id."""

    return {"session_id": session_id, "messages": get_history_messages(session_id)}


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    """Clears the stored chat messages for a given session id."""

    clear_session_history(session_id)
    return {"message": f"History cleared for session '{session_id}'."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api:app", host="0.0.0.0", port=8001, reload=True)
