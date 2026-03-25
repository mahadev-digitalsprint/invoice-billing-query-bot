import asyncio
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_service import UPLOAD_DIR, service

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str = Field(default="default", min_length=1)


@asynccontextmanager
async def lifespan(_: FastAPI):
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


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(INDEX_FILE)


@app.get("/health")
async def health():
    dashboard = service.get_dashboard_data()
    return {"status": "ok", **dashboard}


@app.get("/dashboard")
async def dashboard():
    return service.get_dashboard_data()


@app.get("/files")
async def list_files():
    return {"files": service.list_pdf_files()}


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
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
        **summary,
    }


@app.post("/chat")
async def chat(request: ChatRequest):
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


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    from rag_service import get_history_messages

    return {"session_id": session_id, "messages": get_history_messages(session_id)}


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    from rag_service import clear_session_history

    clear_session_history(session_id)
    return {"message": f"History cleared for session '{session_id}'."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
