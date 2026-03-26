# Invoice Billing Query Bot

A clean full-stack project with:

- `backend/` for FastAPI, RAG, invoice extraction, and backend runtime data
- `frontend/` for the React + Vite + TypeScript UI
- structured invoice JSON extraction on upload
- FAISS-backed document retrieval with Gemini answering

## Structure

```text
backend/
  api.py
  rag_service.py
  requirements.txt
  uploads/
  structured_data/
  faiss_index/
  static/
frontend/
  src/
  package.json
```

## Requirements

- Python 3.12 or 3.13 recommended
- Node.js 18+
- `GEMINI_API_KEY` in `.env`

## Setup Commands

Backend install:

```bash
source venv/bin/activate
pip install -r backend/requirements.txt
```

Frontend install:

```bash
cd frontend
npm install
```

## Run Commands

Start backend:

```bash
source venv/bin/activate
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8001 --reload
```

Start frontend:

```bash
cd frontend
source ../venv/bin/activate
source ~/.zshrc
npm run dev
```

Build frontend for FastAPI:

```bash
cd frontend
npm run build
```

Compile-check backend:

```bash
source venv/bin/activate
python -m compileall backend
```

## URLs

- Frontend dev UI: `http://127.0.0.1:5173`
- Backend API: `http://127.0.0.1:8001`
- Swagger docs: `http://127.0.0.1:8001/docs`

## API Endpoints

- `GET /api/health`
- `GET /api/dashboard`
- `GET /api/files`
- `GET /api/invoices`
- `GET /api/invoices/{json_file}`
- `POST /api/upload`
- `POST /api/chat`
- `GET /api/history/{session_id}`
- `DELETE /api/history/{session_id}`

## Invoice Flow

When you upload a PDF invoice, the backend:

1. extracts structured invoice fields with Gemini
2. saves the JSON in `backend/structured_data/`
3. rebuilds the FAISS index in `backend/faiss_index/`
4. answers chat from structured data first, then raw PDF evidence

## Notes

- The old Streamlit code has been removed.
- Backend code and backend-owned files now live under `backend/`.
