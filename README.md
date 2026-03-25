# Document Parser

A document Q&A app for PDF files with:

- FastAPI backend
- Built-in web UI served by FastAPI
- Optional Streamlit UI
- FAISS vector search
- Gemini-based answering with source grounding

## Requirements

- Python 3.12 or 3.13 recommended
- Windows PowerShell
- A valid `GEMINI_API_KEY`

## Setup

### 1. Open the project

```powershell
cd C:\Users\wellcome\Desktop\Document_Parser
```

### 2. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Add your Gemini API key

Create or update `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

## Run The FastAPI App

This is the main version now. It serves both the API and the browser UI.

```powershell
.\venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8010
```

Open:

[http://127.0.0.1:8010](http://127.0.0.1:8010)

Useful endpoints:

- `/`
- `/health`
- `/dashboard`
- `/files`
- `/docs`

## Run The Streamlit App

If you want the Streamlit interface instead:

```powershell
.\venv\Scripts\streamlit.exe run streamlit_app.py --server.headless true --server.port 8501
```

Open:

[http://127.0.0.1:8501](http://127.0.0.1:8501)

## Common Commands

### Start FastAPI on the default project port

```powershell
.\venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001
```

### Start FastAPI with auto reload

```powershell
.\venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8010 --reload
```

### Start Streamlit

```powershell
.\venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8501
```

### Compile-check Python files

```powershell
.\venv\Scripts\python.exe -m compileall api.py app.py main.py rag_service.py streamlit_app.py
```

## Project Files

- `api.py`: FastAPI app and routes
- `rag_service.py`: indexing, retrieval, history, and grounded answer pipeline
- `streamlit_app.py`: Streamlit UI
- `static/`: FastAPI-served frontend UI
- `uploads/`: uploaded PDF files
- `faiss_index/`: saved FAISS index

## Troubleshooting

### Port already in use

Run on a different port:

```powershell
.\venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8010
```

or:

```powershell
.\venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8502
```

### FastAPI shows JSON instead of UI

Make sure you are opening the FastAPI root page:

[http://127.0.0.1:8010/](http://127.0.0.1:8010/)

and not `/health`.

### No answers from the model

Check:

- `.env` contains a valid `GEMINI_API_KEY`
- internet access is available for Gemini calls
- PDFs have been uploaded and indexed

### Rebuild the index

Upload PDFs again from the UI, or restart the app and re-index through the interface.

## Notes

- FastAPI is the primary app now.
- Streamlit is available as an optional interface.
- The app stores uploaded files in `uploads/` and the vector index in `faiss_index/`.
