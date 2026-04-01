# Portfolio AI with RAG

A beginner friendly Retrieval-Augmented Generation assistant built on FastAPI + ChromaDB with ingestion pipeline, relevance filtering, health checks, and automated test coverage.

## What this project does

- Ingests structured and unstructured files from a local knowledge base
- Chunks and embeds content using OpenAI embeddings
- Stores vectors in ChromaDB
- Retrieves only relevant chunks for each user question
- Sends grounded context to a FastAPI backend for response generation
- Provides a Streamlit chat UI for local testing

## Engineering highlights

- Startup configuration validation (`OPENAI_API_KEY` check)
- Retrieval and LLM error boundaries with proper HTTP status codes
- Health endpoint for runtime checks
- Knowledge base change detection in startup script

## Stack

- Python
- FastAPI
- Streamlit
- ChromaDB
- OpenAI API
- pdfplumber

## Project structure

- [api.py](api.py): FastAPI chat endpoint with memory per session
- [retriever.py](retriever.py): vector retrieval with relevance threshold
- [ingest.py](ingest.py): file parsing, chunking, embedding, and persistence
- [chat_ui.py](chat_ui.py): Streamlit chat interface
- [inspect_db.py](inspect_db.py): utility to inspect stored vectors
- [run_app.sh](run_app.sh): one-command local runner with smart ingestion
- [knowledge_base/](knowledge_base): source files for RAG grounding

## Quick start

1) Create and activate a virtual environment

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

2) Create a `.env` file in the project root

```env
OPENAI_API_KEY=your_openai_api_key
CHROMA_DB_PATH=./chroma_db
RELEVANCE_THRESHOLD=1.5
```

3) Add knowledge files to [knowledge_base/](knowledge_base)

Supported formats: `.json`, `.pdf`, `.txt`, `.md`

4) Build the vector database

```bash
python3 ingest.py
```

5) Start backend API

```bash
uvicorn api:app --reload --port 8001
```

6) Start chat UI

```bash
streamlit run chat_ui.py
```

Open http://localhost:8501

## One-command run

You can run everything with:

```bash
./run_app.sh
```

Ingestion modes:

- `./run_app.sh` → auto-ingest only when `knowledge_base/` changes
- `./run_app.sh --ingest` → force ingestion every run
- `./run_app.sh --skip-ingest` → skip ingestion

## API endpoints

- `POST /chat`
- `GET /health`

Example health response:

```json
{
	"status": "ok",
	"model": "gpt-4o-mini",
	"collection": "sandesh_knowledge"
}
```

## Notes

- Re-run ingestion whenever files in [knowledge_base/](knowledge_base) change.
- Each chat session uses a unique in-memory session history.
- Responses are grounded by retrieval first, then generation.

## Why I built this
I built this to demonstrate practical backend-focused RAG engineering: resilient ingestion, controlled retrieval, predictable API behavior, and reproducible local workflows.


## Disclaimer:
This project was built by me, with AI-assisted support from GitHub Copilot for brainstorming, code cleanup, refactoring suggestions, and documentation improvements. I reviewed, tested, and finalized all implementation decisions.