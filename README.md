# DeepResearch RAG Agent

DeepResearch RAG Agent is a full-stack Retrieval-Augmented Generation (RAG) application that combines:

- Local paper retrieval from Qdrant
- Fallback web search for freshness and missing evidence
- A LangGraph ReAct-style workflow for controlled routing
- FastAPI backend and Next.js frontend

The system is designed to answer with evidence-backed citations and graceful fallback behavior when confidence is low.

## Core Capabilities

- Agentic routing with explicit nodes (`react_plan`, `retrieve`, `web_search`, `validate_evidence`, `build_context`, `generate`)
- Prompt-driven web-only intent detection (`requires_web`) in planner and reflection steps
- Temporal query handling for phrases like "last month" and "this month"
- Section-aware ingestion for full-text PDFs
- Debug endpoints with trace-level graph visibility
- Logfire instrumentation for API, node-level, Pydantic, and OpenAI telemetry

## Architecture

The LangGraph workflow in `backend/graph.py` is:

1. `react_plan`
2. `retrieve` or `web_search`
3. `validate_evidence`
4. Loop back to `react_plan` if fallback is needed
5. `build_context`
6. `generate`

This gives a deterministic control loop with LLM-assisted planning and reflection.

## Repository Layout

```text
.
├── backend/
│   ├── main.py            # FastAPI API endpoints
│   ├── graph.py           # LangGraph state machine wiring
│   ├── nodes.py           # Agent nodes and routing logic
│   ├── configuration.py   # Model, token, env config helpers
│   ├── ingest.py          # PDF ingestion CLI
│   └── requirements.txt
├── frontend/              # Next.js app
├── docker-compose.yml     # Qdrant service
├── run-dev.ps1            # One-command local startup
└── run-dev.cmd            # Windows launcher
```

## Prerequisites

- Python 3.11+ (tested with 3.12)
- Node.js 18+
- Docker Desktop (for Qdrant)
- OpenAI-compatible API key
- Tavily API key (for web search)

## Quick Start (Recommended)

From repository root:

```powershell
./run-dev.ps1
```

or:

```cmd
run-dev.cmd
```

This script:

1. Starts Qdrant (`docker compose up -d qdrant`)
2. Creates `backend/.venv` (if missing)
3. Installs backend/frontend dependencies
4. Launches backend (`http://localhost:8000`) and frontend (`http://localhost:3000`)

## Manual Setup

### 1) Start Qdrant

```bash
docker compose up -d qdrant
```

### 2) Backend

```bash
cd backend
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3) Frontend

```bash
cd frontend
npm install
npm run dev
```

## Environment Variables

Create `backend/.env` and define the required keys.

Required for normal operation:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`
- `QDRANT_URL`
- `QDRANT_COLLECTION`

Common model settings:

- `OPENAI_CHAT_MODEL` (default fallback: `gpt-4o-mini`)
- `OPENAI_PLANNER_MODEL`
- `OPENAI_QUERY_REWRITE_MODEL`
- `OPENAI_REFLECTION_MODEL`
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)

Token/cost controls:

- `OPENAI_MAX_TOKENS`
- `OPENAI_ANSWER_MAX_TOKENS`
- `OPENAI_PLANNER_MAX_TOKENS`
- `OPENAI_QUERY_REWRITE_MAX_TOKENS`
- `OPENAI_REFLECTION_MAX_TOKENS`

Ingestion and section controls:

- `ARXIV_QUERY`
- `ARXIV_SORT_BY`
- `ARXIV_SORT_ORDER`
- `ARXIV_CONTENT_MODE`
- `ARXIV_MAX_PDF_PAGES`
- `ARXIV_PDF_TIMEOUT_SECONDS`
- `ARXIV_CHUNK_SIZE`
- `ARXIV_CHUNK_OVERLAP`
- `ARXIV_INCLUDE_SECTIONS`
- `ARXIV_EXCLUDE_SECTIONS`
- `ARXIV_FALLBACK_SECTION_KEYWORDS`

Evidence/temporal checks:

- `MIN_TEMPORAL_MATCHED_WEB_DOCS`

## Ingesting ArXiv Data

From `backend/`:

```bash
python ingest.py --limit 200 --query "cat:cs.AI OR cat:cs.LG" --content-mode fulltext
```

Useful flags:

- `--limit`
- `--query`
- `--sort-by` (`Relevance`, `LastUpdatedDate`, `SubmittedDate`)
- `--sort-order` (`Ascending`, `Descending`)
- `--content-mode` (`abstract`, `fulltext`)
- `--max-pdf-pages`
- `--chunk-size`, `--chunk-overlap`

## API Endpoints

- `GET /health`
- `POST /chat`
- `POST /chat/debug`
- `POST /chat/stream-debug`
- `POST /upload-pdf`

Example debug request:

```bash
curl -X POST http://localhost:8000/chat/debug \
	-H "Content-Type: application/json" \
	-d '{"message":"Compare DPO and RLHF based on research findings."}'
```

## Observability

The backend and stream demo include Logfire instrumentation:

- FastAPI request spans
- Node-level graph spans (`node.react_plan`, etc.)
- OpenAI and Pydantic instrumentation

If Logfire configuration fails, the API still runs.

## Evaluation / Streaming Debug

Run the stream demo from `backend/`:

```bash
python stream_demo.py "what is retrieval augmented generation"
```

Run eval mode:

```bash
python stream_demo.py --run-eval --llm-judge
```

## Troubleshooting

### ImportError: cannot import name 'pack' from 'struct'

Cause: local module shadowing Python stdlib `struct`.

Status in this repo: fixed by moving helpers to `graph_utils.py`.

If you still see it, clear stale bytecode in `backend/__pycache__/` and restart the server.

### 402 / insufficient credits / max token budget

Lower token limits and/or use cheaper model settings:

- `OPENAI_CHAT_MODEL=gpt-4o-mini`
- reduce `OPENAI_*_MAX_TOKENS`

## Notes

- CORS currently allows `http://localhost:3000`.
- Web-only requests are handled via `requires_web` intent from planner/reflection logic plus graph safeguards.
