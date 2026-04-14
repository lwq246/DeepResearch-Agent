# DeepResearch-Agent: Autonomous PDF and Web Research Orchestrator

DeepResearch-Agent is a Retrieval-Augmented Generation (RAG) system that unifies private document retrieval and live web intelligence in one assistant.

Traditional RAG pipelines often fail when local documents are stale, missing a specific fact, or lacking one side of a comparison. This project addresses that with a LangGraph ReAct-style loop that evaluates evidence quality, performs fallback web retrieval when needed, and enforces evidence guards before answering.

## Key Agentic Capabilities

- Autonomous routing: A planner node decides whether to retrieve locally, perform web search, or proceed to synthesis.
- Self-correcting fallback: Web search is triggered when local confidence is weak (for example, low relevance or missing evidence).
- Comparison coverage guard: For queries like "A vs B", the graph checks whether evidence covers both entities.
- Section-aware PDF processing: PDF text is split into semantic sections and chunked per section for better retrieval precision.

## Architecture

### Reasoning Loop (LangGraph)

The workflow is implemented as a state graph with explicit reasoning steps:

- `react_plan`: Classifies query intent and chooses next action.
- `retrieve`: Vector retrieval from Qdrant.
- `web_search`: Tavily web retrieval for missing or time-sensitive evidence.
- `validate_evidence`: Checks evidence sufficiency (including comparison/entity coverage logic).
- `build_context`: Deduplicates and prioritizes documents.
- `generate`: Produces grounded answer with inline citations.

## Project Structure

```text
.
├── backend/            # FastAPI API, LangGraph workflow, ingestion logic
├── frontend/           # Next.js 14 chat UI
└── docker-compose.yml  # Local Qdrant service
```
