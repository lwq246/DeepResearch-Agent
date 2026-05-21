import logging
import os
from typing import Any

import logfire
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from .configuration import bool_env
from .configuration import int_env
from .configuration import optional_env
from .configuration import required_env
from .graph import app as lang_graph
from .ingest.authors import build_author_check_llm
from .ingest.authors import extract_authors_from_first_page
from .ingest.chunking import DEFAULT_EXCLUDE_REFERENCES
from .ingest.chunking import chunk_sections_from_extracted_text
from .ingest.documents import build_section_documents
from .ingest.sources import extract_pdf_text_with_pymupdf4llm_from_bytes
from .models.api_models import ChatDebugResponse
from .models.api_models import ChatRequest
from .models.api_models import ChatResponse
from .models.api_models import ChatStreamDebugResponse
from .models.api_models import NodeUpdate
from .models.api_models import UploadPdfResponse
from .configuration import parse_max_pdf_pages
from .retrieval_config import UPLOAD_DEBUG_CONFIG


app = FastAPI(title="ArXiv RAG Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def configure_logfire(app_instance: FastAPI) -> None:
    enabled = bool_env("LOGFIRE_ENABLED")
    if not enabled:
        os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
        return

    service_name = required_env("LOGFIRE_SERVICE_NAME")
    try:
        logfire.configure(service_name=service_name)
        logfire.instrument_fastapi(app_instance)
        logfire.instrument_pydantic()
        logfire.instrument_openai()
        logfire.info("logfire_configured", service_name=service_name)
    except Exception as exc:
        # Keep the API available if observability configuration fails.
        os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
        print(f"[observability] logfire setup failed: {exc}", flush=True)


configure_logfire(app)
logger = logging.getLogger(__name__)


def openai_client_kwargs() -> dict[str, str]:
    base_url = optional_env("OPENAI_BASE_URL")
    if base_url is None:
        return {}
    return {"base_url": base_url}


def extract_pdf_text(pdf_bytes: bytes, max_pages: int | None) -> str:
    return extract_pdf_text_with_pymupdf4llm_from_bytes(pdf_bytes=pdf_bytes, max_pages=max_pages)


def build_upload_documents(
    filename: str,
    source_value: str,
    extracted_text: str,
    authors: list[str],
    min_chunk_chars: int,
    max_chunk_chars: int,
    chunk_overlap_chars: int,
    exclude_references: bool,
) -> list[Document]:
    sections = chunk_sections_from_extracted_text(
        extracted_text=extracted_text,
        min_chars=min_chunk_chars,
        max_chars=max_chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        exclude_references=exclude_references,
    )

    base_metadata = {
        "title": filename,
        "paper_id": filename,
        "source_url": source_value,
        "authors": authors,
    }
    return build_section_documents(base_metadata=base_metadata, sections=sections)


def print_upload_summary(
    filename: str,
    source_value: str,
    documents: list[Document],
    collection_name: str,
) -> None:
    preview_chunks = max(0, UPLOAD_DEBUG_CONFIG.preview_chunks)
    preview_chars = max(40, UPLOAD_DEBUG_CONFIG.preview_chars)

    print(
        f"[upload-pdf] filename={filename} collection={collection_name} "
        f"source={source_value} chunks={len(documents)}",
        flush=True,
    )

    for document in documents[:preview_chunks]:
        section_name = document.metadata.get("section_title", "?")
        chunk_index = document.metadata.get("section_index", "?")
        preview_text = document.page_content.replace("\n", " ").strip()
        if len(preview_text) > preview_chars:
            preview_text = preview_text[:preview_chars] + "..."
        print(
            f"[upload-pdf] section={section_name} chunk={chunk_index} text={preview_text}",
            flush=True,
        )

    remaining = len(documents) - preview_chunks
    if remaining > 0:
        print(f"[upload-pdf] ... {remaining} more chunks not shown", flush=True)


def build_initial_state(message: str) -> dict[str, Any]:
    return {
        "question": message,
        "documents": [],
        "generation": "",
        "requires_web": False,
        "fallback": False,
        "top_score": 0.0,
        "evidence_ok": False,
        "web_attempts": 0,
        "react_step": 0,
        "next_action": "",
        "react_trace": [],
    }


def summarize_node_update(update: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in update.items():
        if key == "documents" and isinstance(value, list):
            summary["documents_count"] = len(value)
            origin_counts: dict[str, int] = {}
            for item in value:
                if not isinstance(item, dict):
                    continue
                origin = str(item.get("origin", "unknown"))
                origin_counts[origin] = origin_counts.get(origin, 0) + 1
            if origin_counts:
                summary["documents_by_origin"] = origin_counts
            continue

        if key == "react_trace" and isinstance(value, list):
            summary["trace_count"] = len(value)
            summary["last_trace"] = value[-1] if value else ""
            continue

        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
        elif isinstance(value, list):
            summary[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            summary[f"{key}_keys"] = sorted(value.keys())
        else:
            summary[key] = str(value)

    return summary

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    with logfire.span("chat", message=request.message):
        result = lang_graph.invoke(build_initial_state(request.message))
        answer = result.get("generation", "")
        sources = result.get("documents", [])
        logfire.info(
            "chat_completed",
            answer_chars=len(str(answer)),
            sources_count=len(sources) if isinstance(sources, list) else 0,
        )
        return ChatResponse(answer=answer, sources=sources)


@app.post("/chat/debug", response_model=ChatDebugResponse)
async def chat_debug(request: ChatRequest) -> ChatDebugResponse:
    with logfire.span("chat_debug", message=request.message):
        result = lang_graph.invoke(build_initial_state(request.message))
        response = ChatDebugResponse(
            answer=result.get("generation", ""),
            sources=result.get("documents", []),
            trace=result.get("react_trace", []),
            top_score=float(result.get("top_score", 0.0)),
            evidence_ok=bool(result.get("evidence_ok", False)),
            web_attempts=int(result.get("web_attempts", 0)),
            fallback=bool(result.get("fallback", False)),
        )
        logfire.info(
            "chat_debug_completed",
            top_score=response.top_score,
            evidence_ok=response.evidence_ok,
            web_attempts=response.web_attempts,
            fallback=response.fallback,
        )
        return response


# @app.post("/chat/stream-debug", response_model=ChatStreamDebugResponse)
# async def chat_stream_debug(request: ChatRequest) -> ChatStreamDebugResponse:
#     with logfire.span("chat_stream_debug", message=request.message):
#         state = build_initial_state(request.message)
#         visited_nodes: list[str] = []
#         node_updates: list[NodeUpdate] = []

#         for event in lang_graph.stream(state, stream_mode="updates"):
#             if not isinstance(event, dict):
#                 continue
#             for node_name, update in event.items():
#                 visited_nodes.append(str(node_name))
#                 if isinstance(update, dict):
#                     state.update(update)
#                     node_updates.append(
#                         NodeUpdate(
#                             node=str(node_name),
#                             updated_keys=sorted(update.keys()),
#                             summary=summarize_node_update(update),
#                         )
#                     )
#                 else:
#                     node_updates.append(
#                         NodeUpdate(
#                             node=str(node_name),
#                             updated_keys=[],
#                             summary={"value": str(update)},
#                         )
#                     )

#         response = ChatStreamDebugResponse(
#             answer=str(state.get("generation", "")),
#             sources=state.get("documents", []),
#             trace=state.get("react_trace", []),
#             top_score=float(state.get("top_score", 0.0)),
#             evidence_ok=bool(state.get("evidence_ok", False)),
#             web_attempts=int(state.get("web_attempts", 0)),
#             fallback=bool(state.get("fallback", False)),
#             visited_nodes=visited_nodes,
#             node_updates=node_updates,
#         )
#         logfire.info(
#             "chat_stream_debug_completed",
#             visited_nodes_count=len(visited_nodes),
#             node_updates_count=len(node_updates),
#             evidence_ok=response.evidence_ok,
#             web_attempts=response.web_attempts,
#             fallback=response.fallback,
#         )
#         return response


@app.post("/upload-pdf", response_model=UploadPdfResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadPdfResponse:
    with logfire.span("upload_pdf", filename=file.filename or "uploaded.pdf"):
        filename = file.filename or "uploaded.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        max_pdf_pages = parse_max_pdf_pages(required_env("UPLOAD_MAX_PDF_PAGES"))
        min_chunk_chars = int_env("ARXIV_MIN_CHUNK_CHARS")
        max_chunk_chars = int_env("ARXIV_MAX_CHUNK_CHARS")
        chunk_overlap_chars = int_env("ARXIV_CHUNK_OVERLAP_CHARS")
        exclude_references = DEFAULT_EXCLUDE_REFERENCES
        upload_llm_author_check = bool_env("UPLOAD_LLM_AUTHOR_CHECK_FIRST_PAGE")

        authors: list[str] = []
        if upload_llm_author_check:
            try:
                authors = extract_authors_from_first_page(
                    pdf_bytes=pdf_bytes,
                    fallback_authors=[],
                    author_check_llm=build_author_check_llm(),
                )
            except Exception as exc:
                logger.warning("Upload author extraction failed for %s: %s", filename, exc)
                authors = []

        extracted_text = extract_pdf_text(pdf_bytes=pdf_bytes, max_pages=max_pdf_pages)
        if not extracted_text:
            raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

        source_value = f"upload://{filename}"
        documents = build_upload_documents(
            filename=filename,
            source_value=source_value,
            extracted_text=extracted_text,
            authors=authors,
            min_chunk_chars=min_chunk_chars,
            max_chunk_chars=max_chunk_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            exclude_references=exclude_references,
        )
        if not documents:
            raise HTTPException(status_code=400, detail="Failed to split PDF into section-aware chunks.")

        embedding_model = required_env("OPENAI_EMBEDDING_MODEL")
        qdrant_url = required_env("QDRANT_URL")
        collection_name = required_env("QDRANT_COLLECTION")
        embeddings = OpenAIEmbeddings(model=embedding_model, **openai_client_kwargs())

        print_upload_summary(
            filename=filename,
            source_value=source_value,
            documents=documents,
            collection_name=collection_name,
        )

        try:
            vector_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                collection_name=collection_name,
                url=qdrant_url,
                content_payload_key="page_content",
                metadata_payload_key="metadata",
            )
            vector_store.add_documents(documents)
        except Exception:
            logfire.info("upload_pdf_existing_collection_failed_create_new", collection_name=collection_name)
            QdrantVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=collection_name,
                url=qdrant_url,
                content_payload_key="page_content",
                metadata_payload_key="metadata",
            )

        print(
            f"[upload-pdf] stored chunks={len(documents)} in collection={collection_name} at {qdrant_url}",
            flush=True,
        )
        logfire.info(
            "upload_pdf_completed",
            filename=filename,
            chunks=len(documents),
            collection_name=collection_name,
        )

    return UploadPdfResponse(filename=filename, chunks_indexed=len(documents))
