import os
from typing import Any

import fitz
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

try:
    from .graph import app as lang_graph
    from .ingest import is_section_selected
    from .models.api_models import ChatDebugResponse
    from .models.api_models import ChatRequest
    from .models.api_models import ChatResponse
    from .models.api_models import ChatStreamDebugResponse
    from .models.api_models import NodeUpdate
    from .models.api_models import UploadPdfResponse
    from .ingest import normalize_pdf_text
    from .ingest import parse_csv_list
    from .ingest import split_into_sections
except ImportError:
    from graph import app as lang_graph
    from ingest import is_section_selected
    from models.api_models import ChatDebugResponse
    from models.api_models import ChatRequest
    from models.api_models import ChatResponse
    from models.api_models import ChatStreamDebugResponse
    from models.api_models import NodeUpdate
    from models.api_models import UploadPdfResponse
    from ingest import normalize_pdf_text
    from ingest import parse_csv_list
    from ingest import split_into_sections


app = FastAPI(title="ArXiv RAG Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def openai_client_kwargs() -> dict[str, str]:
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not base_url:
        return {}
    return {"base_url": base_url}


def extract_pdf_text(pdf_bytes: bytes, max_pages: int) -> str:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_count = min(max_pages, document.page_count)
        text_chunks: list[str] = []
        for page_index in range(page_count):
            text = document.load_page(page_index).get_text("text")
            if text.strip():
                text_chunks.append(text)
        return normalize_pdf_text("\n".join(text_chunks))
    finally:
        document.close()


def build_upload_documents(
    filename: str,
    source_value: str,
    extracted_text: str,
    chunk_size: int,
    chunk_overlap: int,
    include_sections_csv: str,
    exclude_sections_csv: str,
    fallback_section_keywords_csv: str,
) -> list[Document]:
    include_sections = parse_csv_list(include_sections_csv)
    exclude_sections = parse_csv_list(exclude_sections_csv)
    fallback_section_keywords = parse_csv_list(fallback_section_keywords_csv)

    sections = split_into_sections(extracted_text, fallback_section_keywords)
    selected_sections: list[tuple[str, str]] = []
    for section_name, section_content in sections:
        if is_section_selected(section_name, include_sections, exclude_sections):
            selected_sections.append((section_name, section_content))

    if not selected_sections:
        selected_sections = [("full_text", extracted_text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents: list[Document] = []
    full_text_length = len(extracted_text)
    for section_index, (section_name, section_content) in enumerate(selected_sections):
        chunks = splitter.split_text(section_content)
        for chunk_index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "title": filename,
                        "paper_id": filename,
                        "source_url": source_value,
                        "pdf_url": source_value,
                        "published": "",
                        "updated": "",
                        "primary_category": "",
                        "categories": [],
                        "dataset": "uploaded_pdf",
                        "query": "manual_upload",
                        "content_source": "pdf_full_text",
                        "full_text_length": full_text_length,
                        "section": section_name,
                        "section_index": section_index,
                        "chunk_index": chunk_index,
                    },
                )
            )

    return documents


def print_upload_summary(
    filename: str,
    source_value: str,
    documents: list[Document],
    collection_name: str,
) -> None:
    preview_chunks = max(0, int(os.getenv("UPLOAD_PRINT_PREVIEW_CHUNKS", "3")))
    preview_chars = max(40, int(os.getenv("UPLOAD_PRINT_PREVIEW_CHARS", "160")))

    print(
        f"[upload-pdf] filename={filename} collection={collection_name} "
        f"source={source_value} chunks={len(documents)}",
        flush=True,
    )

    for document in documents[:preview_chunks]:
        section_name = document.metadata.get("section", "?")
        chunk_index = document.metadata.get("chunk_index", "?")
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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    result = lang_graph.invoke(build_initial_state(request.message))
    return ChatResponse(answer=result.get("generation", ""), sources=result.get("documents", []))


@app.post("/chat/debug", response_model=ChatDebugResponse)
async def chat_debug(request: ChatRequest) -> ChatDebugResponse:
    result = lang_graph.invoke(build_initial_state(request.message))
    return ChatDebugResponse(
        answer=result.get("generation", ""),
        sources=result.get("documents", []),
        trace=result.get("react_trace", []),
        top_score=float(result.get("top_score", 0.0)),
        evidence_ok=bool(result.get("evidence_ok", False)),
        web_attempts=int(result.get("web_attempts", 0)),
        fallback=bool(result.get("fallback", False)),
    )


@app.post("/chat/stream-debug", response_model=ChatStreamDebugResponse)
async def chat_stream_debug(request: ChatRequest) -> ChatStreamDebugResponse:
    state = build_initial_state(request.message)
    visited_nodes: list[str] = []
    node_updates: list[NodeUpdate] = []

    for event in lang_graph.stream(state, stream_mode="updates"):
        if not isinstance(event, dict):
            continue
        for node_name, update in event.items():
            visited_nodes.append(str(node_name))
            if isinstance(update, dict):
                state.update(update)
                node_updates.append(
                    NodeUpdate(
                        node=str(node_name),
                        updated_keys=sorted(update.keys()),
                        summary=summarize_node_update(update),
                    )
                )
            else:
                node_updates.append(
                    NodeUpdate(
                        node=str(node_name),
                        updated_keys=[],
                        summary={"value": str(update)},
                    )
                )

    return ChatStreamDebugResponse(
        answer=str(state.get("generation", "")),
        sources=state.get("documents", []),
        trace=state.get("react_trace", []),
        top_score=float(state.get("top_score", 0.0)),
        evidence_ok=bool(state.get("evidence_ok", False)),
        web_attempts=int(state.get("web_attempts", 0)),
        fallback=bool(state.get("fallback", False)),
        visited_nodes=visited_nodes,
        node_updates=node_updates,
    )


@app.post("/upload-pdf", response_model=UploadPdfResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadPdfResponse:
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    max_pdf_pages = int(os.getenv("UPLOAD_MAX_PDF_PAGES", "30"))
    chunk_size = int(os.getenv("ARXIV_CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("ARXIV_CHUNK_OVERLAP", "100"))
    include_sections_csv = os.getenv(
        "ARXIV_INCLUDE_SECTIONS",
        (
            "abstract,introduction,background,related work,method,methodology,approach,"
            "experiment,experiments,results,discussion,conclusion,limitations"
        ),
    )
    exclude_sections_csv = os.getenv(
        "ARXIV_EXCLUDE_SECTIONS",
        "references,acknowledgements,acknowledgments,appendix",
    )
    fallback_section_keywords_csv = os.getenv(
        "ARXIV_FALLBACK_SECTION_KEYWORDS",
        "abstract,introduction,method,approach,experiment,results,conclusion",
    )

    extracted_text = extract_pdf_text(pdf_bytes=pdf_bytes, max_pages=max_pdf_pages)
    if not extracted_text:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

    source_value = f"upload://{filename}"
    documents = build_upload_documents(
        filename=filename,
        source_value=source_value,
        extracted_text=extracted_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_sections_csv=include_sections_csv,
        exclude_sections_csv=exclude_sections_csv,
        fallback_section_keywords_csv=fallback_section_keywords_csv,
    )
    if not documents:
        raise HTTPException(status_code=400, detail="Failed to split PDF into section-aware chunks.")

    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "arxiv_docs")
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

    return UploadPdfResponse(filename=filename, chunks_indexed=len(documents))
