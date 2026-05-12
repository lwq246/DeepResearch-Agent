import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from .documents import build_documents
from .documents import build_documents_from_local_pdf
from .env import openai_client_kwargs
from .sources import fetch_arxiv_results


def ingest(
    limit: int,
    query: str,
    sort_by: str,
    sort_order: str,
    arxiv_request_delay_seconds: float,
    arxiv_429_max_retries: int,
    arxiv_429_backoff_seconds: int,
    content_mode: str,
    max_pdf_pages: int | None,
    pdf_timeout_seconds: int,
    min_chunk_chars: int,
    max_chunk_chars: int,
    chunk_overlap_chars: int,
    exclude_references: bool,
    llm_author_check_first_page: bool,
    local_pdf: str,
) -> None:
    load_dotenv()

    if min_chunk_chars <= 0:
        raise ValueError("min_chunk_chars must be greater than 0.")
    if max_chunk_chars <= min_chunk_chars:
        raise ValueError("max_chunk_chars must be greater than min_chunk_chars.")
    if chunk_overlap_chars < 0:
        raise ValueError("chunk_overlap_chars must be 0 or greater.")
    if chunk_overlap_chars >= max_chunk_chars:
        raise ValueError("chunk_overlap_chars must be smaller than max_chunk_chars.")
    if arxiv_request_delay_seconds <= 0:
        raise ValueError("arxiv_request_delay_seconds must be greater than 0.")
    if arxiv_429_max_retries < 0:
        raise ValueError("arxiv_429_max_retries must be 0 or greater.")
    if arxiv_429_backoff_seconds <= 0:
        raise ValueError("arxiv_429_backoff_seconds must be greater than 0.")

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "arxiv_docs")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    local_pdf_path = local_pdf.strip()
    if local_pdf_path:
        if content_mode != "fulltext":
            print("Warning: --local-pdf mode uses fulltext extraction regardless of --content-mode.")
        documents = build_documents_from_local_pdf(
            pdf_path=local_pdf_path,
            max_pdf_pages=max_pdf_pages,
            min_chunk_chars=min_chunk_chars,
            max_chunk_chars=max_chunk_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            exclude_references=exclude_references,
            llm_author_check_first_page=llm_author_check_first_page,
        )
    else:
        print(f"Querying arXiv API with query='{query}', limit={limit}, sort_by={sort_by}, sort_order={sort_order}")
        dataset = fetch_arxiv_results(
            query=query,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
            request_delay_seconds=arxiv_request_delay_seconds,
            http_429_max_retries=arxiv_429_max_retries,
            http_429_backoff_seconds=arxiv_429_backoff_seconds,
        )
        documents = build_documents(
            dataset,
            content_mode=content_mode,
            max_pdf_pages=max_pdf_pages,
            pdf_timeout_seconds=pdf_timeout_seconds,
            min_chunk_chars=min_chunk_chars,
            max_chunk_chars=max_chunk_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            exclude_references=exclude_references,
            llm_author_check_first_page=llm_author_check_first_page,
        )

    if not documents:
        raise RuntimeError("No documents were created from arXiv results.")

    print(
        f"Built {len(documents)} chunks (content_mode={content_mode}, "
        f"max_pdf_pages={'all' if max_pdf_pages is None else max_pdf_pages}, "
        f"min_chunk_chars={min_chunk_chars}, max_chunk_chars={max_chunk_chars}, "
        f"chunk_overlap_chars={chunk_overlap_chars})."
    )

    embeddings = OpenAIEmbeddings(model=embedding_model, **openai_client_kwargs())
    print(f"Uploading to Qdrant collection '{collection_name}' at {qdrant_url}")

    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        url=qdrant_url,
    )
    print("Ingestion completed.")
