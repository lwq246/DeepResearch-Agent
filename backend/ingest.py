import argparse
import os
import re
from typing import Any

import arxiv
import fitz
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore


SECTION_KEYWORDS = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methodology",
    "materials and methods",
    "approach",
    "experiment",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "future work",
    "acknowledgements",
    "acknowledgments",
    "references",
]

SECTION_KEYWORDS_PATTERN = "|".join(sorted((re.escape(keyword) for keyword in SECTION_KEYWORDS), key=len, reverse=True))

NUMBERED_SECTION_RE = re.compile(
    r"(?mi)^\s*(?P<header>(?:\d+(?:\.\d+)*|[ivxlcdm]+)[\)\.\-:]?\s+[A-Za-z][^\n]{1,120})\s*$"
)
PLAIN_SECTION_RE = re.compile(
    rf"(?mi)^\s*(?P<header>{SECTION_KEYWORDS_PATTERN})(?:\s*[:.\-])?\s*$"
)
KEYWORD_HEADING_RE = re.compile(
    rf"(?mi)^\s*(?P<header>(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+|[A-Z])[\)\.\-:]?\s*)?"
    rf"(?:{SECTION_KEYWORDS_PATTERN})(?:\s+[A-Za-z][^\n]{{0,80}})?)(?:\s*[:.\-])?\s*$"
)


def openai_client_kwargs() -> dict[str, str]:
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not base_url:
        return {}
    return {"base_url": base_url}


def parse_sort_criterion(sort_by: str) -> arxiv.SortCriterion:
    normalized = sort_by.strip().lower()
    if normalized == "submitteddate":
        return arxiv.SortCriterion.SubmittedDate
    if normalized == "lastupdateddate":
        return arxiv.SortCriterion.LastUpdatedDate
    return arxiv.SortCriterion.Relevance


def parse_sort_order(sort_order: str) -> arxiv.SortOrder:
    normalized = sort_order.strip().lower()
    if normalized == "ascending":
        return arxiv.SortOrder.Ascending
    return arxiv.SortOrder.Descending


def fetch_arxiv_results(
    query: str,
    limit: int,
    sort_by: str,
    sort_order: str,
) -> list[arxiv.Result]:
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=parse_sort_criterion(sort_by),
        sort_order=parse_sort_order(sort_order),
    )

    # Keep request rate polite to avoid arXiv throttling.
    client = arxiv.Client(page_size=min(100, limit), delay_seconds=3.0, num_retries=5)
    return list(client.results(search))


def normalize_pdf_text(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(pdf_url: str, max_pages: int, timeout_seconds: int) -> str:
    response = requests.get(pdf_url, timeout=timeout_seconds)
    response.raise_for_status()

    doc = fitz.open(stream=response.content, filetype="pdf")
    chunks: list[str] = []
    page_count = min(max_pages, doc.page_count)
    for page_index in range(page_count):
        text = doc.load_page(page_index).get_text("text")
        if text.strip():
            chunks.append(text)

    doc.close()
    return normalize_pdf_text("\n".join(chunks))


def clean_header(header: str) -> str:
    without_numbers = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)[\)\.\-:]?\s+", "", header.strip(), flags=re.I)
    without_trailing_punct = re.sub(r"[\s:\-.]+$", "", without_numbers)
    return without_trailing_punct.lower()


def parse_csv_list(raw_value: str) -> list[str]:
    return [item.strip().lower() for item in raw_value.split(",") if item.strip()]


def find_section_boundaries(text: str, fallback_keywords: list[str]) -> list[tuple[int, int, str]]:
    boundaries: list[tuple[int, int, str]] = []

    for match in NUMBERED_SECTION_RE.finditer(text):
        boundaries.append((match.start(), match.end(), match.group("header")))
    for match in PLAIN_SECTION_RE.finditer(text):
        boundaries.append((match.start(), match.end(), match.group("header")))
    for match in KEYWORD_HEADING_RE.finditer(text):
        boundaries.append((match.start(), match.end(), match.group("header")))

    if not boundaries:
        for keyword in fallback_keywords:
            pattern = re.compile(
                rf"(?mi)^\s*(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+)[\)\.\-:]?\s*)?"
                rf"(?P<header>{re.escape(keyword)}(?:\s+[A-Za-z][^\n]{{0,80}})?)"
                rf"(?:\s*[:.\-])?\s*$"
            )
            for match in pattern.finditer(text):
                boundaries.append((match.start(), match.end(), match.group("header")))

    boundaries.sort(key=lambda item: item[0])

    deduped: list[tuple[int, int, str]] = []
    seen_positions: set[int] = set()
    for start, end, header in boundaries:
        if start in seen_positions:
            continue
        seen_positions.add(start)
        deduped.append((start, end, header))

    return deduped


def split_into_sections(text: str, fallback_keywords: list[str]) -> list[tuple[str, str]]:
    boundaries = find_section_boundaries(text, fallback_keywords)
    if not boundaries:
        cleaned = normalize_pdf_text(text)
        return [("full_text", cleaned)] if cleaned else []

    sections: list[tuple[str, str]] = []
    for idx, (_, header_end, header_text) in enumerate(boundaries):
        next_start = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(text)
        content = normalize_pdf_text(text[header_end:next_start])
        if not content:
            continue
        sections.append((clean_header(header_text), content))

    if not sections:
        cleaned = normalize_pdf_text(text)
        return [("full_text", cleaned)] if cleaned else []
    return sections


def is_section_selected(section_name: str, include_sections: list[str], exclude_sections: list[str]) -> bool:
    for excluded in exclude_sections:
        if excluded in section_name:
            return False
    if not include_sections:
        return True
    return any(included in section_name for included in include_sections)


def build_documents(
    rows: list[arxiv.Result],
    query: str,
    content_mode: str,
    max_pdf_pages: int,
    pdf_timeout_seconds: int,
    chunk_size: int,
    chunk_overlap: int,
    include_sections: list[str],
    exclude_sections: list[str],
    fallback_section_keywords: list[str],
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    docs: list[Document] = []
    full_text_count = 0
    abstract_fallback_count = 0
    total_chunks = 0

    for row in rows:
        abstract = (row.summary or "").strip()
        if not abstract:
            continue

        title = (row.title or "Untitled Paper").strip()
        source_url = row.entry_id
        paper_id = source_url.rstrip("/").split("/")[-1] if source_url else "unknown"
        categories = list(row.categories or [])
        published = row.published.isoformat() if row.published else ""
        updated = row.updated.isoformat() if row.updated else ""
        primary_category = row.primary_category or ""
        pdf_url = row.pdf_url or ""

        sections: list[tuple[str, str]] = [("abstract", abstract)]
        content_source = "abstract"
        full_text_length = len(abstract)

        if content_mode == "fulltext" and pdf_url:
            try:
                extracted = extract_pdf_text(
                    pdf_url=pdf_url,
                    max_pages=max_pdf_pages,
                    timeout_seconds=pdf_timeout_seconds,
                )
                if extracted:
                    sections = split_into_sections(extracted, fallback_section_keywords)
                    content_source = "pdf_full_text"
                    full_text_count += 1
                    full_text_length = len(extracted)
                else:
                    abstract_fallback_count += 1
                    content_source = "abstract_fallback_empty_pdf_text"
            except Exception as exc:  # noqa: BLE001
                abstract_fallback_count += 1
                content_source = f"abstract_fallback_pdf_error:{type(exc).__name__}"

        selected_sections: list[tuple[str, str]] = []
        for section_name, section_content in sections:
            if is_section_selected(section_name, include_sections, exclude_sections):
                selected_sections.append((section_name, section_content))

        if not selected_sections:
            selected_sections = [("abstract", abstract)]

        for section_index, (section_name, section_content) in enumerate(selected_sections):
            chunks = splitter.split_text(section_content)
            for chunk_index, chunk in enumerate(chunks):
                total_chunks += 1
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "title": title,
                            "paper_id": paper_id,
                            "source_url": source_url,
                            "pdf_url": pdf_url,
                            "published": published,
                            "updated": updated,
                            "primary_category": primary_category,
                            "categories": categories,
                            "dataset": "arxiv_api",
                            "query": query,
                            "content_source": content_source,
                            "full_text_length": full_text_length,
                            "section": section_name,
                            "section_index": section_index,
                            "chunk_index": chunk_index,
                        },
                    )
                )

    if content_mode == "fulltext":
        print(
            f"Full-text extraction summary: {full_text_count} succeeded, {abstract_fallback_count} fell back to abstract."
        )
    print(f"Built {len(docs)} chunks total (chunk_size={chunk_size}, chunk_overlap={chunk_overlap}).")

    return docs


def ingest(
    limit: int,
    query: str,
    sort_by: str,
    sort_order: str,
    content_mode: str,
    max_pdf_pages: int,
    pdf_timeout_seconds: int,
    chunk_size: int,
    chunk_overlap: int,
    include_sections_csv: str,
    exclude_sections_csv: str,
    fallback_section_keywords_csv: str,
) -> None:
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "arxiv_docs")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    print(f"Querying arXiv API with query='{query}', limit={limit}, sort_by={sort_by}, sort_order={sort_order}")
    dataset = fetch_arxiv_results(query=query, limit=limit, sort_by=sort_by, sort_order=sort_order)

    print(
        f"Building documents from {len(dataset)} arXiv API results with content_mode={content_mode}, "
        f"max_pdf_pages={max_pdf_pages}"
    )

    include_sections = parse_csv_list(include_sections_csv)
    exclude_sections = parse_csv_list(exclude_sections_csv)
    fallback_section_keywords = parse_csv_list(fallback_section_keywords_csv)

    documents = build_documents(
        dataset,
        query=query,
        content_mode=content_mode,
        max_pdf_pages=max_pdf_pages,
        pdf_timeout_seconds=pdf_timeout_seconds,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_sections=include_sections,
        exclude_sections=exclude_sections,
        fallback_section_keywords=fallback_section_keywords,
    )
    if not documents:
        raise RuntimeError("No documents were created from arXiv results.")

    embeddings = OpenAIEmbeddings(model=embedding_model, **openai_client_kwargs())

    print(f"Uploading {len(documents)} documents to Qdrant collection '{collection_name}' at {qdrant_url}")
    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        url=qdrant_url,
    )
    print("Ingestion completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest arXiv abstracts or full-text into Qdrant.")
    parser.add_argument("--limit", type=int, default=500, help="Number of papers to ingest (default: 500)")
    parser.add_argument(
        "--query",
        type=str,
        default=os.getenv("ARXIV_QUERY", "cat:cs.AI OR cat:cs.LG"),
        help="arXiv API search query (default: cat:cs.AI OR cat:cs.LG)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default=os.getenv("ARXIV_SORT_BY", "SubmittedDate"),
        choices=["Relevance", "LastUpdatedDate", "SubmittedDate"],
        help="Sort criterion for arXiv API results",
    )
    parser.add_argument(
        "--sort-order",
        type=str,
        default=os.getenv("ARXIV_SORT_ORDER", "Descending"),
        choices=["Ascending", "Descending"],
        help="Sort order for arXiv API results",
    )
    parser.add_argument(
        "--content-mode",
        type=str,
        default=os.getenv("ARXIV_CONTENT_MODE", "abstract"),
        choices=["abstract", "fulltext"],
        help="Use only abstract text or extract full text from PDFs",
    )
    parser.add_argument(
        "--max-pdf-pages",
        type=int,
        default=int(os.getenv("ARXIV_MAX_PDF_PAGES", "25")),
        help="Maximum number of PDF pages to extract per paper in fulltext mode",
    )
    parser.add_argument(
        "--pdf-timeout-seconds",
        type=int,
        default=int(os.getenv("ARXIV_PDF_TIMEOUT_SECONDS", "30")),
        help="HTTP timeout for PDF download requests",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("ARXIV_CHUNK_SIZE", "800")),
        help="Chunk size used for section-aware text splitting",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("ARXIV_CHUNK_OVERLAP", "100")),
        help="Chunk overlap used for section-aware text splitting",
    )
    parser.add_argument(
        "--include-sections",
        type=str,
        default=os.getenv(
            "ARXIV_INCLUDE_SECTIONS",
            "abstract,introduction,background,related work,method,methodology,approach,experiment,experiments,results,discussion,conclusion,limitations",
        ),
        help="Comma-separated section names to include when content_mode is fulltext",
    )
    parser.add_argument(
        "--exclude-sections",
        type=str,
        default=os.getenv("ARXIV_EXCLUDE_SECTIONS", "references,acknowledgements,acknowledgments,appendix"),
        help="Comma-separated section names to always skip",
    )
    parser.add_argument(
        "--fallback-section-keywords",
        type=str,
        default=os.getenv(
            "ARXIV_FALLBACK_SECTION_KEYWORDS",
            "abstract,introduction,method,approach,experiment,results,conclusion",
        ),
        help="Keywords used as fallback section detectors if regex headings are not found",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest(
        limit=args.limit,
        query=args.query,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
        content_mode=args.content_mode,
        max_pdf_pages=args.max_pdf_pages,
        pdf_timeout_seconds=args.pdf_timeout_seconds,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        include_sections_csv=args.include_sections,
        exclude_sections_csv=args.exclude_sections,
        fallback_section_keywords_csv=args.fallback_section_keywords,
    )
