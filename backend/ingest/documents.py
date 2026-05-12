from pathlib import Path
from typing import Any

import arxiv
from langchain_core.documents import Document

from .authors import build_author_check_llm
from .authors import extract_authors_from_first_page
from .chunking import chunk_sections_from_extracted_text
from .env import paper_id_from_source_url
from .sources import download_pdf_bytes
from .sources import extract_pdf_text_with_pymupdf4llm_from_bytes


def build_section_documents(base_metadata: dict[str, Any], sections: list[dict[str, Any]]) -> list[Document]:
    docs: list[Document] = []
    for section in sections:
        docs.append(
            Document(
                page_content=str(section["chunk_text"]),
                metadata={
                    **base_metadata,
                    "section_title": str(section["section_title"]),
                    "section_index": int(section["section_index"]),
                },
            )
        )
    return docs


def build_abstract_document(base_metadata: dict[str, Any], abstract: str) -> Document:
    return Document(
        page_content=abstract,
        metadata={
            **base_metadata,
            "section_title": "abstract",
            "section_index": 0,
        },
    )


def build_documents_from_local_pdf(
    pdf_path: str,
    max_pdf_pages: int | None,
    min_chunk_chars: int,
    max_chunk_chars: int,
    chunk_overlap_chars: int,
    exclude_references: bool,
    llm_author_check_first_page: bool = False,
) -> list[Document]:
    path = Path(pdf_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Local PDF not found: {pdf_path}")

    pdf_bytes = path.read_bytes()
    extracted_text = extract_pdf_text_with_pymupdf4llm_from_bytes(
        pdf_bytes=pdf_bytes,
        max_pages=max_pdf_pages,
    )
    sections = chunk_sections_from_extracted_text(
        extracted_text=extracted_text,
        min_chars=min_chunk_chars,
        max_chars=max_chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        exclude_references=exclude_references,
    )

    title = path.stem
    source_url = str(path.resolve())
    authors: list[str] = []
    if llm_author_check_first_page:
        try:
            authors = extract_authors_from_first_page(
                pdf_bytes=pdf_bytes,
                fallback_authors=[],
                author_check_llm=build_author_check_llm(),
            )
        except Exception:  # noqa: BLE001
            authors = []

    base_metadata = {
        "title": title,
        "paper_id": title,
        "source_url": source_url,
        "authors": authors,
    }

    docs = build_section_documents(base_metadata=base_metadata, sections=sections)

    print(
        f"Local PDF summary: path='{path}', sections={len(sections)}, chunks={len(docs)}."
    )
    return docs


def build_documents(
    rows: list[arxiv.Result],
    content_mode: str,
    max_pdf_pages: int | None,
    pdf_timeout_seconds: int,
    min_chunk_chars: int,
    max_chunk_chars: int,
    chunk_overlap_chars: int,
    exclude_references: bool,
    llm_author_check_first_page: bool,
) -> list[Document]:
    docs: list[Document] = []
    fulltext_success_count = 0
    abstract_fallback_count = 0
    author_check_llm = build_author_check_llm() if llm_author_check_first_page else None

    for row in rows:
        abstract = (row.summary or "").strip()
        title = (row.title or "Untitled Paper").strip()
        source_url = row.entry_id
        fallback_authors = [author.name for author in (row.authors or []) if getattr(author, "name", "")]
        resolved_authors = fallback_authors

        pdf_url = row.pdf_url or ""
        used_fulltext = False
        pdf_bytes: bytes | None = None

        if pdf_url and (content_mode == "fulltext" or author_check_llm is not None):
            try:
                pdf_bytes = download_pdf_bytes(pdf_url=pdf_url, timeout_seconds=pdf_timeout_seconds)
            except Exception:  # noqa: BLE001
                pdf_bytes = None

        if author_check_llm is not None and pdf_bytes is not None:
            try:
                resolved_authors = extract_authors_from_first_page(
                    pdf_bytes=pdf_bytes,
                    fallback_authors=fallback_authors,
                    author_check_llm=author_check_llm,
                )
            except Exception:  # noqa: BLE001
                resolved_authors = fallback_authors

        base_metadata = {
            "title": title,
            "paper_id": paper_id_from_source_url(source_url),
            "source_url": source_url,
            "authors": resolved_authors,
        }

        if content_mode == "fulltext" and pdf_bytes is not None:
            try:
                extracted_text = extract_pdf_text_with_pymupdf4llm_from_bytes(
                    pdf_bytes=pdf_bytes,
                    max_pages=max_pdf_pages,
                )
                sections = chunk_sections_from_extracted_text(
                    extracted_text=extracted_text,
                    min_chars=min_chunk_chars,
                    max_chars=max_chunk_chars,
                    chunk_overlap_chars=chunk_overlap_chars,
                    exclude_references=exclude_references,
                )
            except Exception:  # noqa: BLE001
                sections = []

            if sections:
                used_fulltext = True
                fulltext_success_count += 1
                docs.extend(build_section_documents(base_metadata=base_metadata, sections=sections))

        if not used_fulltext and abstract:
            abstract_fallback_count += 1
            docs.append(build_abstract_document(base_metadata=base_metadata, abstract=abstract))

    if content_mode == "fulltext":
        print(
            "Full-text summary: "
            f"papers_with_fulltext={fulltext_success_count}, "
            f"abstract_fallbacks={abstract_fallback_count}."
        )
    else:
        print(f"Abstract summary: papers={abstract_fallback_count}.")

    return docs
