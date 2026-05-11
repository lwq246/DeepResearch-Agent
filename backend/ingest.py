import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import arxiv
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore


MAJOR_SECTION_KEYWORDS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "materials and methods",
    "approach",
    "experiments",
    "experiment",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "future work",
    "references",
    "bibliography",
}

MAJOR_SECTION_KEYWORDS_PATTERN = "|".join(
    sorted((re.escape(keyword) for keyword in MAJOR_SECTION_KEYWORDS), key=len, reverse=True)
)

EXTRACTED_SENTINEL_HEADING_RE = re.compile(r"^\s*@@HEADER@@\s+(?P<title>[^\n]{2,160})\s*$", flags=re.I)
EXTRACTED_MARKDOWN_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(?P<title>[^\n]{2,160})\s*$", flags=re.I)
EXTRACTED_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?P<title>(?:\d+(?:\.\d+){0,3}|[IVX]{1,8})[\)\.\-:]?\s+[A-Za-z][^\n]{1,120})\s*$",
    flags=re.I,
)
EXTRACTED_KEYWORD_HEADING_RE = re.compile(
    rf"^\s*(?:(?:\d+(?:\.\d+){{0,3}}|[IVX]{{1,8}})[\)\.\-:]?\s+)?(?P<title>{MAJOR_SECTION_KEYWORDS_PATTERN})(?:\s*[:.\-])?\s*$",
    flags=re.I,
)


def openai_client_kwargs() -> dict[str, str]:
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not base_url:
        return {}
    return {"base_url": base_url}


def bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def int_env_default(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def float_env_default(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def parse_max_pdf_pages(value: str | None) -> int | None:
    if value is None:
        return None

    raw = str(value).strip().lower()
    if raw in {"", "all", "none", "null", "0", "-1"}:
        return None

    parsed = int(raw)
    if parsed <= 0:
        return None
    return parsed


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


def paper_id_from_source_url(source_url: str | None) -> str:
    if not source_url:
        return "unknown"
    return source_url.rstrip("/").split("/")[-1]


def normalize_pdf_text(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def canonical_heading_key(title: str) -> str:
    lowered = title.strip().lower()
    lowered = re.sub(r"^\s*(?:\d+(?:\.\d+){0,3}|[ivx]{1,8})[\)\.\-:]?\s+", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip(" :.-")


def normalize_section_title(title: str) -> str:
    cleaned = title.replace("**", "").replace("__", "").replace("`", "")
    cleaned = re.sub(r"\s+", " ", cleaned.strip())
    cleaned = re.sub(r"[ \t:\-.]+$", "", cleaned)
    return cleaned or "Preamble"


def is_semantic_section_heading(title: str) -> bool:
    candidate = normalize_section_title(title)
    normalized = canonical_heading_key(candidate)
    if normalized in MAJOR_SECTION_KEYWORDS:
        return True

    # Keep numeric-only section markers like "3" or "3.2".
    if re.fullmatch(r"(?:\d+(?:\.\d+){0,5}|[ivx]{1,8})", candidate, flags=re.I):
        return True

    # Keep classical numbered headings like "3.2 Methods" or "IV. Results".
    if re.match(r"^(?:\d+(?:\.\d+){0,5}|[ivx]{1,8})(?:[\)\.\-:]?\s+).+", candidate, flags=re.I):
        return True

    return False


def should_exclude_section(title: str, exclude_references: bool) -> bool:
    if not exclude_references:
        return False
    normalized = canonical_heading_key(title)
    return normalized.startswith("references") or normalized.startswith("bibliography")


def split_index_on_word_boundary(text: str, start: int, max_chars: int) -> int:
    end = min(len(text), start + max_chars)
    if end >= len(text):
        return len(text)

    window = text[start:end]
    last_space = max(window.rfind(" "), window.rfind("\t"), window.rfind("\n"))
    if last_space <= 0:
        return end

    # Avoid creating tiny fragments when the nearest boundary is too far back.
    if last_space < int(max_chars * 0.6):
        return end

    return start + last_space


def overlap_suffix_on_word_boundary(text: str, overlap_chars: int) -> str:
    if not text or overlap_chars <= 0:
        return ""

    start = max(0, len(text) - overlap_chars)
    if start > 0 and start < len(text) and text[start - 1].isalnum() and text[start].isalnum():
        # Move to the next word boundary so overlap doesn't start mid-word.
        while start < len(text) and text[start].isalnum():
            start += 1
        while start < len(text) and text[start].isspace():
            start += 1

    suffix = text[start:].strip()
    if suffix:
        return suffix

    return text[-overlap_chars:].strip()


def split_oversized_content(content: str, max_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", content) if part and part.strip()]
    if not paragraphs:
        paragraphs = [part.strip() for part in re.split(r"(?<=[.!?])\s+", content) if part and part.strip()]
    if not paragraphs:
        return [content.strip()] if content.strip() else []

    chunks: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        candidate = paragraph if not buffer else f"{buffer}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            buffer = candidate
            continue

        if buffer:
            chunks.append(buffer.strip())
            buffer = ""

        if len(paragraph) <= max_chars:
            buffer = paragraph
            continue

        start = 0
        while start < len(paragraph):
            cut = split_index_on_word_boundary(paragraph, start=start, max_chars=max_chars)
            if cut <= start:
                cut = min(len(paragraph), start + max_chars)

            piece = paragraph[start:cut].strip()
            if piece:
                chunks.append(piece)
            start = cut
            while start < len(paragraph) and paragraph[start].isspace():
                start += 1

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks


def apply_chunk_overlap(parts: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0 or len(parts) <= 1:
        return parts

    overlapped: list[str] = [parts[0].strip()]
    for index in range(1, len(parts)):
        previous = parts[index - 1].strip()
        current = parts[index].strip()
        prefix = overlap_suffix_on_word_boundary(previous, overlap_chars=overlap_chars) if previous else ""
        if prefix:
            overlapped.append(f"{prefix}\n{current}".strip())
        else:
            overlapped.append(current)

    return overlapped


def detect_heading_from_extracted_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None

    for pattern, require_semantic_check in (
        (EXTRACTED_SENTINEL_HEADING_RE, True),
        (EXTRACTED_MARKDOWN_HEADING_RE, True),
        (EXTRACTED_NUMBERED_HEADING_RE, False),
        (EXTRACTED_KEYWORD_HEADING_RE, False),
    ):
        match = pattern.match(stripped)
        if not match:
            continue

        title = normalize_section_title(match.group("title"))
        if require_semantic_check and not is_semantic_section_heading(title):
            continue
        is_numeric_title = bool(re.fullmatch(r"(?:\d+(?:\.\d+){0,5}|[ivx]{1,8})", title, flags=re.I))
        min_len = 1 if is_numeric_title else 2
        if min_len <= len(title) <= 140:
            return title

    return None


def is_reference_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    candidate = re.sub(r"^#+\s*", "", stripped)
    candidate = candidate.replace("**", "").replace("__", "").replace("`", "")
    candidate = re.sub(r"^\s*(?:\d+(?:\.\d+){0,3}|[ivx]{1,8})[\)\.\-:]?\s+", "", candidate, flags=re.I)
    candidate = re.sub(r"\s+", " ", candidate).strip().lower()

    return candidate.startswith("references") or candidate.startswith("bibliography")


def is_running_header_or_footer_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    # Drop plain page numbers.
    if re.fullmatch(r"\d{1,3}", stripped):
        return True

    # Drop lines like "...ANALYSIS25" that are repeated page headers.
    without_page_no = re.sub(r"\d{1,3}$", "", stripped).strip()
    letters = [char for char in without_page_no if char.isalpha()]
    if len(letters) < 20:
        return False

    upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return upper_ratio >= 0.9


def cleanup_extracted_text_artifacts(extracted_text: str) -> str:
    lines = extracted_text.splitlines()
    cleaned: list[str] = []
    in_picture_block = False

    for line in lines:
        lowered = line.lower()

        if "start of picture text" in lowered:
            in_picture_block = True
            continue
        if "end of picture text" in lowered:
            in_picture_block = False
            continue
        if in_picture_block:
            continue

        if "intentionally omitted" in lowered:
            continue
        if is_running_header_or_footer_line(line):
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def truncate_text_at_references(extracted_text: str) -> str:
    lines = extracted_text.splitlines()
    for index, line in enumerate(lines):
        if is_reference_heading_line(line):
            return "\n".join(lines[:index]).strip()
    return extracted_text


def chunk_sections_from_extracted_text(
    extracted_text: str,
    min_chars: int,
    max_chars: int,
    chunk_overlap_chars: int,
    exclude_references: bool,
) -> list[dict[str, Any]]:
    if not extracted_text.strip():
        return []

    precleaned_text = cleanup_extracted_text_artifacts(extracted_text)
    normalized_text = truncate_text_at_references(precleaned_text) if exclude_references else precleaned_text
    lines = normalized_text.splitlines()

    raw_sections: list[dict[str, Any]] = []

    current_title = "Preamble"
    current_lines: list[str] = []

    def flush_current() -> None:
        content = normalize_pdf_text("\n".join(current_lines))
        if not content:
            return
        raw_sections.append(
            {
                "section_title": normalize_section_title(current_title),
                "chunk_text": content,
            }
        )

    for line in lines:
        heading = detect_heading_from_extracted_line(line)
        if heading:
            flush_current()
            current_title = heading
            current_lines = []
            continue
        current_lines.append(line)

    flush_current()

    if not raw_sections:
        fallback_text = normalize_pdf_text(normalized_text)
        if not fallback_text:
            return []
        raw_sections = [
            {
                "section_title": "Preamble",
                "chunk_text": fallback_text,
            }
        ]

    cleaned_sections: list[dict[str, Any]] = []
    for section in raw_sections:
        section_title = str(section["section_title"])
        if should_exclude_section(section_title, exclude_references=exclude_references):
            continue

        content = str(section["chunk_text"]).strip()
        if len(content) < min_chars:
            continue

        base_parts = [content] if len(content) <= max_chars else split_oversized_content(content, max_chars=max_chars)
        parts = apply_chunk_overlap(base_parts, overlap_chars=chunk_overlap_chars)
        for part_index, part in enumerate(parts, start=1):
            part_text = part.strip()
            if len(part_text) < min_chars:
                continue

            part_title = section_title if len(parts) == 1 else f"{section_title} (part {part_index})"
            cleaned_sections.append(
                {
                    "section_title": part_title,
                    "chunk_text": part_text,
                }
            )

    for section_index, section in enumerate(cleaned_sections):
        section["section_index"] = section_index

    return cleaned_sections


def extract_pdf_text_with_pymupdf4llm_from_bytes(pdf_bytes: bytes, max_pages: int | None) -> str:
    import pymupdf4llm  # type: ignore[import-not-found]

    page_limit = None if max_pages is None else max(1, max_pages)
    temp_path = ""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(pdf_bytes)
        temp_path = handle.name

    try:
        if page_limit is None:
            result = pymupdf4llm.to_markdown(temp_path)
        else:
            pages = list(range(page_limit))
            try:
                result = pymupdf4llm.to_markdown(temp_path, pages=pages)
            except TypeError:
                result = pymupdf4llm.to_markdown(temp_path)
            except Exception:
                # Some versions may reject explicit page lists for certain PDFs.
                result = pymupdf4llm.to_markdown(temp_path)
        return normalize_pdf_text(str(result))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def download_pdf_bytes(pdf_url: str, timeout_seconds: int) -> bytes:
    response = requests.get(pdf_url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.content


def fetch_arxiv_results(
    query: str,
    limit: int,
    sort_by: str,
    sort_order: str,
    request_delay_seconds: float,
    http_429_max_retries: int,
    http_429_backoff_seconds: int,
) -> list[arxiv.Result]:
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=parse_sort_criterion(sort_by),
        sort_order=parse_sort_order(sort_order),
    )

    client = arxiv.Client(
        page_size=min(100, limit),
        delay_seconds=max(3.0, request_delay_seconds),
        num_retries=1,
    )

    attempt = 0
    while True:
        try:
            return list(client.results(search))
        except arxiv.HTTPError as exc:
            if exc.status != 429 or attempt >= http_429_max_retries:
                raise

            sleep_seconds = min(900, http_429_backoff_seconds * (2**attempt))
            print(
                "arXiv rate limit hit (HTTP 429). "
                f"Retrying in {sleep_seconds}s (attempt {attempt + 1}/{http_429_max_retries})."
            )
            time.sleep(sleep_seconds)
            attempt += 1

def build_author_check_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_AUTHOR_CHECK_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")),
        temperature=0,
        max_tokens=int_env_default("OPENAI_AUTHOR_CHECK_MAX_TOKENS", 160),
        **openai_client_kwargs(),
    )

def parse_json_object(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    candidates = [text]

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.insert(0, text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None

def normalize_author_list(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        name = str(item).strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(name)
    return normalized

def extract_authors_from_first_page(
    pdf_bytes: bytes,
    fallback_authors: list[str],
    author_check_llm: ChatOpenAI,
) -> list[str]:
    first_page_text = extract_pdf_text_with_pymupdf4llm_from_bytes(pdf_bytes=pdf_bytes, max_pages=1)
    if not first_page_text.strip():
        return fallback_authors

    response = author_check_llm.invoke(
        [
            (
                "system",
                (
                    "Extract paper author names from first-page text. "
                    "Return strict JSON only with this schema: "
                    '{"authors": ["Author One", "Author Two"]}. '
                    "Do not include affiliations, emails, or explanations."
                ),
            ),
            (
                "human",
                f"First page text:\n{first_page_text[:12000]}",
            ),
        ]
    )

    response_text = response.content if isinstance(response.content, str) else str(response.content)
    parsed = parse_json_object(response_text)
    if not parsed:
        return fallback_authors

    maybe_authors = parsed.get("authors")
    if not isinstance(maybe_authors, list):
        return fallback_authors

    normalized = normalize_author_list(maybe_authors)
    return normalized if normalized else fallback_authors


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest arXiv papers into Qdrant with section-aware chunking.",
        epilog=(
            "Typical usage only needs --limit and --query. "
            "All other options are optional tuning knobs with defaults."
        ),
    )

    common = parser.add_argument_group("Common options")
    common.add_argument("--limit", type=int, default=100, help="Number of papers to ingest")
    common.add_argument(
        "--query",
        type=str,
        default=os.getenv("ARXIV_QUERY", "cat:cs.AI OR cat:cs.LG"),
        help="arXiv API search query",
    )
    common.add_argument(
        "--sort-by",
        type=str,
        default=os.getenv("ARXIV_SORT_BY", "SubmittedDate"),
        choices=["Relevance", "LastUpdatedDate", "SubmittedDate"],
        help="Sort criterion for arXiv API results",
    )
    common.add_argument(
        "--sort-order",
        type=str,
        default=os.getenv("ARXIV_SORT_ORDER", "Descending"),
        choices=["Ascending", "Descending"],
        help="Sort order for arXiv API results",
    )

    arxiv_controls = parser.add_argument_group("arXiv request controls (optional)")
    arxiv_controls.add_argument(
        "--arxiv-request-delay-seconds",
        type=float,
        default=float_env_default("ARXIV_REQUEST_DELAY_SECONDS", 6.0),
        help="Delay between arXiv API page requests in seconds",
    )
    arxiv_controls.add_argument(
        "--arxiv-429-max-retries",
        type=int,
        default=int_env_default("ARXIV_429_MAX_RETRIES", 6),
        help="Maximum retries when arXiv responds with HTTP 429",
    )
    arxiv_controls.add_argument(
        "--arxiv-429-backoff-seconds",
        type=int,
        default=int_env_default("ARXIV_429_BACKOFF_SECONDS", 20),
        help="Base backoff seconds for HTTP 429 retries (exponential)",
    )

    content = parser.add_argument_group("Content and chunking (optional)")
    content.add_argument(
        "--content-mode",
        type=str,
        default=os.getenv("ARXIV_CONTENT_MODE", "fulltext"),
        choices=["abstract", "fulltext"],
        help="Ingest abstracts only or extract full text from PDFs",
    )
    content.add_argument(
        "--max-pdf-pages",
        type=str,
        default=os.getenv("ARXIV_MAX_PDF_PAGES", "all"),
        help="Maximum number of PDF pages to parse in fulltext mode (default: all)",
    )
    content.add_argument(
        "--pdf-timeout-seconds",
        type=int,
        default=int_env_default("ARXIV_PDF_TIMEOUT_SECONDS", 30),
        help="HTTP timeout for PDF download requests",
    )
    content.add_argument(
        "--min-chunk-chars",
        type=int,
        default=int_env_default("ARXIV_MIN_CHUNK_CHARS", 80),
        help="Minimum characters required for a chunk",
    )
    content.add_argument(
        "--max-chunk-chars",
        type=int,
        default=int_env_default("ARXIV_MAX_CHUNK_CHARS", 2500),
        help="Maximum characters before splitting a chunk",
    )
    content.add_argument(
        "--chunk-overlap-chars",
        type=int,
        default=int_env_default("ARXIV_CHUNK_OVERLAP_CHARS", 500),
        help="Number of trailing characters repeated at the start of the next chunk",
    )
    content.add_argument(
        "--exclude-references",
        action="store_true",
        default=bool_env("ARXIV_EXCLUDE_REFERENCES", True),
        help="Skip sections titled references or bibliography",
    )
    content.add_argument(
        "--keep-references",
        dest="exclude_references",
        action="store_false",
        help="Do not skip references/bibliography sections",
    )
    content.add_argument(
        "--llm-author-check-first-page",
        action="store_true",
        default=bool_env("ARXIV_LLM_AUTHOR_CHECK_FIRST_PAGE", False),
        help="Use LLM on first PDF page to infer authors and store them in metadata",
    )
    content.add_argument(
        "--no-llm-author-check-first-page",
        dest="llm_author_check_first_page",
        action="store_false",
        help="Disable first-page LLM author extraction",
    )

    debug = parser.add_argument_group("Local/debug option (optional)")
    debug.add_argument(
        "--local-pdf",
        type=str,
        default=os.getenv("ARXIV_LOCAL_PDF", ""),
        help="Optional local PDF path for single-file testing (skips arXiv API)",
    )

    parsed = parser.parse_args()
    try:
        parsed.max_pdf_pages = parse_max_pdf_pages(parsed.max_pdf_pages)
    except ValueError as exc:
        raise ValueError("--max-pdf-pages must be a positive integer or one of: all, none, 0") from exc
    return parsed


if __name__ == "__main__":
    args = parse_args()
    ingest(
        limit=args.limit,
        query=args.query,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
        arxiv_request_delay_seconds=args.arxiv_request_delay_seconds,
        arxiv_429_max_retries=args.arxiv_429_max_retries,
        arxiv_429_backoff_seconds=args.arxiv_429_backoff_seconds,
        content_mode=args.content_mode,
        max_pdf_pages=args.max_pdf_pages,
        pdf_timeout_seconds=args.pdf_timeout_seconds,
        min_chunk_chars=args.min_chunk_chars,
        max_chunk_chars=args.max_chunk_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
        exclude_references=args.exclude_references,
        llm_author_check_first_page=args.llm_author_check_first_page,
        local_pdf=args.local_pdf,
    )
