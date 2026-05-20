import os
import tempfile
import time

import arxiv
import requests

from .chunking import normalize_pdf_text
from backend.configuration import parse_sort_criterion
from backend.configuration import parse_sort_order


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
