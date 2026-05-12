import os

from . import ingest
from .env import bool_env
from .env import float_env_default
from .env import int_env_default
from .env import parse_max_pdf_pages
from . import parse_args


def main() -> None:
    args = parse_args()
    ingest(
        limit=args.limit,
        query=os.getenv("ARXIV_QUERY", "cat:cs.AI OR cat:cs.LG"),
        sort_by=os.getenv("ARXIV_SORT_BY", "SubmittedDate"),
        sort_order=os.getenv("ARXIV_SORT_ORDER", "Descending"),
        arxiv_request_delay_seconds=float_env_default("ARXIV_REQUEST_DELAY_SECONDS", 6.0),
        arxiv_429_max_retries=int_env_default("ARXIV_429_MAX_RETRIES", 6),
        arxiv_429_backoff_seconds=int_env_default("ARXIV_429_BACKOFF_SECONDS", 20),
        content_mode=os.getenv("ARXIV_CONTENT_MODE", "fulltext"),
        max_pdf_pages=parse_max_pdf_pages(os.getenv("ARXIV_MAX_PDF_PAGES", "all")),
        pdf_timeout_seconds=int_env_default("ARXIV_PDF_TIMEOUT_SECONDS", 30),
        min_chunk_chars=int_env_default("ARXIV_MIN_CHUNK_CHARS", 80),
        max_chunk_chars=int_env_default("ARXIV_MAX_CHUNK_CHARS", 2500),
        chunk_overlap_chars=int_env_default("ARXIV_CHUNK_OVERLAP_CHARS", 500),
        exclude_references=bool_env("ARXIV_EXCLUDE_REFERENCES", True),
        llm_author_check_first_page=bool_env("ARXIV_LLM_AUTHOR_CHECK_FIRST_PAGE", False),
        local_pdf=os.getenv("ARXIV_LOCAL_PDF", ""),
    )


if __name__ == "__main__":
    main()
