from pathlib import Path

from dotenv import load_dotenv

from .pipeline import ingest
from .chunking import DEFAULT_EXCLUDE_REFERENCES
from backend.configuration import bool_env
from backend.configuration import float_env
from backend.configuration import int_env
from backend.configuration import optional_env
from backend.configuration import parse_max_pdf_pages
from backend.configuration import required_env
from .cli import parse_args
# python -m backend.ingest --limit 27

def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    args = parse_args()
    ingest(
        limit=args.limit,
        query=required_env("ARXIV_QUERY"),
        sort_by=required_env("ARXIV_SORT_BY"),
        sort_order=required_env("ARXIV_SORT_ORDER"),
        arxiv_request_delay_seconds=float_env("ARXIV_REQUEST_DELAY_SECONDS"),
        arxiv_429_max_retries=int_env("ARXIV_429_MAX_RETRIES"),
        arxiv_429_backoff_seconds=int_env("ARXIV_429_BACKOFF_SECONDS"),
        content_mode=required_env("ARXIV_CONTENT_MODE"),
        max_pdf_pages=parse_max_pdf_pages(required_env("ARXIV_MAX_PDF_PAGES")),
        pdf_timeout_seconds=int_env("ARXIV_PDF_TIMEOUT_SECONDS"),
        min_chunk_chars=int_env("ARXIV_MIN_CHUNK_CHARS"),
        max_chunk_chars=int_env("ARXIV_MAX_CHUNK_CHARS"),
        chunk_overlap_chars=int_env("ARXIV_CHUNK_OVERLAP_CHARS"),
        exclude_references=DEFAULT_EXCLUDE_REFERENCES,
        llm_author_check_first_page=bool_env("ARXIV_LLM_AUTHOR_CHECK_FIRST_PAGE"),
        local_pdf=optional_env("ARXIV_LOCAL_PDF") or "",
    )


if __name__ == "__main__":
    main()
