import argparse
import os

from .env import bool_env
from .env import float_env_default
from .env import int_env_default
from .env import parse_max_pdf_pages


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
