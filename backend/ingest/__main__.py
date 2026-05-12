from . import ingest
from . import parse_args


def main() -> None:
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


if __name__ == "__main__":
    main()
