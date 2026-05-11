import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from ingest import build_documents_from_local_pdf, openai_client_kwargs, parse_max_pdf_pages


def bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_paper_dir = script_dir.parent / "paper"

    parser = argparse.ArgumentParser(
        description="Ingest all local PDFs from a folder into Qdrant."
    )
    parser.add_argument(
        "--paper-dir",
        type=str,
        default=str(default_paper_dir),
        help="Folder containing PDF files (default: ../paper)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="Glob pattern for files in --paper-dir (default: *.pdf)",
    )
    parser.add_argument(
        "--max-pdf-pages",
        type=str,
        default=os.getenv("ARXIV_MAX_PDF_PAGES", "all"),
        help="Maximum pages to parse per PDF (default: all)",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=int(os.getenv("ARXIV_MIN_CHUNK_CHARS", "80")),
        help="Minimum characters required for a chunk",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=int(os.getenv("ARXIV_MAX_CHUNK_CHARS", "2500")),
        help="Maximum characters before splitting a chunk",
    )
    parser.add_argument(
        "--chunk-overlap-chars",
        type=int,
        default=int(os.getenv("ARXIV_CHUNK_OVERLAP_CHARS", "500")),
        help="Trailing overlap repeated in the next chunk",
    )
    parser.add_argument(
        "--exclude-references",
        action="store_true",
        default=bool_env("ARXIV_EXCLUDE_REFERENCES", True),
        help="Skip references/bibliography sections",
    )
    parser.add_argument(
        "--keep-references",
        dest="exclude_references",
        action="store_false",
        help="Do not skip references/bibliography sections",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.min_chunk_chars <= 0:
        raise ValueError("min_chunk_chars must be greater than 0.")
    if args.max_chunk_chars <= args.min_chunk_chars:
        raise ValueError("max_chunk_chars must be greater than min_chunk_chars.")
    if args.chunk_overlap_chars < 0:
        raise ValueError("chunk_overlap_chars must be 0 or greater.")
    if args.chunk_overlap_chars >= args.max_chunk_chars:
        raise ValueError("chunk_overlap_chars must be smaller than max_chunk_chars.")

    max_pdf_pages = parse_max_pdf_pages(args.max_pdf_pages)

    paper_dir = Path(args.paper_dir).expanduser().resolve()
    if not paper_dir.exists() or not paper_dir.is_dir():
        raise FileNotFoundError(f"Paper directory not found: {paper_dir}")

    pdf_files = sorted(path for path in paper_dir.glob(args.pattern) if path.is_file())
    if not pdf_files:
        raise RuntimeError(f"No files matched '{args.pattern}' under: {paper_dir}")

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "arxiv_docs")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    print(f"Found {len(pdf_files)} PDF files in: {paper_dir}")

    all_documents = []
    for pdf_path in pdf_files:
        print(f"Building chunks for: {pdf_path.name}")
        documents = build_documents_from_local_pdf(
            pdf_path=str(pdf_path),
            max_pdf_pages=max_pdf_pages,
            min_chunk_chars=args.min_chunk_chars,
            max_chunk_chars=args.max_chunk_chars,
            chunk_overlap_chars=args.chunk_overlap_chars,
            exclude_references=args.exclude_references,
        )
        all_documents.extend(documents)

    if not all_documents:
        raise RuntimeError("No chunks were created from the provided PDFs.")

    print(
        f"Uploading {len(all_documents)} chunks to Qdrant collection '{collection_name}' at {qdrant_url}"
    )

    embeddings = OpenAIEmbeddings(model=embedding_model, **openai_client_kwargs())
    QdrantVectorStore.from_documents(
        documents=all_documents,
        embedding=embeddings,
        collection_name=collection_name,
        url=qdrant_url,
    )

    print("Ingestion completed.")


if __name__ == "__main__":
    main()
