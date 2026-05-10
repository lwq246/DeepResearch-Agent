import argparse
from pathlib import Path

import fitz


def extract_with_fitz(pdf_path: Path, max_pages: int | None) -> str:
    doc = fitz.open(pdf_path)
    chunks: list[str] = []
    try:
        page_count = doc.page_count if max_pages is None else min(max(1, max_pages), doc.page_count)
        for page_index in range(page_count):
            try:
                text = doc.load_page(page_index).get_text("text")
            except Exception:  # noqa: BLE001
                continue
            if text.strip():
                chunks.append(text)
    finally:
        doc.close()

    return "\n".join(chunks).strip()


def extract_with_pymupdf4llm(pdf_path: Path, max_pages: int | None) -> str:
    import pymupdf4llm  # type: ignore[import-not-found]

    page_limit = None if max_pages is None else max(1, max_pages)
    if page_limit is None:
        result = pymupdf4llm.to_markdown(str(pdf_path))
    else:
        pages = list(range(page_limit))
        try:
            result = pymupdf4llm.to_markdown(str(pdf_path), pages=pages)
        except (TypeError, ValueError):
            # Some versions reject explicit page ranges; fall back to full-document extraction.
            result = pymupdf4llm.to_markdown(str(pdf_path))

    return str(result).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from one PDF and print a preview (pymupdf4llm first by default)."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to a local PDF file")
    parser.add_argument(
        "--engine",
        type=str,
        default="auto",
        choices=["auto", "pymupdf4llm", "fitz"],
        help="Extraction engine: auto (default), pymupdf4llm, or fitz",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to process (default: all pages)",
    )
    parser.add_argument(
        "--head-chars",
        type=int,
        default=3000,
        help="Number of characters to print from the start (default: 3000)",
    )
    parser.add_argument(
        "--tail-chars",
        type=int,
        default=1000,
        help="Number of characters to print from the end (default: 1000)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save the full extracted text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pdf_path.exists() or not args.pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {args.pdf_path}")

    max_pages = None if args.max_pages is None or args.max_pages <= 0 else args.max_pages
    engine_used = ""

    if args.engine == "fitz":
        extracted = extract_with_fitz(args.pdf_path, max_pages=max_pages)
        engine_used = "fitz"
    elif args.engine == "pymupdf4llm":
        extracted = extract_with_pymupdf4llm(args.pdf_path, max_pages=max_pages)
        engine_used = "pymupdf4llm"
    else:
        try:
            extracted = extract_with_pymupdf4llm(args.pdf_path, max_pages=max_pages)
            engine_used = "pymupdf4llm"
        except Exception as exc:  # noqa: BLE001
            print(f"pymupdf4llm unavailable/failed ({type(exc).__name__}); falling back to fitz.")
            extracted = extract_with_fitz(args.pdf_path, max_pages=max_pages)
            engine_used = "fitz"

    print(f"PDF: {args.pdf_path}")
    print(f"Engine: {engine_used}")
    print(f"Max pages: {'all' if max_pages is None else max_pages}")
    print(f"Extracted characters: {len(extracted)}")

    if args.out is not None:
        args.out.write_text(extracted, encoding="utf-8")
        print(f"Saved full extracted text to: {args.out}")

    print("\n--- EXTRACT PREVIEW (HEAD) ---")
    print(extracted[: args.head_chars])
    print("\n--- EXTRACT PREVIEW (TAIL) ---")
    print(extracted[-args.tail_chars :])


if __name__ == "__main__":
    main()
