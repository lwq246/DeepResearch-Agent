from __future__ import annotations

import argparse
import json
import os
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models


def create_client() -> QdrantClient:
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=qdrant_url)


def find_points_by_exact_title(
    client: QdrantClient,
    collection_name: str,
    paper_title: str,
    batch_size: int = 256,
) -> list[Any]:
    points: list[Any] = []
    next_offset: models.PointId | None = None

    filter_by_title = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.title",
                match=models.MatchValue(value=paper_title),
            )
        ]
    )

    while True:
        page, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_by_title,
            with_payload=True,
            with_vectors=False,
            limit=batch_size,
            offset=next_offset,
        )
        points.extend(page)
        if next_offset is None:
            break

    return points


def print_points(points: list[Any], full_text: bool, preview_chars: int) -> None:
    print(f"Found {len(points)} point(s).")
    for idx, point in enumerate(points, start=1):
        payload = point.payload or {}
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        page_content = payload.get("page_content", "") if isinstance(payload, dict) else ""
        content = str(page_content).strip()
        preview = content.replace("\n", " ").strip()
        if len(preview) > preview_chars:
            preview = preview[:preview_chars] + "..."

        print(f"\n[{idx}] id={point.id}")
        print(f"title={metadata.get('title', '')}")
        print(f"section={metadata.get('section', '')} chunk_index={metadata.get('chunk_index', '')}")
        print(f"source_url={metadata.get('source_url', '')}")
        if full_text:
            print("full_text=")
            print(content)
        else:
            print(f"preview={preview}")


def dump_points_json(points: list[Any], output_path: str) -> None:
    rows: list[dict[str, Any]] = []
    for point in points:
        rows.append(
            {
                "id": point.id,
                "payload": point.payload,
            }
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    print(f"\nSaved results to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve all Qdrant points for a specific paper title."
    )
    parser.add_argument(
        "paper_title",
        help="Exact paper title stored in metadata.title",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "arxiv_docs"),
        help="Qdrant collection name (default: env QDRANT_COLLECTION or arxiv_docs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Scroll page size (default: 256)",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional output file path for full JSON dump",
    )
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Print full page_content for each point instead of preview.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=220,
        help="Preview length in characters when --full-text is not used (default: 220)",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    client = create_client()
    points = find_points_by_exact_title(
        client=client,
        collection_name=args.collection,
        paper_title=args.paper_title,
        batch_size=max(1, args.batch_size),
    )

    print_points(
        points,
        full_text=args.full_text,
        preview_chars=max(40, args.preview_chars),
    )
    if args.json_out:
        dump_points_json(points, args.json_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())