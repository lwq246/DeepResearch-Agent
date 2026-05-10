import argparse
import os
import re
from typing import Any, Iterable

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


def point_payload(point: Any) -> dict[str, Any]:
    payload = getattr(point, "payload", None)
    if isinstance(payload, dict):
        return payload
    if isinstance(point, dict):
        maybe_payload = point.get("payload")
        if isinstance(maybe_payload, dict):
            return maybe_payload
    return {}


def base_section_title(title: str) -> str:
    return re.sub(r"\s+\(part\s+\d+\)\s*$", "", title.strip(), flags=re.I)


def iter_points(
    client: QdrantClient,
    collection_name: str,
    scroll_filter: models.Filter | None,
    batch_size: int = 256,
) -> Iterable[Any]:
    offset: models.PointId | None = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=batch_size,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )

        if not points:
            break

        for point in points:
            yield point

        if next_offset is None:
            break
        offset = next_offset


def build_filter(args: argparse.Namespace) -> models.Filter | None:
    must_conditions: list[models.Condition] = []

    if args.paper_id:
        must_conditions.append(
            models.FieldCondition(
                key="metadata.paper_id",
                match=models.MatchValue(value=args.paper_id),
            )
        )

    if args.title:
        must_conditions.append(
            models.FieldCondition(
                key="metadata.title",
                match=models.MatchValue(value=args.title),
            )
        )

    if args.source_url:
        must_conditions.append(
            models.FieldCondition(
                key="metadata.source_url",
                match=models.MatchValue(value=args.source_url),
            )
        )

    if not must_conditions:
        return None

    return models.Filter(must=must_conditions)


def print_headers(args: argparse.Namespace) -> int:
    load_dotenv()

    qdrant_url = args.url or os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = args.collection or os.getenv("QDRANT_COLLECTION", "arxiv_docs")

    client = QdrantClient(url=qdrant_url)
    scroll_filter = build_filter(args)

    if scroll_filter is None:
        raise ValueError("Provide at least one filter: --paper-id, --title, or --source-url")

    by_title: dict[str, dict[str, Any]] = {}

    for point in iter_points(client, collection_name=collection_name, scroll_filter=scroll_filter):
        payload = point_payload(point)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

        raw_title = str(metadata.get("section_title", "")).strip()
        if not raw_title:
            continue

        normalized_title = base_section_title(raw_title)
        try:
            section_index = int(metadata.get("section_index", 10**9))
        except (TypeError, ValueError):
            section_index = 10**9

        current = by_title.get(normalized_title)
        if current is None or section_index < current["section_index"]:
            by_title[normalized_title] = {
                "section_title": normalized_title,
                "section_index": section_index,
            }

    ordered = sorted(by_title.values(), key=lambda item: (item["section_index"], item["section_title"].lower()))

    print(f"Qdrant URL: {qdrant_url}")
    print(f"Collection: {collection_name}")
    print(f"Section headers found: {len(ordered)}")

    for idx, item in enumerate(ordered, start=1):
        print(f"{idx}. {item['section_title']}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print section headers from Qdrant points for one paper.")
    parser.add_argument("--url", type=str, default="", help="Qdrant URL (default: from QDRANT_URL)")
    parser.add_argument(
        "--collection",
        type=str,
        default="",
        help="Qdrant collection name (default: from QDRANT_COLLECTION)",
    )
    parser.add_argument("--paper-id", type=str, default="", help="Exact metadata.paper_id match")
    parser.add_argument("--title", type=str, default="", help="Exact metadata.title match")
    parser.add_argument("--source-url", type=str, default="", help="Exact metadata.source_url match")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(print_headers(parse_args()))
