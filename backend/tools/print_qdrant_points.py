import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


def required_env(name: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    value = raw.strip()
    if not value:
        raise RuntimeError(f"Empty required environment variable: {name}")
    return value


def normalize_base_section(title: str) -> str:
    stripped = title.strip()
    stripped = re.sub(r"\s+\(part\s+\d+\)\s*$", "", stripped, flags=re.I)
    return stripped.casefold()


def point_payload(point: Any) -> dict[str, Any]:
    payload = getattr(point, "payload", None)
    if isinstance(payload, dict):
        return payload
    if isinstance(point, dict):
        maybe_payload = point.get("payload")
        if isinstance(maybe_payload, dict):
            return maybe_payload
    return {}


def point_section_title(point: Any) -> str:
    payload = point_payload(point)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    title = metadata.get("section_title", "")
    return str(title).strip()


def point_to_dict(point: Any) -> dict[str, Any]:
    if hasattr(point, "model_dump"):
        return point.model_dump()
    if hasattr(point, "dict"):
        return point.dict()
    if isinstance(point, dict):
        return point
    return {"value": str(point)}


def build_filter(args: argparse.Namespace) -> models.Filter | None:
    must: list[models.Condition] = []

    if args.paper_id:
        must.append(
            models.FieldCondition(
                key="metadata.paper_id",
                match=models.MatchValue(value=args.paper_id),
            )
        )

    if args.title:
        must.append(
            models.FieldCondition(
                key="metadata.title",
                match=models.MatchValue(value=args.title),
            )
        )

    if args.source_url:
        must.append(
            models.FieldCondition(
                key="metadata.source_url",
                match=models.MatchValue(value=args.source_url),
            )
        )

    if not must:
        return None

    return models.Filter(must=must)


def section_matches(found: str, wanted: str, mode: str) -> bool:
    found_norm = found.casefold()
    wanted_norm = wanted.casefold()

    if mode == "exact":
        return found_norm == wanted_norm
    if mode == "contains":
        return wanted_norm in found_norm

    return normalize_base_section(found) == normalize_base_section(wanted)


def iter_points(
    client: QdrantClient,
    collection: str,
    scroll_filter: models.Filter | None,
    with_vectors: bool,
    batch_size: int,
):
    offset: models.PointId | None = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=batch_size,
            with_payload=True,
            with_vectors=with_vectors,
            offset=offset,
        )
        if not points:
            break

        for point in points:
            yield point

        if next_offset is None:
            break
        offset = next_offset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print all full Qdrant points for a specific paper (and optional section), without truncation."
    )
    parser.add_argument("--url", type=str, default="", help="Qdrant URL (default: QDRANT_URL or http://localhost:6333)")
    parser.add_argument(
        "--collection",
        type=str,
        default="",
        help="Qdrant collection (default: QDRANT_COLLECTION or arxiv_docs)",
    )

    parser.add_argument("--paper-id", type=str, default="", help="metadata.paper_id exact match")
    parser.add_argument("--title", type=str, default="", help="metadata.title exact match")
    parser.add_argument("--source-url", type=str, default="", help="metadata.source_url exact match")

    parser.add_argument("--section-title", type=str, default="", help="Section title to filter")
    parser.add_argument(
        "--section-match",
        type=str,
        choices=["base", "exact", "contains"],
        default="base",
        help="How to match --section-title (default: base)",
    )

    parser.add_argument("--batch-size", type=int, default=256, help="Scroll batch size (default: 256)")
    parser.add_argument("--no-vectors", action="store_true", help="Do not include vectors in output")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional JSON output file path (writes full points list)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()

    qdrant_url = args.url or required_env("QDRANT_URL")
    collection = args.collection or required_env("QDRANT_COLLECTION")
    with_vectors = not args.no_vectors

    scroll_filter = build_filter(args)
    if scroll_filter is None:
        raise ValueError("Provide at least one paper filter: --paper-id, --title, or --source-url")

    client = QdrantClient(url=qdrant_url)

    points_out: list[dict[str, Any]] = []
    for point in iter_points(
        client=client,
        collection=collection,
        scroll_filter=scroll_filter,
        with_vectors=with_vectors,
        batch_size=max(1, args.batch_size),
    ):
        if args.section_title:
            found = point_section_title(point)
            if not section_matches(found=found, wanted=args.section_title, mode=args.section_match):
                continue

        points_out.append(point_to_dict(point))

    print(f"Qdrant URL: {qdrant_url}")
    print(f"Collection: {collection}")
    print(f"Points found: {len(points_out)}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(points_out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote full points JSON to: {output_path}")
        return 0

    for index, point in enumerate(points_out, start=1):
        print()
        print(f"=== POINT {index} / {len(points_out)} ===")
        print(json.dumps(point, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
