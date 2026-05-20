import argparse
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

#py tools\delete_qdrant_points.py --title "A dataset of clinically generated visual questions and answers about radiology images.pdf" --apply 
def required_env(name: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    value = raw.strip()
    if not value:
        raise RuntimeError(f"Empty required environment variable: {name}")
    return value


def build_filter(args: argparse.Namespace) -> models.Filter:
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
        raise ValueError("Provide at least one filter: --paper-id, --title, or --source-url")

    return models.Filter(must=must)


def count_matches(
    client: QdrantClient,
    collection: str,
    query_filter: models.Filter,
    batch_size: int,
) -> int:
    total = 0
    offset: models.PointId | None = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=query_filter,
            limit=batch_size,
            with_payload=False,
            with_vectors=False,
            offset=offset,
        )
        total += len(points)
        if next_offset is None:
            break
        offset = next_offset

    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete Qdrant points for one paper using metadata filters."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="",
        help="Qdrant URL (default: QDRANT_URL or http://localhost:6333)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="",
        help="Qdrant collection (default: QDRANT_COLLECTION or arxiv_docs)",
    )

    parser.add_argument("--paper-id", type=str, default="", help="metadata.paper_id exact match")
    parser.add_argument("--title", type=str, default="", help="metadata.title exact match")
    parser.add_argument("--source-url", type=str, default="", help="metadata.source_url exact match")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Scroll batch size for counting matches (default: 256)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete points. Without this flag, runs a dry-run count only.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for delete operation completion before exiting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()

    qdrant_url = args.url or required_env("QDRANT_URL")
    collection = args.collection or required_env("QDRANT_COLLECTION")

    client = QdrantClient(url=qdrant_url)
    query_filter = build_filter(args)

    matches = count_matches(
        client=client,
        collection=collection,
        query_filter=query_filter,
        batch_size=max(1, args.batch_size),
    )

    print(f"Qdrant URL: {qdrant_url}")
    print(f"Collection: {collection}")
    print(f"Matched points: {matches}")

    if not args.apply:
        print("Dry-run only. Re-run with --apply to delete these points.")
        return 0

    if matches == 0:
        print("No matching points found. Nothing deleted.")
        return 0

    client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(filter=query_filter),
        wait=args.wait,
    )

    remaining = count_matches(
        client=client,
        collection=collection,
        query_filter=query_filter,
        batch_size=max(1, args.batch_size),
    )
    deleted = matches - remaining

    print(f"Delete requested. Deleted: {deleted}")
    print(f"Remaining matches: {remaining}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
