from typing import Any

try:
    from ..graph_utils import unwrap_metadata
    from .shared import build_qdrant_document
    from .shared import can_take_chunk_for_paper
    from .shared import document_paper_group_key
    from .shared import increment_paper_chunk_count
    from .shared import metadata_authors
    from .shared import normalize_author_name
    from .shared import score_web_documents
except ImportError:
    from graph_utils import unwrap_metadata
    from nodes.shared import build_qdrant_document
    from nodes.shared import can_take_chunk_for_paper
    from nodes.shared import document_paper_group_key
    from nodes.shared import increment_paper_chunk_count
    from nodes.shared import metadata_authors
    from nodes.shared import normalize_author_name
    from nodes.shared import score_web_documents


def author_matches(metadata: dict[str, Any], requested_author: str) -> bool:
    requested = normalize_author_name(requested_author).casefold()
    if not requested:
        return False

    for candidate in metadata_authors(metadata):
        normalized = normalize_author_name(candidate).casefold()
        if not normalized:
            continue
        if requested in normalized or normalized in requested:
            return True
    return False


def retrieve_documents_by_author(
    vector_store: Any,
    question: str,
    requested_author: str,
    max_scan_points: int,
    max_author_candidates: int,
    max_context_chunks: int,
    max_chunks_per_paper: int,
    max_unique_papers: int,
) -> list[dict[str, Any]]:
    client = getattr(vector_store, "client", None)
    collection_name = getattr(vector_store, "collection_name", "")
    if client is None or not collection_name:
        return []

    author_candidates: list[dict[str, Any]] = []
    offset = None
    scanned_points = 0
    batch_size = min(256, max(1, max_scan_points))
    reached_candidate_limit = False

    while scanned_points < max_scan_points:
        limit = min(batch_size, max_scan_points - scanned_points)
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            break

        scanned_points += len(points)

        for point in points:
            payload = getattr(point, "payload", None)
            if not isinstance(payload, dict):
                continue

            raw_metadata = payload.get("metadata")
            if not isinstance(raw_metadata, dict):
                continue

            metadata = unwrap_metadata(raw_metadata)
            if not author_matches(metadata, requested_author):
                continue

            author_candidates.append(
                build_qdrant_document(
                    metadata=metadata,
                    content=str(payload.get("page_content", "")),
                    score=0.0,
                )
            )
            if len(author_candidates) >= max_author_candidates:
                reached_candidate_limit = True
                break

        if reached_candidate_limit:
            break

        if next_offset is None:
            break
        offset = next_offset

    if not author_candidates:
        return []

    ranked_candidates = score_web_documents(question, author_candidates)
    if not ranked_candidates:
        ranked_candidates = author_candidates

    documents: list[dict[str, Any]] = []
    paper_chunk_counts: dict[str, int] = {}
    for candidate in ranked_candidates:
        paper_key = document_paper_group_key(candidate)

        if not can_take_chunk_for_paper(
            paper_chunk_counts=paper_chunk_counts,
            paper_key=paper_key,
            max_chunks_per_paper=max_chunks_per_paper,
            max_unique_papers=max_unique_papers,
        ):
            continue

        documents.append(candidate)
        increment_paper_chunk_count(paper_chunk_counts=paper_chunk_counts, paper_key=paper_key)

        if len(documents) >= max_context_chunks:
            break

    return documents
