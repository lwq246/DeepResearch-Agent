import json
import re
from datetime import date
from math import sqrt
from typing import Any, Sequence

from ..configuration import bool_env
from ..configuration import get_embeddings
from ..graph_utils import safe_float


def parse_json_object(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    candidates = [text]

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.insert(0, text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def llm_json_response(llm: Any, system_prompt: str, human_prompt: str) -> dict[str, Any] | None:
    response = llm.invoke(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    response_text = response.content if isinstance(response.content, str) else str(response.content)
    return parse_json_object(response_text)


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if value is None:
        return default
    return bool(value)


def summarize_documents_for_prompt(documents: list[dict[str, Any]], max_items: int = 12) -> str:
    if not documents:
        return "No documents."

    lines: list[str] = []
    for idx, doc in enumerate(documents[:max_items], start=1):
        title = str(doc.get("title", ""))
        origin = str(doc.get("origin", ""))
        source = str(doc.get("source", ""))
        published = str(doc.get("published", doc.get("updated", "")))
        score = safe_float(doc.get("score", 0.0))
        snippet = str(doc.get("content", "")).replace("\n", " ")
        lines.append(
            f"{idx}. origin={origin} score={score:.3f} title={title} "
            f"published={published or '-'} source={source} snippet={snippet}"
        )

    return "\n".join(lines)


def current_date_iso() -> str:
    return date.today().isoformat()


def resolve_requires_web(state: Any, _question: str) -> bool:
    return (
        bool_env("FORCE_WEB_FALLBACK")
        or bool(state.get("requires_web", False))
    )


def metadata_authors(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("authors", [])
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        return [part.strip() for part in re.split(r"[,;]", raw) if part.strip()]
    return []


def build_qdrant_document(
    metadata: dict[str, Any],
    content: str,
    score: float,
) -> dict[str, Any]:
    return {
        "title": str(metadata.get("title", "ArXiv document")),
        "content": content,
        "source": str(metadata.get("source_url", metadata.get("source", ""))),
        "paper_id": str(metadata.get("paper_id", "")),
        "section": str(metadata.get("section", metadata.get("section_title", ""))),
        "authors": metadata_authors(metadata),
        "published": str(metadata.get("published", "")),
        "score": float(score),
        "origin": "qdrant",
    }


def metadata_paper_group_key(metadata: dict[str, Any]) -> str:
    return str(
        metadata.get("paper_id")
        or metadata.get("source_url")
        or metadata.get("source")
        or metadata.get("title")
        or "unknown"
    )


def document_paper_group_key(document: dict[str, Any]) -> str:
    return str(document.get("paper_id") or document.get("source") or document.get("title") or "unknown")


def document_dedup_key(document: dict[str, Any]) -> str:
    paper_id = str(document.get("paper_id", "")).strip().lower()
    source = str(document.get("source", "")).strip().lower()
    title = str(document.get("title", "")).strip().lower()
    section = str(document.get("section", "")).strip().lower()
    content_prefix = str(document.get("content", "")).strip().lower()[:200]
    return "||".join([paper_id, source, title, section, content_prefix])


def can_take_chunk_for_paper(
    paper_chunk_counts: dict[str, int],
    paper_key: str,
    max_chunks_per_paper: int,
    max_unique_papers: int,
) -> bool:
    current_chunks = paper_chunk_counts.get(paper_key, 0)
    if current_chunks >= max_chunks_per_paper:
        return False
    if current_chunks == 0 and len(paper_chunk_counts) >= max_unique_papers:
        return False
    return True


def increment_paper_chunk_count(paper_chunk_counts: dict[str, int], paper_key: str) -> None:
    paper_chunk_counts[paper_key] = paper_chunk_counts.get(paper_key, 0) + 1


def temporal_window_label(year: int, month: int) -> str:
    month_name = date(year, month, 1).strftime("%B")
    return f"{month_name} {year}"


def document_mentions_target_month(document: dict[str, Any], year: int, month: int) -> bool:
    haystack = (
        f"{document.get('title', '')}\n"
        f"{document.get('content', '')}\n"
        f"{document.get('source', '')}\n"
        f"{document.get('published', '')}\n"
        f"{document.get('updated', '')}"
    ).lower()

    month_full = date(year, month, 1).strftime("%B").lower()
    month_abbr = date(year, month, 1).strftime("%b").lower()
    year_str = str(year)
    month_num = f"{month:02d}"

    candidates = {
        f"{month_full} {year_str}",
        f"{month_abbr} {year_str}",
        f"{year_str}-{month_num}",
        f"{year_str}/{month_num}",
        f"{month_num}/{year_str}",
        f"{month_num}-{year_str}",
    }
    return any(token in haystack for token in candidates)


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def lexical_overlap_score(question: str, text: str) -> float:
    question_terms = set(re.findall(r"[a-z0-9]{3,}", question.lower()))
    if not question_terms:
        return 0.0

    text_terms = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
    if not text_terms:
        return 0.0

    return len(question_terms & text_terms) / len(question_terms)


def score_web_documents(question: str, web_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not web_documents:
        return []

    query = question.strip()
    document_texts = [
        (
            f"{str(doc.get('title', ''))}\n"
            f"{str(doc.get('content', ''))}\n"
            f"{str(doc.get('source', ''))}"
        )
        for doc in web_documents
    ]

    scores: list[float]
    try:
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(query)
        document_vectors = embeddings.embed_documents(document_texts)
        if len(document_vectors) != len(document_texts):
            raise ValueError("Embedding result count mismatch")
        scores = [cosine_similarity(query_vector, vector) for vector in document_vectors]
    except Exception:
        scores = [lexical_overlap_score(query, text) for text in document_texts]

    scored_documents: list[dict[str, Any]] = []
    for document, score in zip(web_documents, scores):
        scored_document = dict(document)
        scored_document["score"] = float(score)
        scored_documents.append(scored_document)

    scored_documents.sort(key=lambda item: safe_float(item.get("score", 0.0)), reverse=True)
    return scored_documents
