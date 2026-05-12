import os

import arxiv


def openai_client_kwargs() -> dict[str, str]:
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not base_url:
        return {}
    return {"base_url": base_url}


def bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def int_env_default(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def float_env_default(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def parse_max_pdf_pages(value: str | None) -> int | None:
    if value is None:
        return None

    raw = str(value).strip().lower()
    if raw in {"", "all", "none", "null", "0", "-1"}:
        return None

    parsed = int(raw)
    if parsed <= 0:
        return None
    return parsed


def parse_sort_criterion(sort_by: str) -> arxiv.SortCriterion:
    normalized = sort_by.strip().lower()
    if normalized == "submitteddate":
        return arxiv.SortCriterion.SubmittedDate
    if normalized == "lastupdateddate":
        return arxiv.SortCriterion.LastUpdatedDate
    return arxiv.SortCriterion.Relevance


def parse_sort_order(sort_order: str) -> arxiv.SortOrder:
    normalized = sort_order.strip().lower()
    if normalized == "ascending":
        return arxiv.SortOrder.Ascending
    return arxiv.SortOrder.Descending


def paper_id_from_source_url(source_url: str | None) -> str:
    if not source_url:
        return "unknown"
    return source_url.rstrip("/").split("/")[-1]
