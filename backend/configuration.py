import os
from functools import lru_cache

import arxiv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore


def required_env(name: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    value = raw.strip()
    if not value:
        raise RuntimeError(f"Empty required environment variable: {name}")
    return value


def optional_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def openai_client_kwargs() -> dict[str, str]:
    base_url = optional_env("OPENAI_BASE_URL")
    if base_url is None:
        return {}
    return {"base_url": base_url}


def int_env(name: str) -> int:
    raw = required_env(name)
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"Invalid integer for {name}: {raw}")


def float_env(name: str) -> float:
    raw = required_env(name)
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"Invalid float for {name}: {raw}")


def bool_env(name: str) -> bool:
    raw = required_env(name).lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean for {name}: {raw}")


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


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=required_env("OPENAI_EMBEDDING_MODEL"),
        **openai_client_kwargs(),
    )


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    validate_embeddings = bool_env("QDRANT_VALIDATE_EMBEDDINGS")
    validate_collection_config = bool_env("QDRANT_VALIDATE_COLLECTION_CONFIG")
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        collection_name=required_env("QDRANT_COLLECTION"),
        url=required_env("QDRANT_URL"),
        validate_embeddings=validate_embeddings,
        validate_collection_config=validate_collection_config,
    )


@lru_cache(maxsize=1)
def get_search_tool() -> TavilySearchResults:
    return TavilySearchResults(max_results=5)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=required_env("OPENAI_CHAT_MODEL"),
        temperature=0,
        max_tokens=int_env("OPENAI_ANSWER_MAX_TOKENS"),
        **openai_client_kwargs(),
    )


@lru_cache(maxsize=1)
def get_planner_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=required_env("OPENAI_PLANNER_MODEL"),
        temperature=0,
        max_tokens=int_env("OPENAI_PLANNER_MAX_TOKENS"),
        **openai_client_kwargs(),
    )


@lru_cache(maxsize=1)
def get_query_rewrite_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=required_env("OPENAI_QUERY_REWRITE_MODEL"),
        temperature=0,
        max_tokens=int_env("OPENAI_QUERY_REWRITE_MAX_TOKENS"),
        **openai_client_kwargs(),
    )


@lru_cache(maxsize=1)
def get_reflection_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=required_env("OPENAI_REFLECTION_MODEL"),
        temperature=0,
        max_tokens=int_env("OPENAI_REFLECTION_MAX_TOKENS"),
        **openai_client_kwargs(),
    )
