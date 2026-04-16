import os
from functools import lru_cache

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore


def openai_client_kwargs() -> dict[str, str]:
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not base_url:
        return {}
    return {"base_url": base_url}


def int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


def float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_chat_model(primary_env: str) -> str:
    primary = os.getenv(primary_env, "").strip()
    if primary:
        return primary

    shared = os.getenv("OPENAI_CHAT_MODEL", "").strip()
    if shared:
        return shared

    return "gpt-4o-mini"


def resolve_max_tokens(primary_env: str, default: int) -> int:
    primary = os.getenv(primary_env)
    if primary is not None:
        try:
            value = int(primary)
            if value > 0:
                return value
        except ValueError:
            pass
    return int_env("OPENAI_MAX_TOKENS", default)


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        **openai_client_kwargs(),
    )


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        collection_name=os.getenv("QDRANT_COLLECTION", "arxiv_docs"),
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )


@lru_cache(maxsize=1)
def get_search_tool() -> TavilySearchResults:
    return TavilySearchResults(max_results=5)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0,
        max_tokens=resolve_max_tokens("OPENAI_ANSWER_MAX_TOKENS", 1200),
        **openai_client_kwargs(),
    )


@lru_cache(maxsize=1)
def get_planner_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=resolve_chat_model("OPENAI_PLANNER_MODEL"),
        temperature=0,
        max_tokens=resolve_max_tokens("OPENAI_PLANNER_MAX_TOKENS", 192),
        **openai_client_kwargs(),
    )


@lru_cache(maxsize=1)
def get_query_rewrite_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=resolve_chat_model("OPENAI_QUERY_REWRITE_MODEL"),
        temperature=0,
        max_tokens=resolve_max_tokens("OPENAI_QUERY_REWRITE_MAX_TOKENS", 192),
        **openai_client_kwargs(),
    )


@lru_cache(maxsize=1)
def get_reflection_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=resolve_chat_model("OPENAI_REFLECTION_MODEL"),
        temperature=0,
        max_tokens=resolve_max_tokens("OPENAI_REFLECTION_MAX_TOKENS", 256),
        **openai_client_kwargs(),
    )
