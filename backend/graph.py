import os
import re
from math import sqrt
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal, Sequence

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, StateGraph

try:
    from .models.graph_state import GraphState
except ImportError:
    from models.graph_state import GraphState


load_dotenv()


COMPARISON_PATTERNS = [
    r"\bcompare\s+(?P<a>.+?)\s+(?:and|vs\.?|versus)\s+(?P<b>.+?)(?:[?.!]|$)",
    r"\bcomparison\s+between\s+(?P<a>.+?)\s+(?:and|vs\.?|versus)\s+(?P<b>.+?)(?:[?.!]|$)",
    r"\bdifference\s+between\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)(?:[?.!]|$)",
    r"\b(?P<a>[a-zA-Z0-9][a-zA-Z0-9\-+/. ]{1,60})\s+(?:vs\.?|versus)\s+(?P<b>[a-zA-Z0-9][a-zA-Z0-9\-+/. ]{1,60})(?:[?.!]|$)",
]

ENTITY_SUFFIX_CLEANUPS = (
    "based on research findings",
    "based on findings",
    "based on research",
    "in research",
    "from research",
)

ENTITY_TRAILING_QUALIFIER_CLEANUPS = (
    "for domain qa",
    "for domain question answering",
    "for question answering",
    "for domain-specific qa",
    "for domain-specific question answering",
    "in domain qa",
    "in question answering",
)

STATIC_ENTITY_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("rag", "retrieval augmented generation", "retrieval-augmented generation"),
    (
        "fine-tuning",
        "fine tuning",
        "finetuning",
        "supervised fine-tuning",
        "sft",
        "instruction tuning",
    ),
    ("rlhf", "reinforcement learning from human feedback"),
    ("dpo", "direct preference optimization"),
    ("qa", "question answering", "question-answering"),
)


def append_trace(state: GraphState, message: str) -> list[str]:
    trace = list(state.get("react_trace", []))
    trace.append(message)
    return trace


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def prefers_web_only(question: str) -> bool:
    q = question.strip().lower()
    if not q:
        return False

    web_only_markers = [
        "online",
        "web",
        "internet",
        "search online",
        "search the web",
        "from online",
        "from the web",
        "not qdrant",
        "not the qdrant",
        "not qdrant database",
        "outside qdrant",
        "instead of qdrant",
    ]
    return any(marker in q for marker in web_only_markers)


def query_requires_web(question: str) -> bool:
    q = question.strip().lower()
    if not q:
        return False

    recency_markers = [
        "latest",
        "today",
        "current",
        "this week",
        "this month",
        "this year",
        "recent news",
        "announcement",
        "announcements",
        "release",
        "released",
        "launch",
        "launched",
        "what happened",
    ]
    if any(marker in q for marker in recency_markers):
        return True

    # Questions tied to explicit years are often recency-sensitive and benefit from web checks.
    return re.search(r"\b(20[2-9][0-9])\b", q) is not None


def extract_years(text: str) -> list[str]:
    return re.findall(r"\b(20[2-9][0-9])\b", text)


def target_year_for_question(question: str) -> str | None:
    years = extract_years(question)
    if years:
        return years[-1]
    if query_requires_web(question):
        return str(datetime.now().year)
    return None


def build_web_query(question: str) -> str:
    query = question.strip()
    if not query_requires_web(query):
        return query

    month_name = datetime.now().strftime("%B")
    year = str(datetime.now().year)
    if extract_years(query):
        return f"{query} latest official announcements"
    return f"{query} {month_name} {year} latest official announcements"


def document_mentions_year(document: dict[str, Any], year: str) -> bool:
    haystack = (
        f"{document.get('title', '')}\n"
        f"{document.get('content', '')}\n"
        f"{document.get('source', '')}"
    )
    return re.search(rf"\b{re.escape(year)}\b", haystack) is not None


def contains_any_phrase(text: str, phrases: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def normalize_comparison_entity(entity: str) -> str:
    cleaned = entity.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"^[\"'`]+|[\"'`]+$", "", cleaned)
    cleaned = re.sub(r"[\s,;:!.?]+$", "", cleaned)
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)

    for suffix in ENTITY_SUFFIX_CLEANUPS:
        if cleaned.endswith(f" {suffix}"):
            cleaned = cleaned[: -(len(suffix) + 1)].strip()

    for qualifier in ENTITY_TRAILING_QUALIFIER_CLEANUPS:
        if cleaned.endswith(f" {qualifier}"):
            cleaned = cleaned[: -(len(qualifier) + 1)].strip()

    cleaned = re.sub(r"\s+(for|in|on)\s+(domain\s+)?qa$", "", cleaned).strip()
    cleaned = re.sub(r"\s+(for|in|on)\s+question\s*[- ]?answering$", "", cleaned).strip()

    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned


def extract_comparison_entities(question: str) -> tuple[str, str] | None:
    for pattern in COMPARISON_PATTERNS:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if not match:
            continue

        entity_a = normalize_comparison_entity(match.group("a"))
        entity_b = normalize_comparison_entity(match.group("b"))
        if not entity_a or not entity_b or entity_a == entity_b:
            continue
        return entity_a, entity_b

    return None


def build_question_alias_map(question: str) -> dict[str, set[str]]:
    alias_map: dict[str, set[str]] = {}
    for match in re.finditer(r"([a-zA-Z][a-zA-Z0-9\- ]{2,})\s*\(([^)]+)\)", question):
        long_form = normalize_comparison_entity(match.group(1))
        short_form = normalize_comparison_entity(match.group(2))
        if not long_form or not short_form:
            continue
        alias_map.setdefault(long_form, set()).add(short_form)
        alias_map.setdefault(short_form, set()).add(long_form)
    return alias_map


def entity_aliases(entity: str, alias_map: dict[str, set[str]]) -> tuple[str, ...]:
    aliases: set[str] = {entity}
    aliases.update(alias_map.get(entity, set()))

    expanded: set[str] = set(alias for alias in aliases if alias)
    for alias in list(expanded):
        alias_lower = alias.lower()
        for group in STATIC_ENTITY_ALIAS_GROUPS:
            if any(term in alias_lower for term in group):
                expanded.update(group)

    normalized: set[str] = set()
    for alias in expanded:
        cleaned = normalize_comparison_entity(alias)
        if not cleaned:
            continue
        normalized.add(cleaned)
        normalized.add(cleaned.replace("-", " "))
        normalized.add(cleaned.replace(" ", "-"))

    return tuple(alias for alias in normalized if alias)


def comparison_entity_coverage(
    documents: list[dict[str, Any]],
    entity_a: str,
    entity_b: str,
    alias_map: dict[str, set[str]],
) -> tuple[bool, bool]:
    aliases_a = entity_aliases(entity_a, alias_map)
    aliases_b = entity_aliases(entity_b, alias_map)
    has_entity_a = False
    has_entity_b = False

    for document in documents:
        haystack = (
            f"{document.get('title', '')}\n"
            f"{document.get('content', '')}\n"
            f"{document.get('section', '')}"
        )
        if not has_entity_a and contains_any_phrase(haystack, aliases_a):
            has_entity_a = True
        if not has_entity_b and contains_any_phrase(haystack, aliases_b):
            has_entity_b = True
        if has_entity_a and has_entity_b:
            break

    return has_entity_a, has_entity_b


def unwrap_metadata(raw_metadata: dict[str, Any]) -> dict[str, Any]:
    nested = raw_metadata.get("metadata")
    if isinstance(nested, dict):
        merged = dict(raw_metadata)
        merged.update(nested)
        return merged
    return raw_metadata


def openai_client_kwargs() -> dict[str, str]:
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not base_url:
        return {}
    return {"base_url": base_url}


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
        **openai_client_kwargs(),
    )


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
    except Exception:  # noqa: BLE001
        # Fallback keeps relevance scoring available even if embedding calls fail.
        scores = [lexical_overlap_score(query, text) for text in document_texts]

    scored_documents: list[dict[str, Any]] = []
    for document, score in zip(web_documents, scores):
        scored_document = dict(document)
        scored_document["score"] = float(score)
        scored_documents.append(scored_document)

    scored_documents.sort(key=lambda item: safe_float(item.get("score", 0.0)), reverse=True)
    return scored_documents


def react_plan(state: GraphState) -> dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])
    step = int(state.get("react_step", 0)) + 1
    max_steps = int_env("REACT_MAX_STEPS", 4)

    has_local_docs = any(str(doc.get("origin", "")) == "qdrant" for doc in documents)
    has_web_docs = any(str(doc.get("origin", "")) == "web" for doc in documents)
    web_only = prefers_web_only(question)
    should_force_web = bool_env("FORCE_WEB_FALLBACK", False) or query_requires_web(question) or web_only

    if step >= max_steps:
        action = "build_context"
        thought = "Reached max ReAct steps; building context from current observations."
    elif state.get("fallback", False):
        if int(state.get("web_attempts", 0)) < int_env("MAX_WEB_ATTEMPTS", 2):
            action = "web_search"
            thought = "Evidence quality is insufficient; run another web retrieval step."
        else:
            action = "build_context"
            thought = "Reached web retry limit; proceed with best available evidence."
    elif web_only and not has_web_docs:
        action = "web_search"
        thought = "Question explicitly asks for online results; run web retrieval before local search."
    elif not has_local_docs:
        action = "retrieve"
        thought = "Need local evidence first; run vector retrieval."
    elif should_force_web and not has_web_docs:
        action = "web_search"
        thought = "Local evidence may be weak/outdated; gather web evidence."
    else:
        action = "build_context"
        thought = "Evidence looks sufficient; build final context."

    trace = state.get("react_trace", [])
    trace_entry = f"step={step} action={action} thought={thought}"

    return {
        "react_step": step,
        "next_action": action,
        "react_trace": trace + [trace_entry],
    }


def route_react_action(state: GraphState) -> Literal["retrieve", "web_search", "build_context"]:
    action = str(state.get("next_action", "build_context"))
    if action == "retrieve":
        return "retrieve"
    if action == "web_search":
        return "web_search"
    return "build_context"


def retrieve(state: GraphState) -> dict[str, Any]:
    question = state["question"]
    vector_store = get_vector_store()
    max_qdrant_candidates = int_env("MAX_QDRANT_CANDIDATES", 40)
    max_context_chunks = int_env("MAX_CONTEXT_CHUNKS", 10)
    max_chunks_per_paper = int_env("MAX_CHUNKS_PER_PAPER", 3)
    max_unique_papers = int_env("MAX_UNIQUE_PAPERS", 4)
    relevance_threshold = float_env("RELEVANCE_THRESHOLD", 0.65)
    force_web_fallback = bool_env("FORCE_WEB_FALLBACK", False) or query_requires_web(question)

    matches = vector_store.similarity_search_with_relevance_scores(question, k=max_qdrant_candidates)
    if not matches:
        return {
            "documents": [],
            "fallback": True,
            "top_score": 0.0,
            "react_trace": append_trace(state, "retrieve: no vector matches; fallback requested"),
        }

    documents: list[dict[str, Any]] = []
    paper_chunk_counts: dict[str, int] = {}
    top_score = 0.0

    for doc, score in matches:
        top_score = max(top_score, float(score))
        metadata = unwrap_metadata(doc.metadata)
        paper_key = str(
            metadata.get("paper_id")
            or metadata.get("source_url")
            or metadata.get("source")
            or metadata.get("title")
            or "unknown"
        )

        current_chunks = paper_chunk_counts.get(paper_key, 0)
        if current_chunks >= max_chunks_per_paper:
            continue
        if current_chunks == 0 and len(paper_chunk_counts) >= max_unique_papers:
            continue

        documents.append(
            {
                "title": str(metadata.get("title", "ArXiv document")),
                "content": doc.page_content,
                "source": str(metadata.get("source_url", metadata.get("source", ""))),
                "paper_id": str(metadata.get("paper_id", "")),
                "section": str(metadata.get("section", "")),
                "score": float(score),
                "origin": "qdrant",
            }
        )
        paper_chunk_counts[paper_key] = current_chunks + 1

        if len(documents) >= max_context_chunks:
            break

    return {
        "documents": documents,
        "fallback": force_web_fallback or (top_score < relevance_threshold) or (not documents),
        "top_score": top_score,
        "evidence_ok": False,
        "web_attempts": 0,
        "react_trace": append_trace(
            state,
            (
                f"retrieve: kept={len(documents)} top_score={top_score:.3f} "
                f"threshold={relevance_threshold:.3f}"
            ),
        ),
    }


def web_search(state: GraphState) -> dict[str, Any]:
    question = state["question"]
    existing_documents = state.get("documents", [])
    max_web_results = int_env("MAX_WEB_RESULTS", 5)
    web_relevance_threshold = float_env("WEB_RELEVANCE_THRESHOLD", 0.65)
    search_tool = get_search_tool()
    web_query = build_web_query(question)
    results = search_tool.invoke({"query": web_query})

    web_documents: list[dict[str, Any]] = []
    for item in results[:max_web_results]:
        web_documents.append(
            {
                "title": str(item.get("title", "Web result")),
                "content": str(item.get("content", "")),
                "source": str(item.get("url", "")),
                "origin": "web",
            }
        )

    scored_web_documents = score_web_documents(question, web_documents)
    top_web_score = max((safe_float(doc.get("score", 0.0)) for doc in scored_web_documents), default=0.0)

    # Preserve local evidence and append web evidence on fallback.
    return {
        "documents": existing_documents + scored_web_documents,
        "fallback": False,
        "web_attempts": int(state.get("web_attempts", 0)) + 1,
        "react_trace": append_trace(
            state,
            (
                f"web_search: query='{web_query}' added={len(scored_web_documents)} "
                f"top_web_score={top_web_score:.3f} threshold={web_relevance_threshold:.3f} "
                f"total={len(existing_documents) + len(scored_web_documents)}"
            ),
        ),
    }


def validate_evidence(state: GraphState) -> dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])
    web_only = prefers_web_only(question)
    requires_web = bool_env("FORCE_WEB_FALLBACK", False) or query_requires_web(question) or web_only

    min_local_docs = int_env("MIN_LOCAL_DOCS", 2)
    min_web_docs = int_env("MIN_WEB_DOCS", 2)
    min_web_content_chars = int_env("MIN_WEB_CONTENT_CHARS", 200)
    min_year_aligned_web_docs = int_env("MIN_YEAR_ALIGNED_WEB_DOCS", 1)
    web_relevance_threshold = float_env("WEB_RELEVANCE_THRESHOLD", 0.45)
    max_web_attempts = int_env("MAX_WEB_ATTEMPTS", 2)
    web_attempts = int(state.get("web_attempts", 0))

    local_docs = [doc for doc in documents if str(doc.get("origin", "")) == "qdrant"]
    web_docs = [doc for doc in documents if str(doc.get("origin", "")) == "web"]

    local_ok = len(local_docs) >= min_local_docs
    web_rich_docs = [
        doc
        for doc in web_docs
        if len(str(doc.get("content", "")).strip()) >= min_web_content_chars
        and str(doc.get("source", "")).strip()
        and safe_float(doc.get("score", 0.0)) >= web_relevance_threshold
    ]
    top_web_score = max((safe_float(doc.get("score", 0.0)) for doc in web_docs), default=0.0)
    unique_web_sources = {
        str(doc.get("source", "")).strip().lower()
        for doc in web_rich_docs
        if str(doc.get("source", "")).strip()
    }

    target_year = target_year_for_question(question)
    if target_year:
        year_aligned_web_docs = [doc for doc in web_rich_docs if document_mentions_year(doc, target_year)]
    else:
        year_aligned_web_docs = web_rich_docs

    year_alignment_ok = (target_year is None) or (len(year_aligned_web_docs) >= min_year_aligned_web_docs)
    web_ok = len(web_rich_docs) >= min_web_docs and len(unique_web_sources) >= 1 and year_alignment_ok

    comparison_entities = extract_comparison_entities(question)
    comparison_required = comparison_entities is not None
    entity_a = ""
    entity_b = ""
    has_entity_a = False
    has_entity_b = False
    if comparison_required:
        entity_a, entity_b = comparison_entities or ("", "")
        alias_map = build_question_alias_map(question)
        has_entity_a, has_entity_b = comparison_entity_coverage(
            documents,
            entity_a,
            entity_b,
            alias_map,
        )
    comparison_ok = (not comparison_required) or (has_entity_a and has_entity_b)

    if requires_web:
        evidence_ok = web_ok and comparison_ok
    else:
        evidence_ok = (local_ok or web_ok) and comparison_ok

    needs_more_web = (not evidence_ok) and (web_attempts < max_web_attempts)

    return {
        "fallback": needs_more_web,
        "evidence_ok": evidence_ok,
        "react_trace": append_trace(
            state,
            (
                "validate_evidence: "
                f"local_ok={local_ok} web_ok={web_ok} evidence_ok={evidence_ok} "
                f"top_web_score={top_web_score:.3f} web_threshold={web_relevance_threshold:.3f} "
                f"target_year={target_year or '-'} year_aligned_web_docs={len(year_aligned_web_docs)} "
                f"needs_more_web={needs_more_web} web_only={web_only} comparison_required={comparison_required} "
                f"entity_a={entity_a or '-'} entity_b={entity_b or '-'} "
                f"has_entity_a={has_entity_a} has_entity_b={has_entity_b}"
            ),
        ),
    }


def route_after_validation(state: GraphState) -> Literal["react_plan", "build_context"]:
    if state.get("fallback", False):
        return "react_plan"
    if state.get("evidence_ok", False):
        return "build_context"
    return "build_context"


def build_context(state: GraphState) -> dict[str, Any]:
    documents = state.get("documents", [])
    if not documents:
        return {
            "documents": [],
            "react_trace": append_trace(state, "build_context: no documents available"),
        }

    max_final_context_docs = int_env("MAX_FINAL_CONTEXT_DOCS", 8)
    web_only = prefers_web_only(state.get("question", ""))
    prefer_web_first = bool_env("FORCE_WEB_FALLBACK", False) or query_requires_web(state.get("question", ""))

    seen_keys: set[str] = set()
    unique_documents: list[dict[str, Any]] = []

    for document in documents:
        paper_id = str(document.get("paper_id", "")).strip().lower()
        source = str(document.get("source", "")).strip().lower()
        title = str(document.get("title", "")).strip().lower()
        content_prefix = str(document.get("content", "")).strip().lower()[:120]
        key = paper_id or source or title or content_prefix

        if not key or key in seen_keys:
            continue

        seen_keys.add(key)
        unique_documents.append(document)

    if web_only:
        unique_documents = [
            doc for doc in unique_documents if str(doc.get("origin", "")) == "web"
        ]

    # Prioritize local vector-store evidence, then web results, and prefer higher scores.
    unique_documents.sort(
        key=lambda doc: (
            (
                0
                if str(doc.get("origin", "")) == ("web" if prefer_web_first else "qdrant")
                else 1
            ),
            -safe_float(doc.get("score", 0.0)),
        )
    )

    final_documents = unique_documents[:max_final_context_docs]
    return {
        "documents": final_documents,
        "react_trace": append_trace(
            state,
            f"build_context: unique={len(unique_documents)} final={len(final_documents)}",
        ),
    }


def generate(state: GraphState) -> dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])
    max_prompt_docs = int_env("MAX_PROMPT_DOCS", 8)
    prompt_documents = documents[:max_prompt_docs]

    web_only = prefers_web_only(question)
    if web_only:
        web_documents = [
            doc for doc in prompt_documents if str(doc.get("origin", "")) == "web"
        ]
        if web_documents:
            prompt_documents = web_documents[:max_prompt_docs]
        else:
            return {
                "generation": (
                    "I could not find suitable online-only sources for this request yet. "
                    "Try rephrasing with specific keywords or retry to fetch fresh web results."
                ),
                "documents": [],
                "react_trace": append_trace(
                    state,
                    "generate_guard: web_only requested but no web documents available",
                ),
            }

    target_year = target_year_for_question(question)
    if target_year:
        year_filtered_documents = [
            doc
            for doc in prompt_documents
            if str(doc.get("origin", "")) != "web" or document_mentions_year(doc, target_year)
        ]
        if year_filtered_documents:
            prompt_documents = year_filtered_documents[:max_prompt_docs]

    comparison_partial_warning = ""
    comparison_trace_note = ""
    comparison_entities = extract_comparison_entities(question)
    if comparison_entities:
        entity_a, entity_b = comparison_entities
        alias_map = build_question_alias_map(question)
        has_entity_a, has_entity_b = comparison_entity_coverage(
            prompt_documents,
            entity_a,
            entity_b,
            alias_map,
        )

        if not (has_entity_a and has_entity_b):
            missing_topics: list[str] = []
            if not has_entity_a:
                missing_topics.append(entity_a)
            if not has_entity_b:
                missing_topics.append(entity_b)

            aliases_a = entity_aliases(entity_a, alias_map)
            aliases_b = entity_aliases(entity_b, alias_map)

            topical_docs = [
                doc
                for doc in prompt_documents
                if contains_any_phrase(
                    f"{doc.get('title', '')}\n{doc.get('content', '')}",
                    aliases_a + aliases_b,
                )
            ]
            if topical_docs:
                prompt_documents = topical_docs[:max_prompt_docs]

            comparison_partial_warning = (
                "Coverage warning: this comparison lacks explicit evidence for "
                f"{', '.join(missing_topics)}. "
                "Provide a partial comparison using only supported evidence. "
                "Start by stating the missing side clearly and avoid unsupported claims."
            )
            comparison_trace_note = (
                "generate_guard: partial comparison mode due to missing dual-topic evidence "
                f"entity_a={entity_a} entity_b={entity_b} "
                f"has_entity_a={has_entity_a} has_entity_b={has_entity_b}"
            )

    context = "\n\n".join(
        [
            (
                f"[{idx + 1}] {doc.get('title', 'Source')}\n"
                f"Origin: {doc.get('origin', 'unknown')}\n"
                f"URL: {doc.get('source', '')}\n"
                f"Section: {doc.get('section', 'unknown')}\n"
                f"{doc.get('content', '')}"
            )
            for idx, doc in enumerate(prompt_documents)
        ]
    )

    if not context.strip():
        context = "No relevant context found."

    llm = get_llm()
    response = llm.invoke(
        [
            (
                "system",
                (
                    "You are a research assistant for machine learning papers. "
                    "Answer only from provided context. Cite claims using [n] where n is the source index. "
                    "Prefer synthesizing across multiple sources when available and avoid duplicate citations. "
                    "If evidence is thin or outdated, explicitly say what is uncertain. "
                    "Do not invent model names, versions, dates, or announcements not explicitly present in context."
                ),
            ),
            (
                "human",
                (
                    f"Question: {question}\n\nContext:\n{context}\n\n"
                    f"{comparison_partial_warning}\n\n"
                    "Return a concise, factual answer with inline citations like [1], [2]. "
                    "When possible, compare findings across at least two different sources."
                ),
            ),
        ]
    )

    answer_text = response.content if isinstance(response.content, str) else str(response.content)

    def collect_citation_indices(text: str) -> list[int]:
        indices: list[int] = []
        for match in re.findall(r"\[(\d+)\]", text):
            try:
                index = int(match) - 1
            except ValueError:
                continue
            if 0 <= index < len(prompt_documents) and index not in indices:
                indices.append(index)
        return indices

    citation_indices = collect_citation_indices(answer_text)

    numbered_item_count = len(re.findall(r"(?m)^\s*\d+\.\s+", answer_text))
    needs_grounding_retry = (
        len(prompt_documents) >= 3 and len(citation_indices) <= 1 and numbered_item_count >= 3
    )

    if needs_grounding_retry:
        retry_response = llm.invoke(
            [
                (
                    "system",
                    (
                        "You are correcting a draft that over-relies on one citation. "
                        "Rewrite using only claims explicitly supported by the provided context. "
                        "If only one source truly supports the answer, keep it short and say evidence is limited. "
                        "Do not output a long numbered list of independent events unless each event is clearly stated in context. "
                        "Cite every factual claim with [n]."
                    ),
                ),
                (
                    "human",
                    (
                        f"Question: {question}\n\nContext:\n{context}\n\n"
                        f"Draft to correct:\n{answer_text}\n\n"
                        "Return a concise corrected answer with inline citations."
                    ),
                ),
            ]
        )
        answer_text = (
            retry_response.content
            if isinstance(retry_response.content, str)
            else str(retry_response.content)
        )
        citation_indices = collect_citation_indices(answer_text)

    if citation_indices:
        old_to_new: dict[int, int] = {
            (old_index + 1): (new_position + 1)
            for new_position, old_index in enumerate(citation_indices)
        }
        max_source_number = len(prompt_documents)

        def remap_citation(match: re.Match[str]) -> str:
            try:
                old_number = int(match.group(1))
            except ValueError:
                return match.group(0)
            if old_number in old_to_new:
                return f"[{old_to_new[old_number]}]"
            if old_number < 1 or old_number > max_source_number:
                return ""
            return ""

        answer_text = re.sub(r"\[(\d+)\]", remap_citation, answer_text)
        answer_text = re.sub(r"\s+([,.;:!?])", r"\1", answer_text)
        answer_text = re.sub(r" {2,}", " ", answer_text).strip()
        cited_documents = [prompt_documents[idx] for idx in citation_indices]
    else:
        answer_text = re.sub(r"\[(\d+)\]", "", answer_text)
        answer_text = re.sub(r"\s+([,.;:!?])", r"\1", answer_text)
        answer_text = re.sub(r" {2,}", " ", answer_text).strip()
        cited_documents = prompt_documents[: min(3, len(prompt_documents))]

    trace_state: GraphState | dict[str, Any] = state
    if comparison_trace_note:
        trace_state = dict(state)
        trace_state["react_trace"] = append_trace(state, comparison_trace_note)

    generation_trace = append_trace(
        trace_state,
        (
            f"generate: prompt_docs={len(prompt_documents)} cited_docs={len(cited_documents)} "
            f"partial_comparison={bool(comparison_partial_warning)}"
        ),
    )

    return {
        "generation": answer_text,
        "documents": cited_documents,
        "react_trace": generation_trace,
    }


workflow = StateGraph(GraphState)
workflow.add_node("react_plan", react_plan)
workflow.add_node("retrieve", retrieve)
workflow.add_node("web_search", web_search)
workflow.add_node("validate_evidence", validate_evidence)
workflow.add_node("build_context", build_context)
workflow.add_node("generate", generate)
workflow.set_entry_point("react_plan")
workflow.add_conditional_edges("react_plan", route_react_action)
workflow.add_edge("retrieve", "validate_evidence")
workflow.add_edge("web_search", "validate_evidence")
workflow.add_conditional_edges("validate_evidence", route_after_validation)
workflow.add_edge("build_context", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
