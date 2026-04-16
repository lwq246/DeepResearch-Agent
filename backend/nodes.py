import json
import re
from datetime import date, timedelta
from math import sqrt
from typing import Any, Literal, Sequence

try:
    from .configuration import bool_env
    from .configuration import float_env
    from .configuration import get_embeddings
    from .configuration import get_llm
    from .configuration import get_planner_llm
    from .configuration import get_query_rewrite_llm
    from .configuration import get_reflection_llm
    from .configuration import get_search_tool
    from .configuration import get_vector_store
    from .configuration import int_env
    from .prompts import ANSWER_SYSTEM_PROMPT
    from .prompts import PLANNER_SYSTEM_PROMPT
    from .prompts import QUERY_REWRITE_SYSTEM_PROMPT
    from .prompts import REFLECTION_SYSTEM_PROMPT
    from .state import GraphState
    from .graph_utils import append_trace
    from .graph_utils import safe_float
    from .graph_utils import unwrap_metadata
except ImportError:
    from configuration import bool_env
    from configuration import float_env
    from configuration import get_embeddings
    from configuration import get_llm
    from configuration import get_planner_llm
    from configuration import get_query_rewrite_llm
    from configuration import get_reflection_llm
    from configuration import get_search_tool
    from configuration import get_vector_store
    from configuration import int_env
    from prompts import ANSWER_SYSTEM_PROMPT
    from prompts import PLANNER_SYSTEM_PROMPT
    from prompts import QUERY_REWRITE_SYSTEM_PROMPT
    from prompts import REFLECTION_SYSTEM_PROMPT
    from state import GraphState
    from graph_utils import append_trace
    from graph_utils import safe_float
    from graph_utils import unwrap_metadata


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


def summarize_documents_for_prompt(documents: list[dict[str, Any]], max_items: int = 8) -> str:
    if not documents:
        return "No documents."

    lines: list[str] = []
    for idx, doc in enumerate(documents[:max_items], start=1):
        title = str(doc.get("title", ""))[:120]
        origin = str(doc.get("origin", ""))
        source = str(doc.get("source", ""))[:120]
        published = str(doc.get("published", doc.get("updated", "")))[:40]
        score = safe_float(doc.get("score", 0.0))
        snippet = str(doc.get("content", "")).replace("\n", " ")[:180]
        lines.append(
            f"{idx}. origin={origin} score={score:.3f} title={title} "
            f"published={published or '-'} source={source} snippet={snippet}"
        )

    return "\n".join(lines)


def current_date_iso() -> str:
    return date.today().isoformat()


def relative_month_target(question: str) -> tuple[int, int] | None:
    q = question.lower()
    today = date.today()
    if "last month" in q:
        last_day_previous_month = date(today.year, today.month, 1) - timedelta(days=1)
        return last_day_previous_month.year, last_day_previous_month.month
    if "this month" in q:
        return today.year, today.month
    return None


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


def llm_plan_action(
    question: str,
    documents: list[dict[str, Any]],
    fallback_action: str,
    fallback_thought: str,
    fallback_requires_web: bool,
    web_attempts: int,
    max_web_attempts: int,
) -> tuple[str, str, bool]:
    if not bool_env("LLM_PLANNER_ENABLED", True):
        return fallback_action, fallback_thought, fallback_requires_web

    parsed = llm_json_response(
        llm=get_planner_llm(),
        system_prompt=PLANNER_SYSTEM_PROMPT,
        human_prompt=(
            f"Question: {question}\n"
            f"Web attempts: {web_attempts}/{max_web_attempts}\n"
            f"Fallback action: {fallback_action}\n"
            f"Fallback thought: {fallback_thought}\n"
            "Available evidence:\n"
            f"{summarize_documents_for_prompt(documents)}\n"
        ),
    )
    if not parsed:
        return fallback_action, fallback_thought, fallback_requires_web

    action = str(parsed.get("action", "")).strip().lower()
    thought = str(parsed.get("thought", "")).strip() or fallback_thought
    requires_web = parsed.get("requires_web", fallback_requires_web)
    if isinstance(requires_web, str):
        requires_web_value = requires_web.strip().lower() in {"1", "true", "yes", "y"}
    elif isinstance(requires_web, bool):
        requires_web_value = requires_web
    else:
        requires_web_value = fallback_requires_web

    if action not in {"retrieve", "web_search", "build_context"}:
        return fallback_action, fallback_thought, requires_web_value
    return action, thought, requires_web_value


def llm_rewrite_web_query(question: str, default_query: str) -> str:
    if not bool_env("LLM_QUERY_REWRITE_ENABLED", True):
        return default_query

    temporal_target = relative_month_target(question)
    if temporal_target:
        target_hint = temporal_window_label(*temporal_target)
    else:
        target_hint = "none"

    parsed = llm_json_response(
        llm=get_query_rewrite_llm(),
        system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
        human_prompt=(
            f"Current date: {current_date_iso()}\n"
            f"Resolved target window: {target_hint}\n"
            f"User question: {question}\n"
            f"Default rewritten query: {default_query}\n"
            "Prefer concise wording and include freshness hints only when useful."
        ),
    )
    if not parsed:
        return default_query

    query = str(parsed.get("query", "")).strip()
    return query or default_query


def llm_reflect_evidence(
    question: str,
    documents: list[dict[str, Any]],
    local_ok: bool,
    web_ok: bool,
    web_attempts: int,
    max_web_attempts: int,
    default_requires_web: bool,
    default_evidence_ok: bool,
    default_needs_more_web: bool,
) -> tuple[bool, bool, bool, list[str], str]:
    if not bool_env("LLM_REFLECTION_ENABLED", True):
        return default_evidence_ok, default_needs_more_web, default_requires_web, [], ""

    temporal_target = relative_month_target(question)
    if temporal_target:
        target_hint = temporal_window_label(*temporal_target)
    else:
        target_hint = "none"

    parsed = llm_json_response(
        llm=get_reflection_llm(),
        system_prompt=REFLECTION_SYSTEM_PROMPT,
        human_prompt=(
            f"Current date: {current_date_iso()}\n"
            f"Resolved target window: {target_hint}\n"
            f"Question: {question}\n"
            f"Signals: local_ok={local_ok}, web_ok={web_ok}, web_attempts={web_attempts}/{max_web_attempts}\n"
            f"Heuristic baseline: evidence_ok={default_evidence_ok}, needs_more_web={default_needs_more_web}\n"
            "Evidence:\n"
            f"{summarize_documents_for_prompt(documents)}\n"
            "Constraint: do not request more web retrieval if max attempts are reached."
        ),
    )
    if not parsed:
        return default_evidence_ok, default_needs_more_web, default_requires_web, [], ""

    evidence_ok = bool(parsed.get("evidence_ok", default_evidence_ok))
    requested_more_web = bool(parsed.get("needs_more_web", default_needs_more_web))
    requires_web = parsed.get("requires_web", default_requires_web)
    if isinstance(requires_web, str):
        requires_web_value = requires_web.strip().lower() in {"1", "true", "yes", "y"}
    elif isinstance(requires_web, bool):
        requires_web_value = requires_web
    else:
        requires_web_value = default_requires_web

    reason = str(parsed.get("reason", "")).strip()
    missing_topics_raw = parsed.get("missing_topics", [])

    missing_topics: list[str] = []
    if isinstance(missing_topics_raw, list):
        for item in missing_topics_raw:
            topic = str(item).strip()
            if topic:
                missing_topics.append(topic)

    if web_attempts >= max_web_attempts:
        requested_more_web = False

    needs_more_web = (not evidence_ok) and requested_more_web
    return evidence_ok, needs_more_web, requires_web_value, missing_topics[:4], reason


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


def react_plan(state: GraphState) -> dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])
    step = int(state.get("react_step", 0)) + 1
    max_steps = int_env("REACT_MAX_STEPS", 4)

    has_local_docs = any(str(doc.get("origin", "")) == "qdrant" for doc in documents)
    has_web_docs = any(str(doc.get("origin", "")) == "web" for doc in documents)
    force_web_fallback = bool_env("FORCE_WEB_FALLBACK", False)
    requires_web = (
        bool(state.get("requires_web", False))
        or force_web_fallback
        or (relative_month_target(question) is not None)
    )

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
    elif requires_web and not has_web_docs:
        action = "web_search"
        thought = "Question requires web evidence; run web retrieval before local search."
    elif not has_local_docs:
        action = "retrieve"
        thought = "Need local evidence first; run vector retrieval."
    elif force_web_fallback and not has_web_docs:
        action = "web_search"
        thought = "Web fallback is forced; gather web evidence."
    else:
        action = "build_context"
        thought = "Evidence looks sufficient; build final context."

    action, thought, requires_web = llm_plan_action(
        question=question,
        documents=documents,
        fallback_action=action,
        fallback_thought=thought,
        fallback_requires_web=requires_web,
        web_attempts=int(state.get("web_attempts", 0)),
        max_web_attempts=int_env("MAX_WEB_ATTEMPTS", 2),
    )

    trace = state.get("react_trace", [])
    trace_entry = f"step={step} action={action} thought={thought}"

    return {
        "react_step": step,
        "next_action": action,
        "requires_web": requires_web,
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
    force_web_fallback = bool_env("FORCE_WEB_FALLBACK", False)

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
                "published": str(metadata.get("published", "")),
                "updated": str(metadata.get("updated", "")),
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
    default_web_query = question.strip()
    web_query = llm_rewrite_web_query(question, default_web_query)
    results = search_tool.invoke({"query": web_query})

    web_documents: list[dict[str, Any]] = []
    for item in results[:max_web_results]:
        web_documents.append(
            {
                "title": str(item.get("title", "Web result")),
                "content": str(item.get("content", "")),
                "source": str(item.get("url", "")),
                "published": str(item.get("published_date", item.get("published", item.get("date", "")))),
                "origin": "web",
            }
        )

    scored_web_documents = score_web_documents(question, web_documents)
    top_web_score = max((safe_float(doc.get("score", 0.0)) for doc in scored_web_documents), default=0.0)

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
    temporal_target = relative_month_target(question)
    requires_web = (
        bool_env("FORCE_WEB_FALLBACK", False)
        or bool(state.get("requires_web", False))
        or (temporal_target is not None)
    )

    min_local_docs = int_env("MIN_LOCAL_DOCS", 2)
    min_web_docs = int_env("MIN_WEB_DOCS", 2)
    min_web_content_chars = int_env("MIN_WEB_CONTENT_CHARS", 200)
    web_relevance_threshold = float_env("WEB_RELEVANCE_THRESHOLD", 0.65)
    max_web_attempts = int_env("MAX_WEB_ATTEMPTS", 2)
    web_attempts = int(state.get("web_attempts", 0))

    local_docs = [doc for doc in documents if str(doc.get("origin", "")) == "qdrant"]
    web_docs = [doc for doc in documents if str(doc.get("origin", "")) == "web"]

    local_ok = len(local_docs) >= min_local_docs
    web_rich_docs = [
        doc
        for doc in web_docs
        if len(str(doc.get("content", "")).strip()) >= min_web_content_chars
        and safe_float(doc.get("score", 0.0)) >= web_relevance_threshold
    ]
    top_web_score = max((safe_float(doc.get("score", 0.0)) for doc in web_docs), default=0.0)

    temporal_matched_web_docs = web_rich_docs
    temporal_label = "-"
    if temporal_target:
        temporal_label = temporal_window_label(*temporal_target)
        temporal_matched_web_docs = [
            doc for doc in web_rich_docs if document_mentions_target_month(doc, *temporal_target)
        ]

    min_temporal_matched_web_docs = int_env("MIN_TEMPORAL_MATCHED_WEB_DOCS", 1)
    temporal_ok = (temporal_target is None) or (
        len(temporal_matched_web_docs) >= min_temporal_matched_web_docs
    )
    web_ok = len(web_rich_docs) >= min_web_docs and temporal_ok

    if requires_web:
        evidence_ok = web_ok
    else:
        evidence_ok = local_ok or web_ok

    needs_more_web = (not evidence_ok) and (web_attempts < max_web_attempts)

    evidence_ok, needs_more_web, requires_web, missing_topics, reflection_reason = llm_reflect_evidence(
        question=question,
        documents=documents,
        local_ok=local_ok,
        web_ok=web_ok,
        web_attempts=web_attempts,
        max_web_attempts=max_web_attempts,
        default_requires_web=requires_web,
        default_evidence_ok=evidence_ok,
        default_needs_more_web=needs_more_web,
    )

    return {
        "fallback": needs_more_web,
        "evidence_ok": evidence_ok,
        "requires_web": requires_web,
        "react_trace": append_trace(
            state,
            (
                "validate_evidence: "
                f"local_ok={local_ok} web_ok={web_ok} evidence_ok={evidence_ok} "
                f"top_web_score={top_web_score:.3f} web_threshold={web_relevance_threshold:.3f} "
                f"needs_more_web={needs_more_web} requires_web={requires_web} "
                f"temporal_target={temporal_label} temporal_matches={len(temporal_matched_web_docs)} "
                f"missing_topics={';'.join(missing_topics) if missing_topics else '-'} "
                f"reflection_reason={reflection_reason or '-'}"
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
    requires_web = (
        bool_env("FORCE_WEB_FALLBACK", False)
        or bool(state.get("requires_web", False))
        or (relative_month_target(state.get("question", "")) is not None)
    )
    prefer_web_first = requires_web

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

    if requires_web:
        unique_documents = [
            doc for doc in unique_documents if str(doc.get("origin", "")) == "web"
        ]

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

    temporal_target = relative_month_target(question)
    requires_web = (
        bool_env("FORCE_WEB_FALLBACK", False)
        or bool(state.get("requires_web", False))
        or (temporal_target is not None)
    )
    if requires_web:
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
                    "generate_guard: requires_web requested but no web documents available",
                ),
            }

    temporal_hint = "none"
    if temporal_target:
        temporal_hint = temporal_window_label(*temporal_target)
        temporal_documents = [
            doc for doc in prompt_documents if document_mentions_target_month(doc, *temporal_target)
        ]
        if temporal_documents:
            prompt_documents = temporal_documents[:max_prompt_docs]
        elif requires_web:
            return {
                "generation": (
                    f"I could not find enough evidence specifically for {temporal_hint}. "
                    "Please retry with additional keywords or provide sources for that exact time window."
                ),
                "documents": [],
                "react_trace": append_trace(
                    state,
                    f"generate_guard: no temporal evidence matched target={temporal_hint}",
                ),
            }

    context = "\n\n".join(
        [
            (
                f"[{idx + 1}] {doc.get('title', 'Source')}\n"
                f"Origin: {doc.get('origin', 'unknown')}\n"
                f"URL: {doc.get('source', '')}\n"
                f"Published: {doc.get('published', doc.get('updated', ''))}\n"
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
                ANSWER_SYSTEM_PROMPT,
            ),
            (
                "human",
                (
                    f"Current date: {current_date_iso()}\n"
                    f"Resolved target window: {temporal_hint}\n"
                    f"Question: {question}\n\nContext:\n{context}"
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

    generation_trace = append_trace(
        state,
        f"generate: prompt_docs={len(prompt_documents)} cited_docs={len(cited_documents)}",
    )

    return {
        "generation": answer_text,
        "documents": cited_documents,
        "react_trace": generation_trace,
    }
