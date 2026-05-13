import re
from functools import wraps
from typing import Any, Literal

import logfire

try:
    from ..configuration import bool_env
    from ..configuration import float_env
    from ..configuration import get_llm
    from ..configuration import get_search_tool
    from ..configuration import get_vector_store
    from ..configuration import int_env
    from ..graph_utils import append_trace
    from ..graph_utils import safe_float
    from ..graph_utils import unwrap_metadata
    from .author_retrieval import retrieve_documents_by_author
    from .llm import llm_extract_author_constraint
    from .llm import llm_plan_action
    from .llm import llm_reflect_evidence
    from .llm import llm_rewrite_web_query
    from .shared import build_qdrant_document
    from .shared import can_take_chunk_for_paper
    from .shared import coerce_bool
    from .shared import current_date_iso
    from .shared import document_dedup_key
    from .shared import document_mentions_target_month
    from .shared import increment_paper_chunk_count
    from .shared import metadata_paper_group_key
    from .shared import relative_month_target
    from .shared import resolve_requires_web
    from .shared import score_web_documents
    from .shared import temporal_window_label
    from ..prompts import ANSWER_SYSTEM_PROMPT
    from ..state import GraphState
except ImportError:
    from configuration import bool_env
    from configuration import float_env
    from configuration import get_llm
    from configuration import get_search_tool
    from configuration import get_vector_store
    from configuration import int_env
    from graph_utils import append_trace
    from graph_utils import safe_float
    from graph_utils import unwrap_metadata
    from nodes.author_retrieval import retrieve_documents_by_author
    from nodes.llm import llm_extract_author_constraint
    from nodes.llm import llm_plan_action
    from nodes.llm import llm_reflect_evidence
    from nodes.llm import llm_rewrite_web_query
    from nodes.shared import build_qdrant_document
    from nodes.shared import can_take_chunk_for_paper
    from nodes.shared import coerce_bool
    from nodes.shared import current_date_iso
    from nodes.shared import document_dedup_key
    from nodes.shared import document_mentions_target_month
    from nodes.shared import increment_paper_chunk_count
    from nodes.shared import metadata_paper_group_key
    from nodes.shared import relative_month_target
    from nodes.shared import resolve_requires_web
    from nodes.shared import score_web_documents
    from nodes.shared import temporal_window_label
    from prompts import ANSWER_SYSTEM_PROMPT
    from state import GraphState


def traced_node(node_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(state: GraphState) -> dict[str, Any]:
            documents = state.get("documents", [])
            with logfire.span(
                f"node.{node_name}",
                question=str(state.get("question", ""))[:200],
                react_step=int(state.get("react_step", 0)),
                web_attempts=int(state.get("web_attempts", 0)),
                documents_count=len(documents) if isinstance(documents, list) else 0,
            ):
                logfire.info(
                    "node_started",
                    node=node_name,
                    node_trace=True,
                    react_step=int(state.get("react_step", 0)),
                    web_attempts=int(state.get("web_attempts", 0)),
                )
                try:
                    result = func(state)
                except Exception as exc:
                    logfire.info(
                        "node_failed",
                        node=node_name,
                        node_trace=True,
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:240],
                    )
                    raise
                result_documents = result.get("documents", []) if isinstance(result, dict) else []
                logfire.info(
                    "node_completed",
                    node=node_name,
                    node_trace=True,
                    updated_keys=sorted(result.keys()) if isinstance(result, dict) else [],
                    fallback=bool(result.get("fallback", False)) if isinstance(result, dict) else False,
                    evidence_ok=bool(result.get("evidence_ok", False)) if isinstance(result, dict) else False,
                    result_documents_count=len(result_documents) if isinstance(result_documents, list) else 0,
                )
                return result

        return wrapper

    return decorator


def pick_react_action(
    *,
    step: int,
    max_steps: int,
    web_attempts: int,
    max_web_attempts: int,
    fallback: bool,
    requires_web: bool,
    has_web_docs: bool,
    has_local_docs: bool,
    force_web_fallback: bool,
) -> tuple[str, str]:
    if step >= max_steps or web_attempts >= max_web_attempts:
        return "build_context", "Safety limit reached; proceeding with available evidence."

    if fallback:
        if web_attempts < max_web_attempts:
            return "web_search", "Evidence quality is insufficient; run another web retrieval step."
        return "build_context", "Reached web retry limit; proceed with best available evidence."

    if requires_web and not has_web_docs:
        return "web_search", "Question requires web evidence; run web retrieval before local search."

    if not has_local_docs:
        return "retrieve", "Need local evidence first; run vector retrieval."

    if force_web_fallback and not has_web_docs:
        return "web_search", "Web fallback is forced; gather web evidence."

    return "build_context", "Evidence looks sufficient; build final context."


@traced_node("react_plan")
def react_plan(state: GraphState) -> dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])
    step = int(state.get("react_step", 0)) + 1
    max_steps = int_env("REACT_MAX_STEPS", 4)
    web_attempts = int(state.get("web_attempts", 0))
    max_web_attempts = int_env("MAX_WEB_ATTEMPTS", 2)

    has_local_docs = any(str(doc.get("origin", "")) == "qdrant" for doc in documents)
    has_web_docs = any(str(doc.get("origin", "")) == "web" for doc in documents)
    force_web_fallback = bool_env("FORCE_WEB_FALLBACK", False)
    requires_web = resolve_requires_web(state, question)

    action, thought = pick_react_action(
        step=step,
        max_steps=max_steps,
        web_attempts=web_attempts,
        max_web_attempts=max_web_attempts,
        fallback=coerce_bool(state.get("fallback", False), default=False),
        requires_web=requires_web,
        has_web_docs=has_web_docs,
        has_local_docs=has_local_docs,
        force_web_fallback=force_web_fallback,
    )

    action, thought, requires_web = llm_plan_action(
        question=question,
        documents=documents,
        fallback_action=action,
        fallback_thought=thought,
        fallback_requires_web=requires_web,
        web_attempts=web_attempts,
        max_web_attempts=max_web_attempts,
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


@traced_node("retrieve")
def retrieve(state: GraphState) -> dict[str, Any]:
    question = state["question"]
    vector_store = get_vector_store()
    max_qdrant_candidates = int_env("MAX_QDRANT_CANDIDATES", 40)
    max_author_scan_points = int_env("MAX_AUTHOR_SCAN_POINTS", 6000)
    max_author_candidates = int_env("MAX_AUTHOR_CANDIDATES", 200)
    max_context_chunks = int_env("MAX_CONTEXT_CHUNKS", 5)
    max_chunks_per_paper = int_env("MAX_CHUNKS_PER_PAPER", 3)
    max_unique_papers = int_env("MAX_UNIQUE_PAPERS", 4)
    relevance_threshold = float_env("RELEVANCE_THRESHOLD", 0.65)
    force_web_fallback = bool_env("FORCE_WEB_FALLBACK", False)

    requested_author = llm_extract_author_constraint(question)
    if requested_author:
        author_documents = retrieve_documents_by_author(
            vector_store=vector_store,
            question=question,
            requested_author=requested_author,
            max_scan_points=max_author_scan_points,
            max_author_candidates=max_author_candidates,
            max_context_chunks=max_context_chunks,
            max_chunks_per_paper=max_chunks_per_paper,
            max_unique_papers=max_unique_papers,
        )
        if not author_documents:
            return {
                "documents": [],
                "fallback": True,
                "top_score": 0.0,
                "evidence_ok": False,
                "web_attempts": 0,
                "author_constraint": requested_author,
                "react_trace": append_trace(
                    state,
                    (
                        f"retrieve: author='{requested_author}' matched=0 "
                        f"scan_limit={max_author_scan_points}; fallback requested"
                    ),
                ),
            }

        top_author_score = max((safe_float(doc.get("score", 0.0)) for doc in author_documents), default=0.0)
        return {
            "documents": author_documents,
            "fallback": force_web_fallback,
            "top_score": top_author_score,
            "evidence_ok": False,
            "web_attempts": 0,
            "author_constraint": requested_author,
            "react_trace": append_trace(
                state,
                (
                    f"retrieve: author='{requested_author}' matched={len(author_documents)} "
                    f"scan_limit={max_author_scan_points} candidate_limit={max_author_candidates} "
                    f"top_score={top_author_score:.3f}"
                ),
            ),
        }

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
        paper_key = metadata_paper_group_key(metadata)

        if not can_take_chunk_for_paper(
            paper_chunk_counts=paper_chunk_counts,
            paper_key=paper_key,
            max_chunks_per_paper=max_chunks_per_paper,
            max_unique_papers=max_unique_papers,
        ):
            continue

        documents.append(
            build_qdrant_document(
                metadata=metadata,
                content=doc.page_content,
                score=float(score),
            )
        )
        increment_paper_chunk_count(paper_chunk_counts=paper_chunk_counts, paper_key=paper_key)

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


@traced_node("web_search")
def web_search(state: GraphState) -> dict[str, Any]:
    question = state["question"]
    existing_documents = state.get("documents", [])
    max_web_results = int_env("MAX_WEB_RESULTS", 5)
    web_relevance_threshold = float_env("WEB_RELEVANCE_THRESHOLD", 0.65)
    search_tool = get_search_tool()
    default_web_query = question.strip()
    web_attempts = int(state.get("web_attempts", 0))
    previous_web_query = str(state.get("last_web_query", "")).strip()
    web_query = llm_rewrite_web_query(
        question,
        default_web_query,
        attempt_index=web_attempts,
        previous_query=previous_web_query,
    )
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

    scored_web_documents = [
        doc
        for doc in score_web_documents(question, web_documents)
        if safe_float(doc.get("score", 0.0)) >= web_relevance_threshold
    ]
    top_web_score = max((safe_float(doc.get("score", 0.0)) for doc in scored_web_documents), default=0.0)

    return {
        "documents": existing_documents + scored_web_documents,
        "fallback": False,
        "web_attempts": web_attempts + 1,
        "last_web_query": web_query,
        "react_trace": append_trace(
            state,
            (
                f"web_search: query='{web_query}' added={len(scored_web_documents)} "
                f"top_web_score={top_web_score:.3f} threshold={web_relevance_threshold:.3f} "
                f"total={len(existing_documents) + len(scored_web_documents)}"
            ),
        ),
    }


@traced_node("validate_evidence")
def validate_evidence(state: GraphState) -> dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])
    temporal_target = relative_month_target(question)
    requires_web = resolve_requires_web(state, question)
    requested_author = str(state.get("author_constraint", "")).strip()

    min_local_docs = int_env("MIN_LOCAL_DOCS", 2)
    min_web_docs = int_env("MIN_WEB_DOCS", 2)
    min_web_content_chars = int_env("MIN_WEB_CONTENT_CHARS", 200)
    web_relevance_threshold = float_env("WEB_RELEVANCE_THRESHOLD", 0.45)
    max_web_attempts = int_env("MAX_WEB_ATTEMPTS", 2)
    web_attempts = int(state.get("web_attempts", 0))

    local_docs = [doc for doc in documents if str(doc.get("origin", "")) == "qdrant"]
    web_docs = [doc for doc in documents if str(doc.get("origin", "")) == "web"]

    local_ok = len(local_docs) >= (1 if requested_author else min_local_docs)
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

    if (not evidence_ok) and (web_attempts < max_web_attempts):
        needs_more_web = True
        if reflection_reason:
            reflection_reason = f"{reflection_reason} | forced_retry_on_insufficient_evidence=true"
        else:
            reflection_reason = "Forced retry because evidence_ok=false and retries remain."

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
                f"author_target={requested_author or '-'} "
                f"missing_topics={';'.join(missing_topics) if missing_topics else '-'} "
                f"reflection_reason={reflection_reason or '-'}"
            ),
        ),
    }


def route_after_validation(state: GraphState) -> Literal["react_plan", "build_context"]:
    if state.get("fallback", False):
        return "react_plan"
    return "build_context"


@traced_node("build_context")
def build_context(state: GraphState) -> dict[str, Any]:
    documents = state.get("documents", [])
    if not documents:
        return {
            "documents": [],
            "react_trace": append_trace(state, "build_context: no documents available"),
        }

    max_final_context_docs = int_env("MAX_FINAL_CONTEXT_DOCS", 5)
    requires_web = resolve_requires_web(state, str(state.get("question", "")))
    prefer_web_first = requires_web

    seen_keys: set[str] = set()
    unique_documents: list[dict[str, Any]] = []

    for document in documents:
        key = document_dedup_key(document)

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

    if not requires_web:
        web_documents = [doc for doc in unique_documents if str(doc.get("origin", "")) == "web"]
        qdrant_documents = [doc for doc in unique_documents if str(doc.get("origin", "")) == "qdrant"]

        if web_documents and qdrant_documents:
            top_web_documents = sorted(
                web_documents,
                key=lambda doc: -safe_float(doc.get("score", 0.0)),
            )[:3]
            top_qdrant_documents = sorted(
                qdrant_documents,
                key=lambda doc: -safe_float(doc.get("score", 0.0)),
            )[:3]
            final_documents = top_web_documents + top_qdrant_documents
        else:
            final_documents = unique_documents[:max_final_context_docs]
    else:
        final_documents = unique_documents[:max_final_context_docs]

    return {
        "documents": final_documents,
        "react_trace": append_trace(
            state,
            (
                f"build_context: unique={len(unique_documents)} final={len(final_documents)} "
                f"web={len([doc for doc in final_documents if str(doc.get('origin', '')) == 'web'])} "
                f"qdrant={len([doc for doc in final_documents if str(doc.get('origin', '')) == 'qdrant'])}"
            ),
        ),
    }


@traced_node("generate")
def generate(state: GraphState) -> dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])
    max_prompt_docs = int_env("MAX_PROMPT_DOCS", 6)
    prompt_documents = documents[:max_prompt_docs]

    temporal_target = relative_month_target(question)
    requires_web = resolve_requires_web(state, question)
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
                f"Authors: {', '.join(doc.get('authors', [])) if isinstance(doc.get('authors', []), list) else doc.get('authors', '')}\n"
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

    def source_identity_key(document: dict[str, Any]) -> str:
        source = str(document.get("source", "")).strip().lower()
        title = str(document.get("title", "")).strip().lower()
        paper_id = str(document.get("paper_id", "")).strip().lower()
        return source or paper_id or title

    def dedupe_by_source(documents_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for document in documents_list:
            key = source_identity_key(document)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(document)
        return unique

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
        cited_chunks = [prompt_documents[idx] for idx in citation_indices]
        cited_documents = dedupe_by_source(cited_chunks)
        source_key_to_new_number: dict[str, int] = {
            source_identity_key(document): idx + 1 for idx, document in enumerate(cited_documents)
        }
        max_source_number = len(prompt_documents)

        def remap_citation(match: re.Match[str]) -> str:
            try:
                old_number = int(match.group(1))
            except ValueError:
                return match.group(0)
            if old_number < 1 or old_number > max_source_number:
                return ""
            document = prompt_documents[old_number - 1]
            new_number = source_key_to_new_number.get(source_identity_key(document))
            if new_number is None:
                return ""
            return f"[{new_number}]"

        answer_text = re.sub(r"\[(\d+)\]", remap_citation, answer_text)
        answer_text = re.sub(r"(\[(\d+)\])(\s*\1)+", r"\1", answer_text)
        answer_text = re.sub(r"\s+([,.;:!?])", r"\1", answer_text)
        answer_text = re.sub(r" {2,}", " ", answer_text).strip()
    else:
        answer_text = re.sub(r"\[(\d+)\]", "", answer_text)
        answer_text = re.sub(r"\s+([,.;:!?])", r"\1", answer_text)
        answer_text = re.sub(r" {2,}", " ", answer_text).strip()
        cited_documents = dedupe_by_source(prompt_documents)[: min(3, len(prompt_documents))]

    generation_trace = append_trace(
        state,
        f"generate: prompt_docs={len(prompt_documents)} cited_docs={len(cited_documents)}",
    )

    return {
        "generation": answer_text,
        "documents": cited_documents,
        "react_trace": generation_trace,
    }
