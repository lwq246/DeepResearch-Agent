from typing import Any

try:
    from ..configuration import bool_env
    from ..configuration import get_planner_llm
    from ..configuration import get_query_rewrite_llm
    from ..configuration import get_reflection_llm
    from ..prompts import AUTHOR_QUERY_EXTRACTION_SYSTEM_PROMPT
    from ..prompts import PLANNER_SYSTEM_PROMPT
    from ..prompts import QUERY_REWRITE_SYSTEM_PROMPT
    from ..prompts import REFLECTION_SYSTEM_PROMPT
    from .shared import coerce_bool
    from .shared import current_date_iso
    from .shared import llm_json_response
    from .shared import normalize_author_name
    from .shared import relative_month_target
    from .shared import summarize_documents_for_prompt
    from .shared import temporal_window_label
except ImportError:
    from configuration import bool_env
    from configuration import get_planner_llm
    from configuration import get_query_rewrite_llm
    from configuration import get_reflection_llm
    from prompts import AUTHOR_QUERY_EXTRACTION_SYSTEM_PROMPT
    from prompts import PLANNER_SYSTEM_PROMPT
    from prompts import QUERY_REWRITE_SYSTEM_PROMPT
    from prompts import REFLECTION_SYSTEM_PROMPT
    from nodes.shared import coerce_bool
    from nodes.shared import current_date_iso
    from nodes.shared import llm_json_response
    from nodes.shared import normalize_author_name
    from nodes.shared import relative_month_target
    from nodes.shared import summarize_documents_for_prompt
    from nodes.shared import temporal_window_label

def llm_extract_author_constraint(question: str) -> str | None:
    if not bool_env("LLM_AUTHOR_QUERY_ENABLED", True):
        return None

    parsed = llm_json_response(
        llm=get_planner_llm(),
        system_prompt=AUTHOR_QUERY_EXTRACTION_SYSTEM_PROMPT,
        human_prompt=f"Question: {question}",
    )

    if not parsed:
        return None

    is_author_query = coerce_bool(parsed.get("is_author_query", False), default=False)
    candidate = normalize_author_name(str(parsed.get("author", "")))
    if is_author_query and len(candidate) >= 3:
        return candidate

    return None


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
    requires_web_value = coerce_bool(parsed.get("requires_web", fallback_requires_web), default=fallback_requires_web)

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

    evidence_ok = coerce_bool(parsed.get("evidence_ok", default_evidence_ok), default=default_evidence_ok)
    requested_more_web = coerce_bool(parsed.get("needs_more_web", default_needs_more_web), default=default_needs_more_web)
    requires_web_value = coerce_bool(parsed.get("requires_web", default_requires_web), default=default_requires_web)

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
