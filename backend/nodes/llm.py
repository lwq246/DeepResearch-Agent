from typing import Any


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
from .shared import summarize_documents_for_prompt


def llm_extract_author_constraint(question: str) -> str | None:
    parsed = llm_json_response(
        llm=get_planner_llm(),
        system_prompt=AUTHOR_QUERY_EXTRACTION_SYSTEM_PROMPT,
        human_prompt=f"Question: {question}",
    )

    if not parsed:
        return None

    is_author_query = coerce_bool(parsed.get("is_author_query", False), default=False)
    candidate = str(parsed.get("author", "")).strip()
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

    if action not in {"retrieve", "web_search", "generate"}:
        return fallback_action, fallback_thought, requires_web_value
    return action, thought, requires_web_value


def llm_rewrite_web_query(
    question: str,
    default_query: str,
    attempt_index: int = 0,
    previous_query: str = "",
) -> str:
    retry_hint = ""
    if attempt_index > 0:
        prior = previous_query.strip() or default_query
        retry_hint = (
            f"Retry attempt: {attempt_index + 1}. "
            f"Previous query was: {prior}\n"
            "Generate a meaningfully different query from the previous one while preserving intent. "
            "Use complementary keywords, aliases, or source-focused phrasing."
        )

    parsed = llm_json_response(
        llm=get_query_rewrite_llm(),
        system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
        human_prompt=(
            f"Current date: {current_date_iso()}\n"
            f"User question: {question}\n"
            f"Default rewritten query: {default_query}\n"
            f"{retry_hint}\n"
            "Prefer concise wording and include freshness hints only when useful."
        ),
    )
    if not parsed:
        if attempt_index <= 0:
            return default_query
        return f"{default_query} official sources"

    query = str(parsed.get("query", "")).strip()
    candidate = query or default_query

    if attempt_index > 0:
        prior_norm = previous_query.strip().casefold()
        if prior_norm and candidate.strip().casefold() == prior_norm:
            return f"{default_query} official sources"

    return candidate


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
    parsed = llm_json_response(
        llm=get_reflection_llm(),
        system_prompt=REFLECTION_SYSTEM_PROMPT,
        human_prompt=(
            f"Current date: {current_date_iso()}\n"
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
