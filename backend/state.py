from typing import Any, TypedDict


class GraphState(TypedDict):
    question: str
    documents: list[dict[str, Any]]
    generation: str
    requires_web: bool
    fallback: bool
    top_score: float
    evidence_ok: bool
    web_attempts: int
    react_step: int
    next_action: str
    react_trace: list[str]
