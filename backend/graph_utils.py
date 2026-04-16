from typing import Any

try:
    from .state import GraphState
except ImportError:
    from state import GraphState


def append_trace(state: GraphState, message: str) -> list[str]:
    trace = list(state.get("react_trace", []))
    trace.append(message)
    return trace


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def unwrap_metadata(raw_metadata: dict[str, Any]) -> dict[str, Any]:
    nested = raw_metadata.get("metadata")
    if isinstance(nested, dict):
        merged = dict(raw_metadata)
        merged.update(nested)
        return merged
    return raw_metadata