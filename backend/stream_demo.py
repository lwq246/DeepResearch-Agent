from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .configuration import get_reflection_llm
    from .graph import app as graph
except ImportError:
    from configuration import get_reflection_llm
    from graph import app as graph


DEFAULT_EVAL_SET: list[dict[str, Any]] = [
    {
        "question": "what is the difference between langchain and langsmith",
        "expected": "LangChain is an app framework, LangSmith is for tracing and evaluation.",
        "expected_keywords": ["framework", "tracing", "evaluation"],
    },
    {
        "question": "what is retrieval augmented generation",
        "expected": "RAG retrieves external context before generation.",
        "expected_keywords": ["retrieve", "context", "generation"],
    },
    {
        "question": "when should i use web search in this rag flow",
        "expected": "Use web search when vector retrieval evidence is weak or stale.",
        "expected_keywords": ["web", "evidence", "stale"],
    },
    {
        "question": "what does evidence_ok mean in this system",
        "expected": "It indicates whether retrieved evidence is strong enough to answer.",
        "expected_keywords": ["evidence", "strong", "answer"],
    },
    {
        "question": "what does fallback mean in the final state",
        "expected": "Fallback means the system had to degrade gracefully due to weak evidence.",
        "expected_keywords": ["fallback", "weak", "evidence"],
    },
    {
        "question": "what is top_score used for",
        "expected": "Top score tracks the best retrieval relevance signal.",
        "expected_keywords": ["score", "retrieval", "relevance"],
    },
    {
        "question": "why is react_trace helpful",
        "expected": "React trace records planner decisions and routing steps.",
        "expected_keywords": ["trace", "decisions", "routing"],
    },
    {
        "question": "how do retries appear in this graph",
        "expected": "Retries appear as repeated planner and web search visits.",
        "expected_keywords": ["retries", "web", "visits"],
    },
    {
        "question": "what is the purpose of build_context",
        "expected": "Build context composes selected evidence before final generation.",
        "expected_keywords": ["context", "evidence", "generation"],
    },
    {
        "question": "what does grounded answer mean in rag",
        "expected": "Grounded answers are supported by cited sources.",
        "expected_keywords": ["grounded", "sources", "supported"],
    },
]


def build_initial_state(question: str) -> dict[str, Any]:
    return {
        "question": question,
        "documents": [],
        "generation": "",
        "requires_web": False,
        "fallback": False,
        "top_score": 0.0,
        "evidence_ok": False,
        "web_attempts": 0,
        "react_step": 0,
        "next_action": "",
        "react_trace": [],
    }


def summarize_update(update: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in update.items():
        if key == "documents" and isinstance(value, list):
            summary["documents_count"] = len(value)
            continue
        if key == "react_trace" and isinstance(value, list):
            summary["trace_count"] = len(value)
            summary["last_trace"] = value[-1] if value else ""
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
        elif isinstance(value, list):
            summary[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            summary[f"{key}_keys"] = sorted(value.keys())
        else:
            summary[key] = str(value)
    return summary


def preview_value(value: Any, max_chars: int = 180) -> Any:
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, dict):
        return f"dict(keys={sorted(value.keys())})"
    text = str(value)
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return value


def source_counts(documents: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for doc in documents:
        origin = str(doc.get("origin", "unknown"))
        counts[origin] = counts.get(origin, 0) + 1
    return counts


def state_delta(before: dict[str, Any], after: dict[str, Any], updated_keys: list[str]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in updated_keys:
        previous = before.get(key)
        current = after.get(key)

        if key == "documents" and isinstance(current, list):
            previous_docs = previous if isinstance(previous, list) else []
            delta[key] = {
                "before_count": len(previous_docs),
                "after_count": len(current),
                "before_by_origin": source_counts(previous_docs),
                "after_by_origin": source_counts(current),
            }
            continue

        if key == "react_trace" and isinstance(current, list):
            previous_trace = previous if isinstance(previous, list) else []
            delta[key] = {
                "before_count": len(previous_trace),
                "after_count": len(current),
                "last_entry": current[-1] if current else "",
            }
            continue

        delta[key] = {
            "before": preview_value(previous),
            "after": preview_value(current),
        }
    return delta


def serialize_sources(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for doc in documents:
        serialized.append(
            {
                "title": str(doc.get("title", "")),
                "source": str(doc.get("source", "")),
                "origin": str(doc.get("origin", "")),
                "score": float(doc.get("score", 0.0)),
            }
        )
    return serialized


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


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def to_keyword_list(raw_keywords: Any) -> list[str]:
    if not isinstance(raw_keywords, list):
        return []
    cleaned: list[str] = []
    for item in raw_keywords:
        token = str(item).strip().lower()
        if token:
            cleaned.append(token)
    return cleaned


def source_text_blob(sources: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    for src in sources[:8]:
        title = str(src.get("title", "")).strip().lower()
        source = str(src.get("source", "")).strip().lower()
        if title:
            pieces.append(title)
        if source:
            pieces.append(source)
    return " ".join(pieces)


def llm_judge_answer(
    question: str,
    expected: str,
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any] | None:
    llm = get_reflection_llm()
    response = llm.invoke(
        [
            (
                "system",
                "You are a strict RAG evaluator. Return JSON only with keys: "
                "correct (bool), grounded (bool), correctness_score (0..1), faithfulness_score (0..1), reason (string).",
            ),
            (
                "human",
                (
                    f"Question: {question}\n"
                    f"Expected answer guidance: {expected}\n"
                    f"Candidate answer: {answer}\n"
                    f"Sources (JSON): {json.dumps(sources[:5], ensure_ascii=True)}\n"
                    "Rules: correctness compares candidate to expected guidance. "
                    "grounded means claims are plausibly supported by the given sources. "
                    "Do not include markdown, only JSON."
                ),
            ),
        ]
    )
    response_text = response.content if isinstance(response.content, str) else str(response.content)
    parsed = parse_json_object(response_text)
    if not parsed:
        return None

    correctness_score = clamp01(float(parsed.get("correctness_score", 0.0)))
    faithfulness_score = clamp01(float(parsed.get("faithfulness_score", 0.0)))

    return {
        "correct": bool(parsed.get("correct", correctness_score >= 0.6)),
        "grounded": bool(parsed.get("grounded", faithfulness_score >= 0.6)),
        "correctness_score": round(correctness_score, 3),
        "faithfulness_score": round(faithfulness_score, 3),
        "reason": str(parsed.get("reason", "")).strip(),
    }


def evaluate_answer(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    expected: str = "",
    expected_keywords: list[str] | None = None,
    use_llm_judge: bool = False,
) -> dict[str, Any]:
    answer_text = answer.strip()
    answer_lower = answer_text.lower()
    keywords = [k.strip().lower() for k in (expected_keywords or []) if str(k).strip()]

    matched_keywords = [keyword for keyword in keywords if keyword in answer_lower]
    if keywords:
        correctness_score = len(matched_keywords) / len(keywords)
    else:
        correctness_score = 1.0 if answer_text else 0.0

    source_blob = source_text_blob(sources)
    overlap_hits = [keyword for keyword in matched_keywords if keyword in source_blob]
    faithfulness_score = 1.0 if sources else 0.0
    if sources and keywords:
        faithfulness_score = 0.5 + 0.5 * (len(overlap_hits) / len(keywords))

    heuristic = {
        "correct": correctness_score >= 0.5 and bool(answer_text),
        "grounded": faithfulness_score >= 0.6,
        "correctness_score": round(clamp01(correctness_score), 3),
        "faithfulness_score": round(clamp01(faithfulness_score), 3),
        "reason": "keyword/source-overlap heuristic",
    }

    result = dict(heuristic)
    method = "heuristic"
    if use_llm_judge:
        judged = llm_judge_answer(question=question, expected=expected, answer=answer_text, sources=sources)
        if judged:
            result = judged
            method = "llm_judge"

    score = 0.6 * float(result["correctness_score"]) + 0.4 * float(result["faithfulness_score"])
    result.update(
        {
            "score": round(clamp01(score), 3),
            "method": method,
            "matched_keywords": matched_keywords,
            "question": question,
            "expected": expected,
        }
    )
    return result


def run_stream(
    question: str,
    thread_id: str = "1",
    json_out: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    state = build_initial_state(question)
    config = {"configurable": {"thread_id": thread_id}}
    visited_nodes: list[str] = []
    node_deltas: list[dict[str, Any]] = []
    fallback_true_events = 0
    react_plan_visits = 0
    web_search_visits = 0

    if verbose:
        print(f"Question: {question}")
        print("Streaming node updates...\n")

    for event in graph.stream(state, config=config, stream_mode="updates"):
        if not isinstance(event, dict):
            continue
        for node_name, update in event.items():
            node_name_str = str(node_name)
            visited_nodes.append(node_name_str)
            if node_name_str == "react_plan":
                react_plan_visits += 1
            if node_name_str == "web_search":
                web_search_visits += 1

            if verbose:
                print(f"[{node_name}]")
            if isinstance(update, dict):
                before = dict(state)
                if verbose:
                    print(summarize_update(update))
                state.update(update)
                updated_keys = sorted(update.keys())
                if bool(update.get("fallback", False)):
                    fallback_true_events += 1

                node_deltas.append(
                    {
                        "node": node_name_str,
                        "updated_keys": updated_keys,
                        "delta": state_delta(before, state, updated_keys),
                    }
                )
            else:
                if verbose:
                    print({"value": str(update)})
                node_deltas.append(
                    {
                        "node": node_name_str,
                        "updated_keys": [],
                        "delta": {"value": str(update)},
                    }
                )
            if verbose:
                print("-" * 60)

    if verbose:
        print("\nFinal answer:\n")
        print(state.get("generation", ""))
        print("\nFinal trace:")
        for line in state.get("react_trace", []):
            print(f"- {line}")

    retry_summary = {
        "react_plan_visits": react_plan_visits,
        "web_search_visits": web_search_visits,
        "web_retry_loops": max(0, web_search_visits - 1),
        "fallback_true_events": fallback_true_events,
    }

    report = {
        "question": question,
        "thread_id": thread_id,
        "visited_nodes": visited_nodes,
        "node_deltas": node_deltas,
        "retry_summary": retry_summary,
        "final_answer": str(state.get("generation", "")),
        "final_sources": serialize_sources(state.get("documents", [])),
        "final_state": {
            "requires_web": bool(state.get("requires_web", False)),
            "fallback": bool(state.get("fallback", False)),
            "evidence_ok": bool(state.get("evidence_ok", False)),
            "web_attempts": int(state.get("web_attempts", 0)),
            "react_step": int(state.get("react_step", 0)),
            "top_score": float(state.get("top_score", 0.0)),
            "trace": state.get("react_trace", []),
        },
    }

    if json_out:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if verbose:
            print(f"\nSaved trace report to: {out_path}")

    return report


def run_eval(
    eval_set: list[dict[str, Any]],
    thread_prefix: str = "eval",
    use_llm_judge: bool = False,
    traces_dir: str | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for index, item in enumerate(eval_set, start=1):
        question = str(item.get("question", "")).strip()
        if not question:
            continue

        expected = str(item.get("expected", "")).strip()
        expected_keywords = to_keyword_list(item.get("expected_keywords", []))
        thread_id = f"{thread_prefix}-{index}"

        json_out: str | None = None
        if traces_dir:
            json_out = str(Path(traces_dir) / f"trace_{index:02d}.json")

        report = run_stream(
            question=question,
            thread_id=thread_id,
            json_out=json_out,
            verbose=False,
        )
        evaluation = evaluate_answer(
            question=question,
            answer=str(report.get("final_answer", "")),
            sources=report.get("final_sources", []),
            expected=expected,
            expected_keywords=expected_keywords,
            use_llm_judge=use_llm_judge,
        )

        retries = int(report.get("retry_summary", {}).get("web_retry_loops", 0))
        fallback = bool(report.get("final_state", {}).get("fallback", False))

        item_result = {
            "question": question,
            "expected": expected,
            "thread_id": thread_id,
            "retries": retries,
            "fallback": fallback,
            "evaluation": evaluation,
            "final_answer": report.get("final_answer", ""),
            "final_sources": report.get("final_sources", []),
        }
        results.append(item_result)

        print(
            f"[{index}/{len(eval_set)}] score={evaluation['score']:.3f} "
            f"correct={evaluation['correct']} grounded={evaluation['grounded']} "
            f"retries={retries} fallback={fallback}"
        )

    total = len(results)
    correct_count = sum(1 for row in results if bool(row["evaluation"]["correct"]))
    grounded_count = sum(1 for row in results if bool(row["evaluation"]["grounded"]))
    total_retries = sum(int(row["retries"]) for row in results)
    fallback_count = sum(1 for row in results if bool(row["fallback"]))
    avg_score = sum(float(row["evaluation"]["score"]) for row in results) / total if total else 0.0

    metrics = {
        "num_examples": total,
        "accuracy": round(correct_count / total, 3) if total else 0.0,
        "grounded_rate": round(grounded_count / total, 3) if total else 0.0,
        "avg_score": round(avg_score, 3),
        "avg_retries": round(total_retries / total, 3) if total else 0.0,
        "fallback_rate": round(fallback_count / total, 3) if total else 0.0,
        "judge_mode": "llm_judge" if use_llm_judge else "heuristic",
    }

    summary = {
        "metrics": metrics,
        "results": results,
    }

    print("\nEvaluation summary:")
    print(json.dumps(metrics, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream LangGraph node updates for debugging.")
    parser.add_argument(
        "question",
        nargs="?",
        default="what is the difference between langchain and langsmith",
        help="Question to ask the graph",
    )
    parser.add_argument("--thread-id", default="1", help="LangGraph configurable thread id")
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write structured trace report JSON",
    )
    parser.add_argument("--run-eval", action="store_true", help="Run batch evaluation dataset")
    parser.add_argument(
        "--eval-json-out",
        default="",
        help="Optional path to write batch evaluation summary JSON",
    )
    parser.add_argument(
        "--eval-traces-dir",
        default="",
        help="Optional directory to save per-question trace JSON files during eval",
    )
    parser.add_argument("--llm-judge", action="store_true", help="Use LLM-as-judge for scoring")
    parser.add_argument(
        "--max-eval",
        type=int,
        default=0,
        help="Run only the first N eval examples (0 means all)",
    )
    args = parser.parse_args()

    if args.run_eval:
        eval_set = DEFAULT_EVAL_SET[: args.max_eval] if args.max_eval > 0 else DEFAULT_EVAL_SET
        summary = run_eval(
            eval_set=eval_set,
            thread_prefix=args.thread_id,
            use_llm_judge=args.llm_judge,
            traces_dir=args.eval_traces_dir or None,
        )

        if args.eval_json_out:
            out_path = Path(args.eval_json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Saved evaluation summary to: {out_path}")
        return

    run_stream(question=args.question, thread_id=args.thread_id, json_out=args.json_out or None)


if __name__ == "__main__":
    main()
