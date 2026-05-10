from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import time
import uuid
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen


@dataclass(frozen=True)
class TestCase:
    case_id: str
    question_type: str
    question: str
    expected_ground_truth: str


@dataclass
class TestResult:
    case_id: str
    question_type: str
    passed: bool
    latency_seconds: float
    reason: str
    question: str
    expected_ground_truth: str
    answer: str
    source_count: int
    evidence_ok: bool | None = None
    web_attempts: int | None = None
    fallback: bool | None = None


TEST_CASES: list[TestCase] = [
    TestCase(
        case_id="T01",
        question_type="Factual",
        question="What algorithm is used to repair off-label vertices in the Image Classifiers paper?",
        expected_ground_truth="A targeted DeepFool-style update procedure.",
    ),
    TestCase(
        case_id="T02",
        question_type="Factual",
        question="What specific fine-tuning technique was used to reduce resources in the Event Log Analysis paper?",
        expected_ground_truth="LoRA (Low-Rank Adaptation) parameter-efficient fine-tuning.",
    ),
    TestCase(
        case_id="T03",
        question_type="Methodological",
        question="How does PACE measure the plausibility score of a sample?",
        expected_ground_truth=(
            "Through an isolation forest, calculating the path length of a sample "
            "through an isolation tree."
        ),
    ),
    TestCase(
        case_id="T04",
        question_type="Numerical",
        question=(
            "What was the total cost and time taken to generate the synthetic dataset "
            "using Claude 3.7 Sonnet?"
        ),
        expected_ground_truth="It cost $23.02 and took 14 hours.",
    ),
    TestCase(
        case_id="T05",
        question_type="Numerical",
        question=(
            "In the Image Classifiers paper, how many loops were tested per model, "
            "and how many models were tested?"
        ),
        expected_ground_truth="1000 loops per model across 6 models (6000 loops total).",
    ),
    TestCase(
        case_id="T06",
        question_type="Comparative",
        question=(
            "In the PACE paper's ablation studies, what happens to the compressed "
            "ensemble size when the confidence parameter (eta) is increased?"
        ),
        expected_ground_truth=(
            "Increasing confidence (relaxing faithfulness) improves pruning and "
            "yields a smaller final ensemble."
        ),
    ),
    TestCase(
        case_id="T07",
        question_type="Comparative",
        question="Which SLM was the fastest during the testing phase of the Event Log Analysis paper?",
        expected_ground_truth="Gemma 7b (0:05 days:hours).",
    ),
    TestCase(
        case_id="T08",
        question_type="Synthesis",
        question=(
            "What is a common theme regarding resources or computation in both the "
            "PACE paper and the Event Log Analysis paper?"
        ),
        expected_ground_truth=(
            "Both reduce computational overhead for deployment (PACE via model compression, "
            "Event Log via SLMs plus LoRA)."
        ),
    ),
    TestCase(
        case_id="T09",
        question_type="Out-of-scope",
        question="What is the stock price of the company that created the Claude 3.7 model?",
        expected_ground_truth=(
            "Graceful refusal based on scope: cannot answer from provided research papers."
        ),
    ),
    TestCase(
        case_id="T10",
        question_type="Anti-Hallucination",
        question=(
            "Does the Image Classifiers paper prove mathematically that all decision "
            "regions are simply connected?"
        ),
        expected_ground_truth=(
            "No. The paper reports finite-resolution empirical evidence, not a formal proof."
        ),
    ),
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def evaluate_answer(case_id: str, answer: str) -> tuple[bool, str]:
    raw = answer.strip()
    text = normalize(raw)

    if case_id == "T01":
        ok = "deepfool" in text and contains_any(text, ["targeted", "target"]) and contains_any(
            text, ["update", "procedure", "step"]
        )
        return ok, "Expected DeepFool-style targeted update procedure"

    if case_id == "T02":
        ok = contains_any(text, ["lora", "low-rank adaptation", "low rank adaptation"])
        return ok, "Expected LoRA / Low-Rank Adaptation"

    if case_id == "T03":
        ok = "isolation forest" in text and contains_any(text, ["path length", "path-length", "path"]) and contains_any(
            text, ["isolation tree", "tree"]
        )
        return ok, "Expected isolation forest path-length explanation"

    if case_id == "T04":
        has_cost = bool(re.search(r"\$?\s*23(?:\.0?2)?", raw.lower()))
        has_time = bool(re.search(r"\b14\s*(hours?|hrs?|h)\b", raw.lower()))
        ok = has_cost and has_time
        return ok, "Expected both $23.02 and 14 hours"

    if case_id == "T05":
        has_1000 = bool(re.search(r"\b1000\b", text))
        has_6_models = bool(re.search(r"\b6\b", text)) and contains_any(text, ["model", "models"])
        has_6000 = bool(re.search(r"\b6000\b", text))
        ok = has_1000 and has_6_models and has_6000
        return ok, "Expected 1000 loops, 6 models, and 6000 total"

    if case_id == "T06":
        has_direction = contains_any(text, ["increase", "increased", "higher", "relax"])
        has_effect = contains_any(text, ["smaller", "reduce", "reduced", "decrease", "pruning"])
        has_object = contains_any(text, ["ensemble", "compressed ensemble", "final ensemble"])
        ok = has_direction and has_effect and has_object
        return ok, "Expected increased eta -> stronger pruning -> smaller ensemble"

    if case_id == "T07":
        ok = contains_all(text, ["gemma", "7b"])
        return ok, "Expected Gemma 7b as fastest SLM"

    if case_id == "T08":
        has_resource_theme = contains_any(text, ["resource", "computation", "computational", "latency", "overhead", "cost"])
        has_reduction = contains_any(text, ["reduce", "reduced", "lower", "optimiz", "efficient"])
        has_pace_side = contains_any(text, ["pace", "ensemble", "compression", "model size"])
        has_event_side = contains_any(text, ["event log", "slm", "lora", "small language model"])
        ok = has_resource_theme and has_reduction and has_pace_side and has_event_side
        return ok, "Expected shared resource/computation reduction theme across both papers"

    if case_id == "T09":
        has_refusal = contains_any(
            text,
            [
                "cannot answer",
                "can't answer",
                "cannot determine",
                "insufficient",
                "not in the provided",
                "based on the provided",
                "out of scope",
                "do not have",
            ],
        )
        hallucinated_price = bool(re.search(r"\$\s*\d", raw)) or "stock price is" in text
        ok = has_refusal and not hallucinated_price
        return ok, "Expected graceful refusal for out-of-scope stock-price question"

    if case_id == "T10":
        has_no = bool(re.search(r"\bno\b", text)) or "not" in text
        has_nonproof = contains_any(text, ["not a formal", "rather than a formal", "not formal", "no formal proof", "not prove mathematically"])
        has_empirical = contains_any(text, ["empirical", "finite-resolution", "finite resolution"])
        ok = has_no and (has_nonproof or has_empirical)
        return ok, "Expected explicit non-proof + empirical evidence framing"

    return False, "Unknown case id"


def post_chat_message(base_url: str, endpoint: str, question: str, timeout: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    body = json.dumps({"message": question}).encode("utf-8")
    request = Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc


def encode_multipart_file(field_name: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----rag-agent-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()

    body_prefix = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{file_path.name}\"\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    body_suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")

    return body_prefix + file_bytes + body_suffix, boundary


def upload_pdf(base_url: str, endpoint: str, file_path: Path, timeout: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    body, boundary = encode_multipart_file("file", file_path)

    request = Request(
        url=url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url} while uploading {file_path.name}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach {url} while uploading {file_path.name}: {exc.reason}") from exc


def upload_pdfs_from_dir(
    base_url: str,
    endpoint: str,
    pdf_dir: Path,
    pdf_glob: str,
    timeout: float,
) -> None:
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        raise RuntimeError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(path for path in pdf_dir.glob(pdf_glob) if path.is_file())
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in {pdf_dir} matching pattern '{pdf_glob}'")

    print()
    print(f"Uploading {len(pdf_files)} PDF file(s) from: {pdf_dir}")
    for index, pdf_file in enumerate(pdf_files, start=1):
        started_at = time.perf_counter()
        payload = upload_pdf(
            base_url=base_url,
            endpoint=endpoint,
            file_path=pdf_file,
            timeout=timeout,
        )
        latency_seconds = time.perf_counter() - started_at
        chunks = payload.get("chunks_indexed", "?")
        print(
            f"[{index}/{len(pdf_files)}] Uploaded {pdf_file.name} "
            f"(chunks={chunks}, {latency_seconds:.2f}s)"
        )


def run_tests(base_url: str, endpoint: str, timeout: float) -> list[TestResult]:
    results: list[TestResult] = []

    for test_case in TEST_CASES:
        started_at = time.perf_counter()
        payload = post_chat_message(
            base_url=base_url,
            endpoint=endpoint,
            question=test_case.question,
            timeout=timeout,
        )
        latency_seconds = time.perf_counter() - started_at

        answer = str(payload.get("answer", ""))
        passed, reason = evaluate_answer(test_case.case_id, answer)

        sources = payload.get("sources", [])
        source_count = len(sources) if isinstance(sources, list) else 0

        evidence_ok_value: bool | None = None
        web_attempts_value: int | None = None
        fallback_value: bool | None = None

        if "evidence_ok" in payload:
            evidence_ok_value = bool(payload.get("evidence_ok"))
        if "web_attempts" in payload:
            try:
                web_attempts_value = int(payload.get("web_attempts"))
            except (TypeError, ValueError):
                web_attempts_value = None
        if "fallback" in payload:
            fallback_value = bool(payload.get("fallback"))

        results.append(
            TestResult(
                case_id=test_case.case_id,
                question_type=test_case.question_type,
                passed=passed,
                latency_seconds=latency_seconds,
                reason=reason,
                question=test_case.question,
                expected_ground_truth=test_case.expected_ground_truth,
                answer=answer,
                source_count=source_count,
                evidence_ok=evidence_ok_value,
                web_attempts=web_attempts_value,
                fallback=fallback_value,
            )
        )

    return results


def print_summary(results: list[TestResult]) -> None:
    print()
    print("=== Ten-Question Evaluation Summary ===")
    print(
        f"{'Case':<6} {'Type':<18} {'Status':<8} {'Latency(s)':<11} {'Sources':<8} {'Evidence':<9} {'WebTry':<7}"
    )
    print("-" * 82)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        evidence = "-" if result.evidence_ok is None else str(result.evidence_ok)
        web_try = "-" if result.web_attempts is None else str(result.web_attempts)
        print(
            f"{result.case_id:<6} {result.question_type:<18} {status:<8} "
            f"{result.latency_seconds:<11.2f} {result.source_count:<8} {evidence:<9} {web_try:<7}"
        )

    total = len(results)
    passed = sum(1 for item in results if item.passed)
    failed = total - passed
    print("-" * 82)
    print(f"Passed: {passed}/{total} | Failed: {failed}/{total}")

    if failed:
        print()
        print("Failed Cases:")
        for result in results:
            if result.passed:
                continue
            print(f"- {result.case_id} ({result.question_type})")
            print(f"  Question: {result.question}")
            print(f"  Expected: {result.expected_ground_truth}")
            print(f"  Actual: {result.answer}")
            print(f"  Check: {result.reason}")


def write_json_report(path: Path, results: list[TestResult], endpoint: str, base_url: str) -> None:
    report = {
        "base_url": base_url,
        "endpoint": endpoint,
        "total": len(results),
        "passed": sum(1 for item in results if item.passed),
        "failed": sum(1 for item in results if not item.passed),
        "results": [asdict(item) for item in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent
    default_pdf_dir = workspace_root / "paper"
    default_report_path = script_dir / "logs" / "ten_question_eval_latest.json"

    parser = argparse.ArgumentParser(
        description="Run the 10 benchmark QA test questions against the local RAG API."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="FastAPI base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--endpoint",
        default="/chat/debug",
        help="API endpoint to call (default: /chat/debug)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout per question in seconds (default: 120)",
    )
    parser.add_argument(
        "--report-json",
        default=str(default_report_path),
        help=f"Path to write JSON report (default: {default_report_path})",
    )
    parser.add_argument(
        "--upload-before-run",
        action="store_true",
        help="Upload PDFs from --pdf-dir before running tests.",
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(default_pdf_dir),
        help=f"Directory containing PDFs to upload (default: {default_pdf_dir})",
    )
    parser.add_argument(
        "--pdf-glob",
        default="*.pdf",
        help="Glob pattern for PDF files in --pdf-dir (default: *.pdf)",
    )
    parser.add_argument(
        "--upload-endpoint",
        default="/upload-pdf",
        help="Upload endpoint used with --upload-before-run (default: /upload-pdf)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.upload_before_run:
        try:
            upload_pdfs_from_dir(
                base_url=args.base_url,
                endpoint=args.upload_endpoint,
                pdf_dir=Path(args.pdf_dir),
                pdf_glob=args.pdf_glob,
                timeout=args.timeout,
            )
        except Exception as exc:
            print(f"Error while uploading PDFs: {exc}", file=sys.stderr)
            return 2

    try:
        results = run_tests(
            base_url=args.base_url,
            endpoint=args.endpoint,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"Error while running tests: {exc}", file=sys.stderr)
        return 2

    print_summary(results)

    report_path = Path(args.report_json)
    write_json_report(
        path=report_path,
        results=results,
        endpoint=args.endpoint,
        base_url=args.base_url,
    )
    print()
    print(f"JSON report written to: {report_path}")

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
