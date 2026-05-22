from __future__ import annotations

import json
import mimetypes
import os
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

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


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
    llm_pass: bool | None = None
    llm_reason: str | None = None
    llm_score: float | None = None


TEST_CASES: list[TestCase] = [
    TestCase(
        case_id="T01",
        question_type="Factual",
        question=(
            "In the disaster management framework proposed by Md. Yasin Kabir, "
            "what specific deep learning architecture is used to classify the tweets?"
        ),
        expected_ground_truth=(
            "A combination of an attention-based Bi-directional Long Short-Term "
            "Memory (BLSTM) and a Convolutional Neural Network (CNN)."
        ),
    ),
    TestCase(
        case_id="T02",
        question_type="Numerical",
        question=(
            "According to the CrisisLex dataset evaluation conducted by Md. Yasin "
            "Kabir, exactly how many total data instances remained after "
            "preprocessing the merged datasets, and specifically how many of those "
            "instances were allocated for validation?"
        ),
        expected_ground_truth=(
            "13,738 total data instances remained, and exactly 1,030 instances "
            "were allocated for validation."
        ),
    ),
    TestCase(
        case_id="T03",
        question_type="Methodological",
        question=(
            "According to the methodology in Fabian Akkerman's PACE paper, what "
            "specific computational technique is iteratively used to search for "
            "separating samples after each solution of the pruning problem?"
        ),
        expected_ground_truth=(
            "Constraint programming (CP) techniques."
        ),
    ),
    TestCase(
        case_id="T04",
        question_type="Comparative",
        question=(
            "In the numerical experiments conducted by Fabian Akkerman on the PACE "
            "model, what was the maximum speedup achieved by their constraint-based "
            "separation formulation over the FIPE method?"
        ),
        expected_ground_truth="Up to a 37x speedup.",
    ),
    TestCase(
        case_id="T05",
        question_type="Anti-Hallucination",
        question=(
            "In the tweet classifier evaluation on Hurricane Harvey and Irma data "
            "by Md. Yasin Kabir, did the researchers successfully train and "
            "evaluate their model on all six of their original target classes, "
            "including the Injured and Sick labels?"
        ),
        expected_ground_truth=(
            "No. The authors explicitly discarded the Injured and Sick labels "
            "during evaluation because there was a lack of enough data instances."
        ),
    ),
    TestCase(
        case_id="T06",
        question_type="Methodological",
        question=(
            "In the MedBLIP architecture proposed by Lejun Gong, what specific "
            "unfreezing strategy was applied to the large language model's decoder "
            "layers to achieve optimal performance?"
        ),
        expected_ground_truth=(
            "They unfroze 31.25% of the fully connected layer weights, specifically "
            "targeting the first five and last five decoder layers."
        ),
    ),
    TestCase(
        case_id="T07",
        question_type="Factual",
        question=(
            "According to Lejun Gong's paper on the MedBLIP model, what specific "
            "algorithm was used to enhance the text data of doctor-patient question-answering?"
        ),
        expected_ground_truth="A mirroring sample generation algorithm.",
    ),
    TestCase(
        case_id="T08",
        question_type="Numerical",
        question=(
            "Based on the experiments by Lejun Gong, what were the exact BLEU-1, "
            "ROUGE-1, and ROUGE-L scores for the highest-performing MedBLIP-31.25-AUG-VC model?"
        ),
        expected_ground_truth="BLEU-1: 62.10%, ROUGE-1: 67.30%, and ROUGE-L: 66.12%.",
    ),
    TestCase(
        case_id="T09",
        question_type="Web Fallback - Conceptual",
        question=(
            "When local document data is insufficient, what is the fundamental difference "
            "between using RAG (Retrieval-Augmented Generation) versus fine-tuning for "
            "domain-specific QA?"
        ),
        expected_ground_truth=(
            "RAG retrieves dynamic, external facts without altering model weights, while "
            "fine-tuning bakes static knowledge and specific styles into the model's "
            "internal weights."
        ),
    ),
    TestCase(
        case_id="T10",
        question_type="Web Fallback - Real-Time",
        question=(
            "What were the biggest AI product launches last month? Search online."
        ),
        expected_ground_truth=(
            "Returns a summarized list of AI released in April 2026"
        ),
    ),
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


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


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def required_env(name: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    value = raw.strip()
    if not value:
        raise RuntimeError(f"Empty required environment variable: {name}")
    return value


def required_float_env(name: str) -> float:
    value = parse_float(required_env(name))
    if value is None:
        raise RuntimeError(f"Invalid float for {name}")
    return value


def required_bool_env(name: str) -> bool:
    raw = required_env(name).lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean for {name}: {raw}")


def load_local_dotenv() -> None:
    if load_dotenv is None:
        return
    script_dir = Path(__file__).resolve().parent
    load_dotenv(script_dir / ".env", override=False)


def llm_judge_answer(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    case_id: str,
    question: str,
    expected_ground_truth: str,
    answer: str,
) -> tuple[bool, str, float | None]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a fair QA evaluator. Compare a model answer against the expected ground truth. "
                    "Prefer semantic equivalence over exact wording. Mark correct when the core expected fact is present, "
                    "even if extra details are included, unless those details directly contradict the ground truth. "
                    "Return JSON only with keys: correct (boolean), reason (string), score (number between 0 and 1)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Case ID: {case_id}\n"
                    f"Question: {question}\n"
                    f"Expected Ground Truth: {expected_ground_truth}\n"
                    f"Model Answer: {answer}\n"
                    "Judge based on factual and semantic alignment. Extra non-contradictory details are allowed."
                ),
            },
        ],
    }
    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            raw_response = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url} while LLM-judging case {case_id}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach {url} while LLM-judging case {case_id}: {exc.reason}") from exc

    response_json = json.loads(raw_response)
    choices = response_json.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"LLM judge returned no choices for case {case_id}")

    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    content_text = str(content)

    parsed = parse_json_object(content_text)
    if parsed is not None:
        correct = coerce_bool(parsed.get("correct", False), default=False)
        reason = str(parsed.get("reason", "")).strip() or "No reason provided by LLM judge"
        score = parse_float(parsed.get("score"))
        return correct, reason, score

    lowered = content_text.lower()
    if "incorrect" in lowered:
        return False, f"Unstructured LLM judge output: {content_text[:220]}", None
    if "correct" in lowered:
        return True, f"Unstructured LLM judge output: {content_text[:220]}", None
    return False, f"Could not parse LLM judge output: {content_text[:220]}", None


def run_llm_judging(
    results: list[TestResult],
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
) -> None:
    print()
    print(f"Running LLM judge for {len(results)} case(s) with model: {model}")
    for index, result in enumerate(results, start=1):
        correct, reason, score = llm_judge_answer(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            case_id=result.case_id,
            question=result.question,
            expected_ground_truth=result.expected_ground_truth,
            answer=result.answer,
        )
        result.llm_pass = correct
        result.llm_reason = reason
        result.llm_score = score
        result.passed = correct
        result.reason = reason

        status = "PASS" if correct else "FAIL"
        score_label = "-" if score is None else f"{score:.2f}"
        print(f"[{index}/{len(results)}] {result.case_id}: {status} (llm_score={score_label})")


def case_passed(result: TestResult) -> bool:
    return bool(result.passed)


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
                passed=False,
                latency_seconds=latency_seconds,
                reason="LLM judge pending",
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
    print("=== Golden QA Evaluation Summary ===")
    print(
        f"{'Case':<6} {'Type':<28} {'LLM':<8} {'Score':<8} {'Latency(s)':<11} {'Sources':<8} {'Evidence':<9} {'WebTry':<7}"
    )
    print("-" * 108)

    for result in results:
        llm_status = "PASS" if result.passed else "FAIL"
        score_label = "-" if result.llm_score is None else f"{result.llm_score:.2f}"
        evidence = "-" if result.evidence_ok is None else str(result.evidence_ok)
        web_try = "-" if result.web_attempts is None else str(result.web_attempts)
        print(
            f"{result.case_id:<6} {result.question_type:<28} {llm_status:<8} {score_label:<8} "
            f"{result.latency_seconds:<11.2f} {result.source_count:<8} {evidence:<9} {web_try:<7}"
        )

    total = len(results)
    passed = sum(1 for item in results if case_passed(item))
    failed = total - passed
    print("-" * 108)
    print("Primary judge: LLM")
    print(f"Passed: {passed}/{total} | Failed: {failed}/{total}")

    if failed:
        print()
        print("Failed Cases:")
        for result in results:
            if case_passed(result):
                continue
            print(f"- {result.case_id} ({result.question_type})")
            print(f"  Question: {result.question}")
            print(f"  Expected: {result.expected_ground_truth}")
            print(f"  Actual: {result.answer}")
            if result.llm_reason:
                print(f"  Check: {result.llm_reason}")
            else:
                print(f"  Check: {result.reason}")


def write_json_report(
    path: Path,
    results: list[TestResult],
    endpoint: str,
    base_url: str,
) -> None:
    judge_mode = "llm"
    passed_count = sum(1 for item in results if case_passed(item))
    failed_count = len(results) - passed_count
    report = {
        "base_url": base_url,
        "endpoint": endpoint,
        "judge_mode": judge_mode,
        "total": len(results),
        "passed": passed_count,
        "failed": failed_count,
        "results": [asdict(item) for item in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    load_local_dotenv()
    script_dir = Path(__file__).resolve().parent
    base_url = required_env("RAG_TEST_BASE_URL")
    endpoint = required_env("RAG_TEST_ENDPOINT")
    timeout = required_float_env("RAG_TEST_TIMEOUT")

    report_json = required_env("RAG_TEST_REPORT_JSON")
    upload_before_run = required_bool_env("RAG_TEST_UPLOAD_BEFORE_RUN")
    pdf_dir = Path(required_env("RAG_TEST_PDF_DIR"))
    pdf_glob = required_env("RAG_TEST_PDF_GLOB")
    upload_endpoint = required_env("RAG_TEST_UPLOAD_ENDPOINT")

    llm_judge_model = required_env("OPENAI_CHAT_MODEL")
    llm_judge_base_url = required_env("OPENAI_BASE_URL")
    llm_judge_api_key = required_env("OPENAI_API_KEY")
    llm_judge_timeout = required_float_env("RAG_TEST_LLM_JUDGE_TIMEOUT")

    if upload_before_run:
        try:
            upload_pdfs_from_dir(
                base_url=base_url,
                endpoint=upload_endpoint,
                pdf_dir=pdf_dir,
                pdf_glob=pdf_glob,
                timeout=timeout,
            )
        except Exception as exc:
            print(f"Error while uploading PDFs: {exc}", file=sys.stderr)
            return 2

    try:
        results = run_tests(
            base_url=base_url,
            endpoint=endpoint,
            timeout=timeout,
        )
    except Exception as exc:
        print(f"Error while running tests: {exc}", file=sys.stderr)
        return 2

    try:
        run_llm_judging(
            results,
            model=llm_judge_model,
            base_url=llm_judge_base_url,
            api_key=llm_judge_api_key,
            timeout=llm_judge_timeout,
        )
    except Exception as exc:
        print(f"Error while running LLM judge: {exc}", file=sys.stderr)
        return 2

    print_summary(results)

    report_path = Path(report_json)
    write_json_report(
        path=report_path,
        results=results,
        endpoint=endpoint,
        base_url=base_url,
    )
    print()
    print(f"JSON report written to: {report_path}")

    return 0 if all(case_passed(result) for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
