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
        question=(
            "According to Lejun Gong's paper on the MedBLIP model, what specific "
            "algorithm was used to enhance the text data of doctor-patient "
            "question-answering?"
        ),
        expected_ground_truth=(
            "A mirroring sample generation algorithm that transforms affirmative "
            "questions into negative ones and creates contrasting scenarios."
        ),
    ),
    TestCase(
        case_id="T02",
        question_type="Methodological",
        question=(
            "In the MedBLIP framework proposed by Lejun Gong and colleagues, what "
            "specific unfreezing strategy was applied to the large language model's "
            "decoder layers to achieve optimal performance?"
        ),
        expected_ground_truth=(
            "They unfroze 31.25% of the fully connected layer weights, specifically "
            "in the first five and last five decoder layers."
        ),
    ),
    TestCase(
        case_id="T03",
        question_type="Numerical",
        question=(
            "Based on the experimental results in Lejun Gong's study, what were the "
            "exact BLEU-1, ROUGE-1, and ROUGE-L scores for their highest-performing "
            "model (MedBLIP-31.25-AUG-VC)?"
        ),
        expected_ground_truth="BLEU-1: 62.10%, ROUGE-1: 67.30%, and ROUGE-L: 66.12%.",
    ),
    TestCase(
        case_id="T04",
        question_type="Comparative",
        question=(
            "In their data augmentation experiments, how does the team led by Lejun "
            "Gong describe the results of using GANs versus their mirroring algorithm "
            "for generating medical question-answer pairs?"
        ),
        expected_ground_truth=(
            "GAN-based training was highly challenging and produced mostly unusable "
            "samples with syntactic/logical collapse and confused medical terms, while "
            "the mirroring algorithm was highly successful."
        ),
    ),
    TestCase(
        case_id="T05",
        question_type="Error Analysis",
        question=(
            "According to the model error analysis provided in Lejun Gong's MedBLIP "
            "research, why did the model fail to accurately identify the increasing "
            "trend of a patient's edema?"
        ),
        expected_ground_truth=(
            "The model relied only on static image data and failed to capture temporal "
            "changes over time."
        ),
    ),
    TestCase(
        case_id="T06",
        question_type="Concept Retrieval",
        question=(
            "In the evaluation conducted by Lejun Gong's team, how exactly was cosine "
            "similarity utilized to analyze the model's learning process?"
        ),
        expected_ground_truth=(
            "It measured the difference between original model weights and post-fine-"
            "tuning weights to map how much medical knowledge was acquired."
        ),
    ),
    TestCase(
        case_id="T07",
        question_type="Anti-Hallucination",
        question=(
            "Does Lejun Gong's paper claim that the MedBLIP model is now capable of "
            "fully replacing human radiologists for complex clinical diagnoses?"
        ),
        expected_ground_truth=(
            "No. The model cannot fully replace human radiologists and is positioned as "
            "an auxiliary efficiency tool."
        ),
    ),
    TestCase(
        case_id="T08",
        question_type="Out-of-scope",
        question=(
            "What was the exact hardware cost to purchase the Nvidia 3090 GPUs used "
            "by Lejun Gong to train MedBLIP?"
        ),
        expected_ground_truth=(
            "Cannot be answered from the provided papers; the paper mentions Nvidia "
            "3090 24GB usage but not purchase cost."
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
        has_mirroring = contains_any(text, ["mirroring", "mirror"])
        has_generation = contains_any(text, ["sample generation", "generation algorithm", "augmentation"])
        has_transformation = contains_any(text, ["affirmative", "negative", "contrasting", "contrastive"])
        ok = has_mirroring and has_generation and has_transformation
        return ok, "Expected mirroring sample-generation algorithm with affirmative/negative contrast"

    if case_id == "T02":
        has_ratio = bool(re.search(r"\b31(?:\.25)?\s*%?\b", text))
        has_layers = contains_any(text, ["first five", "last five", "decoder", "layers"])
        has_unfreeze = contains_any(text, ["unfreeze", "unfroze", "unfreezing"])
        ok = has_ratio and has_layers and has_unfreeze
        return ok, "Expected 31.25% unfreezing in first/last five decoder layers"

    if case_id == "T03":
        has_bleu = bool(re.search(r"\b62(?:\.1|\.10)?\b", text))
        has_rouge1 = bool(re.search(r"\b67(?:\.3|\.30)?\b", text))
        has_rougel = bool(re.search(r"\b66(?:\.12)?\b", text))
        ok = has_bleu and has_rouge1 and has_rougel
        return ok, "Expected BLEU-1 62.10, ROUGE-1 67.30, ROUGE-L 66.12"

    if case_id == "T04":
        has_gan_failure = contains_any(text, ["gan", "gans"]) and contains_any(
            text,
            ["challenging", "unusable", "collapse", "confused medical", "syntactic", "logical"],
        )
        has_mirroring_success = contains_any(text, ["mirroring", "mirror"]) and contains_any(
            text, ["successful", "better", "worked", "effective"]
        )
        ok = has_gan_failure and has_mirroring_success
        return ok, "Expected GAN difficulty/failure contrasted with mirroring success"

    if case_id == "T05":
        has_static_only = contains_any(text, ["static image", "static", "single image"])
        has_temporal_gap = contains_any(text, ["temporal", "over time", "dynamic", "trend"])
        has_edema = "edema" in text
        ok = has_static_only and has_temporal_gap and has_edema
        return ok, "Expected static-image limitation and missed temporal edema trend"

    if case_id == "T06":
        has_cosine = contains_any(text, ["cosine similarity", "cosine"])
        has_weights = contains_any(text, ["weights", "original", "fine-tuning", "fine tuned", "after fine-tuning"])
        has_learning_map = contains_any(text, ["learn", "acquired", "medical knowledge", "remembered"])
        ok = has_cosine and has_weights and has_learning_map
        return ok, "Expected cosine similarity between pre/post fine-tuning weights to map learning"

    if case_id == "T07":
        has_no = bool(re.search(r"\bno\b", text)) or "not" in text
        has_not_replace = contains_any(text, ["cannot fully replace", "not replace", "cannot replace", "auxiliary"])
        has_radiologist = contains_any(text, ["radiologist", "radiologists"])
        ok = has_no and has_not_replace and has_radiologist
        return ok, "Expected explicit statement that MedBLIP does not replace radiologists"

    if case_id == "T08":
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
        mentions_missing_cost = contains_any(text, ["not stated", "not provided", "no cost", "purchase cost"])
        hallucinated_price = bool(re.search(r"\$\s*\d", raw)) or contains_any(text, ["cost was", "price was"])
        ok = (has_refusal or mentions_missing_cost) and not hallucinated_price
        return ok, "Expected refusal/insufficient-info response with no fabricated GPU price"

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
    print("=== Golden QA Evaluation Summary ===")
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
        description="Run the MedBLIP disambiguated Golden QA benchmark against the local RAG API."
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
