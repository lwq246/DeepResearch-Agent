import json
import os
from typing import Any

from langchain_openai import ChatOpenAI

from .env import int_env_default
from .env import openai_client_kwargs
from .sources import extract_pdf_text_with_pymupdf4llm_from_bytes


def build_author_check_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_AUTHOR_CHECK_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")),
        temperature=0,
        max_tokens=int_env_default("OPENAI_AUTHOR_CHECK_MAX_TOKENS", 160),
        **openai_client_kwargs(),
    )


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


def normalize_author_list(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        name = str(item).strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(name)
    return normalized


def extract_authors_from_first_page(
    pdf_bytes: bytes,
    fallback_authors: list[str],
    author_check_llm: ChatOpenAI,
) -> list[str]:
    first_page_text = extract_pdf_text_with_pymupdf4llm_from_bytes(pdf_bytes=pdf_bytes, max_pages=1)
    if not first_page_text.strip():
        return fallback_authors

    response = author_check_llm.invoke(
        [
            (
                "system",
                (
                    "Extract paper author names from first-page text. "
                    "Return strict JSON only with this schema: "
                    '{"authors": ["Author One", "Author Two"]}. '
                    "Do not include affiliations, emails, or explanations."
                ),
            ),
            (
                "human",
                f"First page text:\n{first_page_text[:12000]}",
            ),
        ]
    )

    response_text = response.content if isinstance(response.content, str) else str(response.content)
    parsed = parse_json_object(response_text)
    if not parsed:
        return fallback_authors

    maybe_authors = parsed.get("authors")
    if not isinstance(maybe_authors, list):
        return fallback_authors

    normalized = normalize_author_list(maybe_authors)
    return normalized if normalized else fallback_authors
