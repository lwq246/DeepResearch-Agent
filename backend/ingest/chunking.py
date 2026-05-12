import re
from typing import Any


MAJOR_SECTION_KEYWORDS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "materials and methods",
    "approach",
    "experiments",
    "experiment",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "future work",
    "references",
    "bibliography",
}

MAJOR_SECTION_KEYWORDS_PATTERN = "|".join(
    sorted((re.escape(keyword) for keyword in MAJOR_SECTION_KEYWORDS), key=len, reverse=True)
)

EXTRACTED_SENTINEL_HEADING_RE = re.compile(r"^\s*@@HEADER@@\s+(?P<title>[^\n]{2,160})\s*$", flags=re.I)
EXTRACTED_MARKDOWN_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(?P<title>[^\n]{2,160})\s*$", flags=re.I)
EXTRACTED_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?P<title>(?:\d+(?:\.\d+){0,3}|[IVX]{1,8})[\)\.\-:]?\s+[A-Za-z][^\n]{1,120})\s*$",
    flags=re.I,
)
EXTRACTED_KEYWORD_HEADING_RE = re.compile(
    rf"^\s*(?:(?:\d+(?:\.\d+){{0,3}}|[IVX]{{1,8}})[\)\.\-:]?\s+)?(?P<title>{MAJOR_SECTION_KEYWORDS_PATTERN})(?:\s*[:.\-])?\s*$",
    flags=re.I,
)


def normalize_pdf_text(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def canonical_heading_key(title: str) -> str:
    lowered = title.strip().lower()
    lowered = re.sub(r"^\s*(?:\d+(?:\.\d+){0,3}|[ivx]{1,8})[\)\.\-:]?\s+", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip(" :.-")


def normalize_section_title(title: str) -> str:
    cleaned = title.replace("**", "").replace("__", "").replace("`", "")
    cleaned = re.sub(r"\s+", " ", cleaned.strip())
    cleaned = re.sub(r"[ \t:\-.]+$", "", cleaned)
    return cleaned or "Preamble"


def is_semantic_section_heading(title: str) -> bool:
    candidate = normalize_section_title(title)
    normalized = canonical_heading_key(candidate)
    if normalized in MAJOR_SECTION_KEYWORDS:
        return True

    if re.fullmatch(r"(?:\d+(?:\.\d+){0,5}|[ivx]{1,8})", candidate, flags=re.I):
        return True

    if re.match(r"^(?:\d+(?:\.\d+){0,5}|[ivx]{1,8})(?:[\)\.\-:]?\s+).+", candidate, flags=re.I):
        return True

    return False


def should_exclude_section(title: str, exclude_references: bool) -> bool:
    if not exclude_references:
        return False
    normalized = canonical_heading_key(title)
    return normalized.startswith("references") or normalized.startswith("bibliography")


def split_index_on_word_boundary(text: str, start: int, max_chars: int) -> int:
    end = min(len(text), start + max_chars)
    if end >= len(text):
        return len(text)

    window = text[start:end]
    last_space = max(window.rfind(" "), window.rfind("\t"), window.rfind("\n"))
    if last_space <= 0:
        return end

    if last_space < int(max_chars * 0.6):
        return end

    return start + last_space


def overlap_suffix_on_word_boundary(text: str, overlap_chars: int) -> str:
    if not text or overlap_chars <= 0:
        return ""

    start = max(0, len(text) - overlap_chars)
    if start > 0 and start < len(text) and text[start - 1].isalnum() and text[start].isalnum():
        while start < len(text) and text[start].isalnum():
            start += 1
        while start < len(text) and text[start].isspace():
            start += 1

    suffix = text[start:].strip()
    if suffix:
        return suffix

    return text[-overlap_chars:].strip()


def split_oversized_content(content: str, max_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", content) if part and part.strip()]
    if not paragraphs:
        paragraphs = [part.strip() for part in re.split(r"(?<=[.!?])\s+", content) if part and part.strip()]
    if not paragraphs:
        return [content.strip()] if content.strip() else []

    chunks: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        candidate = paragraph if not buffer else f"{buffer}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            buffer = candidate
            continue

        if buffer:
            chunks.append(buffer.strip())
            buffer = ""

        if len(paragraph) <= max_chars:
            buffer = paragraph
            continue

        start = 0
        while start < len(paragraph):
            cut = split_index_on_word_boundary(paragraph, start=start, max_chars=max_chars)
            if cut <= start:
                cut = min(len(paragraph), start + max_chars)

            piece = paragraph[start:cut].strip()
            if piece:
                chunks.append(piece)
            start = cut
            while start < len(paragraph) and paragraph[start].isspace():
                start += 1

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks


def apply_chunk_overlap(parts: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0 or len(parts) <= 1:
        return parts

    overlapped: list[str] = [parts[0].strip()]
    for index in range(1, len(parts)):
        previous = parts[index - 1].strip()
        current = parts[index].strip()
        prefix = overlap_suffix_on_word_boundary(previous, overlap_chars=overlap_chars) if previous else ""
        if prefix:
            overlapped.append(f"{prefix}\n{current}".strip())
        else:
            overlapped.append(current)

    return overlapped


def detect_heading_from_extracted_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None

    for pattern, require_semantic_check in (
        (EXTRACTED_SENTINEL_HEADING_RE, True),
        (EXTRACTED_MARKDOWN_HEADING_RE, True),
        (EXTRACTED_NUMBERED_HEADING_RE, False),
        (EXTRACTED_KEYWORD_HEADING_RE, False),
    ):
        match = pattern.match(stripped)
        if not match:
            continue

        title = normalize_section_title(match.group("title"))
        if require_semantic_check and not is_semantic_section_heading(title):
            continue
        is_numeric_title = bool(re.fullmatch(r"(?:\d+(?:\.\d+){0,5}|[ivx]{1,8})", title, flags=re.I))
        min_len = 1 if is_numeric_title else 2
        if min_len <= len(title) <= 140:
            return title

    return None


def is_reference_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    candidate = re.sub(r"^#+\s*", "", stripped)
    candidate = candidate.replace("**", "").replace("__", "").replace("`", "")
    candidate = re.sub(r"^\s*(?:\d+(?:\.\d+){0,3}|[ivx]{1,8})[\)\.\-:]?\s+", "", candidate, flags=re.I)
    candidate = re.sub(r"\s+", " ", candidate).strip().lower()

    return candidate.startswith("references") or candidate.startswith("bibliography")


def is_running_header_or_footer_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    if re.fullmatch(r"\d{1,3}", stripped):
        return True

    without_page_no = re.sub(r"\d{1,3}$", "", stripped).strip()
    letters = [char for char in without_page_no if char.isalpha()]
    if len(letters) < 20:
        return False

    upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return upper_ratio >= 0.9


def cleanup_extracted_text_artifacts(extracted_text: str) -> str:
    lines = extracted_text.splitlines()
    cleaned: list[str] = []
    in_picture_block = False

    for line in lines:
        lowered = line.lower()

        if "start of picture text" in lowered:
            in_picture_block = True
            continue
        if "end of picture text" in lowered:
            in_picture_block = False
            continue
        if in_picture_block:
            continue

        if "intentionally omitted" in lowered:
            continue
        if is_running_header_or_footer_line(line):
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def truncate_text_at_references(extracted_text: str) -> str:
    lines = extracted_text.splitlines()
    for index, line in enumerate(lines):
        if is_reference_heading_line(line):
            return "\n".join(lines[:index]).strip()
    return extracted_text


def chunk_sections_from_extracted_text(
    extracted_text: str,
    min_chars: int,
    max_chars: int,
    chunk_overlap_chars: int,
    exclude_references: bool,
) -> list[dict[str, Any]]:
    if not extracted_text.strip():
        return []

    precleaned_text = cleanup_extracted_text_artifacts(extracted_text)
    normalized_text = truncate_text_at_references(precleaned_text) if exclude_references else precleaned_text
    lines = normalized_text.splitlines()

    raw_sections: list[dict[str, Any]] = []

    current_title = "Preamble"
    current_lines: list[str] = []

    def flush_current() -> None:
        content = normalize_pdf_text("\n".join(current_lines))
        if not content:
            return
        raw_sections.append(
            {
                "section_title": normalize_section_title(current_title),
                "chunk_text": content,
            }
        )

    for line in lines:
        heading = detect_heading_from_extracted_line(line)
        if heading:
            flush_current()
            current_title = heading
            current_lines = []
            continue
        current_lines.append(line)

    flush_current()

    if not raw_sections:
        fallback_text = normalize_pdf_text(normalized_text)
        if not fallback_text:
            return []
        raw_sections = [
            {
                "section_title": "Preamble",
                "chunk_text": fallback_text,
            }
        ]

    cleaned_sections: list[dict[str, Any]] = []
    for section in raw_sections:
        section_title = str(section["section_title"])
        if should_exclude_section(section_title, exclude_references=exclude_references):
            continue

        content = str(section["chunk_text"]).strip()
        if len(content) < min_chars:
            continue

        base_parts = [content] if len(content) <= max_chars else split_oversized_content(content, max_chars=max_chars)
        parts = apply_chunk_overlap(base_parts, overlap_chars=chunk_overlap_chars)
        for part_index, part in enumerate(parts, start=1):
            part_text = part.strip()
            if len(part_text) < min_chars:
                continue

            part_title = section_title if len(parts) == 1 else f"{section_title} (part {part_index})"
            cleaned_sections.append(
                {
                    "section_title": part_title,
                    "chunk_text": part_text,
                }
            )

    for section_index, section in enumerate(cleaned_sections):
        section["section_index"] = section_index

    return cleaned_sections
