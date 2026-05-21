from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    max_author_candidates: int = 70
    max_context_docs: int = 12
    max_chunks_per_paper: int = 3
    max_chunks_per_paper_author_query: int = 8
    max_unique_papers: int = 4


@dataclass(frozen=True)
class UploadDebugConfig:
    preview_chunks: int = 3
    preview_chars: int = 160


RETRIEVAL_CONFIG = RetrievalConfig()
UPLOAD_DEBUG_CONFIG = UploadDebugConfig()
