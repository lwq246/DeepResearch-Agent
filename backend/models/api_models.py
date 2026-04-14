from typing import Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


class ChatDebugResponse(ChatResponse):
    trace: list[str]
    top_score: float
    evidence_ok: bool
    web_attempts: int
    fallback: bool


class NodeUpdate(BaseModel):
    node: str
    updated_keys: list[str]
    summary: dict[str, Any]


class ChatStreamDebugResponse(ChatDebugResponse):
    visited_nodes: list[str]
    node_updates: list[NodeUpdate]


class UploadPdfResponse(BaseModel):
    filename: str
    chunks_indexed: int
