"""RAG 编排层: Prompt 组装 / 引用校验 / 事件流式问答."""

from app.services.rag.citation import (
    CitationCheck,
    build_citation_payload,
    parse_citation_numbers,
    validate_answer,
)
from app.services.rag.pipeline import RagEvent, answer_stream, resolve_doc_ids
from app.services.rag.prompts import REFUSAL_ANSWER, SYSTEM_PROMPT, build_user_prompt

__all__ = [
    "REFUSAL_ANSWER",
    "SYSTEM_PROMPT",
    "CitationCheck",
    "RagEvent",
    "answer_stream",
    "build_citation_payload",
    "build_user_prompt",
    "parse_citation_numbers",
    "resolve_doc_ids",
    "validate_answer",
]
