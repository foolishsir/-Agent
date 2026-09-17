"""面试官 Agent 服务.

核心是 ``conductor``: 一个「评估 → 决策 → 再提问」的闭环状态机.
``verifier`` 负责问题可溯源性校验(问题必须基于简历, 不能凭空发挥).
"""

from __future__ import annotations

from app.services.interview.conductor import (
    MAX_RESUME_CHARS,
    InterviewState,
    InterviewTurn,
    next_question,
    plan_interview,
)
from app.services.interview.verifier import (
    TraceabilityResult,
    check_traceability,
    extract_terms,
)

__all__ = [
    "MAX_RESUME_CHARS",
    "InterviewState",
    "InterviewTurn",
    "TraceabilityResult",
    "check_traceability",
    "extract_terms",
    "next_question",
    "plan_interview",
]
