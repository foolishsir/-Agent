"""面试官 Agent 接口.

三个接口对应面试的三个阶段::

    POST /interview/start    上传简历 → 生成提纲 → 问第一个问题
    POST /interview/next     回传对话 → 评估回答 → 决定追问/换话题 → 问下一个
    POST /interview/summary  结束面试 → 生成评估报告

**面试状态由前端持有并每次回传**(``outline`` / ``turns``). 这是有意为之:

- 问答链路本来就是无状态的, 面试沿用同一套模式, 不用为了 I1 先加表;
- 面试会话的数据量很小(提纲 + 十几轮对话), 回传成本可以忽略;
- 调试时能直接在请求体里看到完整状态, 比翻数据库方便.

持久化(把面试记录存成 Conversation)留到 I3.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter
from sqlalchemy import select

from app.core.exceptions import ParamInvalidError
from app.core.logging import get_logger, log_kv
from app.core.response import ok
from app.db.session import get_session_factory
from app.models.document import Chunk, Document, DocumentStatus
from app.services.interview import (
    MAX_RESUME_CHARS,
    InterviewState,
    InterviewTurn,
    check_traceability,
    next_question,
    plan_interview,
)
from app.services.llm import ChatMessage, get_llm_client
from app.services.skills import compose_skills, get_skill_registry

router = APIRouter()
logger = get_logger("docmind.api.interview")


# --------------------------------------------------------------------------- #
# 简历加载
# --------------------------------------------------------------------------- #
async def _load_resume(doc_id: str) -> str:
    """把文档还原成纯文本, 供面试官通读.

    走的是**父块**而不是检索: 面试官要看全局, 不是找片段.
    父块本来就是按章节切分的, 按 ``order_index`` 拼回去基本等于原文.

    为什么不用原始 PDF 再解析一次: 解析结果已经在数据库里了,
    重新解析既要几百毫秒又要处理那堆字体坑, 没必要.
    """
    async with get_session_factory()() as session:
        document = await session.get(Document, doc_id)
        if document is None:
            raise ParamInvalidError(f"文档不存在: {doc_id}")
        if document.status != DocumentStatus.READY:
            raise ParamInvalidError(f"文档尚未处理完成(当前状态: {document.status}), 无法开始面试")

        # 优先父块(章节级, 语义完整); 没有父块时退化为全部子块
        stmt = (
            select(Chunk.content)
            .where(Chunk.doc_id == doc_id, Chunk.chunk_type == "parent")
            .order_by(Chunk.order_index)
        )
        rows = (await session.execute(stmt)).scalars().all()

        if not rows:
            stmt = select(Chunk.content).where(Chunk.doc_id == doc_id).order_by(Chunk.order_index)
            rows = (await session.execute(stmt)).scalars().all()

    text = "\n\n".join(row for row in rows if row and row.strip())
    if not text.strip():
        raise ParamInvalidError("文档内容为空, 无法开始面试")

    return text


def _resolve_skills(skill_ids: list[str]) -> list[Any]:
    """解析勾选的 SKILL. 返回空列表表示用默认风格(不报错)."""
    if not skill_ids:
        return []
    return get_skill_registry().resolve([str(s) for s in skill_ids])


def _restore_state(payload: dict[str, Any], resume: str, skill_ids: list[str]) -> InterviewState:
    """从前端回传的 JSON 还原面试状态."""
    outline: list[dict[str, str]] = []
    for item in payload.get("outline") or []:
        if isinstance(item, dict) and item.get("opening"):
            outline.append(
                {
                    "topic": str(item.get("topic", "")),
                    "angle": str(item.get("angle", "")),
                    "opening": str(item.get("opening", "")),
                }
            )

    turns: list[InterviewTurn] = []
    for item in payload.get("turns") or []:
        if not isinstance(item, dict):
            continue
        turns.append(
            InterviewTurn(
                question=str(item.get("question", "")),
                answer=str(item.get("answer", "")),
                evaluation=item.get("evaluation") or {},
                decision=str(item.get("decision", "")),
            )
        )

    return InterviewState(
        doc_id=str(payload.get("doc_id", "")),
        skill_ids=skill_ids,
        resume=resume,
        outline=outline,
        turns=turns,
        follow_up_depth=int(payload.get("follow_up_depth") or 0),
        topic_index=int(payload.get("topic_index") or 0),
    )


# --------------------------------------------------------------------------- #
# 接口
# --------------------------------------------------------------------------- #
@router.post("/start", summary="开始面试")
async def start_interview(payload: dict[str, Any]) -> dict[str, Any]:
    """读简历 → 规划提纲 → 抛出第一个问题.

    这里会**真实调用两次模型**(规划提纲 + 生成问题), 所以比问答的首字慢.
    但这是必要的: 提纲决定了整场面试有没有主线.
    """
    doc_id = str(payload.get("doc_id") or "").strip()
    if not doc_id:
        raise ParamInvalidError("缺少 doc_id")

    skill_ids = [str(s) for s in (payload.get("skill_ids") or [])]
    skills = _resolve_skills(skill_ids)

    started = time.perf_counter()
    resume = await _load_resume(doc_id)
    resume_cost_ms = int((time.perf_counter() - started) * 1000)

    truncated = len(resume) > MAX_RESUME_CHARS
    if truncated:
        resume = resume[:MAX_RESUME_CHARS]

    state = InterviewState(doc_id=doc_id, skill_ids=skill_ids, resume=resume)
    state.outline = await plan_interview(resume, skills)

    result = await next_question(state, skills)
    question = str(result.get("question", ""))

    log_kv(
        logger,
        "interview.started",
        doc_id=doc_id,
        resume_chars=len(resume),
        truncated=truncated,
        topics=len(state.outline),
        skills=[s.id for s in skills],
    )

    return ok(
        {
            "doc_id": doc_id,
            "skill_ids": [s.id for s in skills],
            "outline": state.outline,
            "question": question,
            "decision": result.get("decision", "OPEN"),
            "constraints": result.get("constraints", {}),
            "traceability": check_traceability(question, resume).to_dict(),
            "resume_chars": len(resume),
            "resume_truncated": truncated,
            "resume_cost_ms": resume_cost_ms,
            "follow_up_depth": state.follow_up_depth,
            "topic_index": state.topic_index,
        }
    )


@router.post("/next", summary="提交回答, 取下一个问题")
async def next_turn(payload: dict[str, Any]) -> dict[str, Any]:
    """提交本轮回答 → 评估 → 决策 → 下一个问题.

    ``decision`` 会告诉前端这一步**为什么**这么走:
    ``FOLLOW_UP`` 是追着刚才的回答深挖, ``NEXT_TOPIC`` 是换话题,
    ``FINISH`` 是到轮次上限了. 前端可以据此显示"追问 2/3".
    """
    doc_id = str(payload.get("doc_id") or "").strip()
    if not doc_id:
        raise ParamInvalidError("缺少 doc_id")

    skill_ids = [str(s) for s in (payload.get("skill_ids") or [])]
    skills = _resolve_skills(skill_ids)
    resume = await _load_resume(doc_id)[:MAX_RESUME_CHARS]

    state = _restore_state(payload, resume, skill_ids)
    result = await next_question(state, skills)

    if result.get("finished"):
        return ok(
            {
                "finished": True,
                "reason": result.get("reason", ""),
                "question": "",
                "decision": "FINISH",
            }
        )

    question = str(result.get("question", ""))
    evaluation = result.get("evaluation") or {}

    return ok(
        {
            "finished": False,
            "question": question,
            "decision": result.get("decision", ""),
            "evaluation": evaluation,
            "evaluation_hint": _evaluation_hint(evaluation),
            "traceability": check_traceability(question, resume).to_dict(),
            "follow_up_depth": state.follow_up_depth,
            "topic_index": state.topic_index,
            "constraints": result.get("constraints", {}),
        }
    )


SUMMARY_SYSTEM = """你是一个面试官，面试刚刚结束。请给候选人一份复盘报告。

只输出 JSON，不要 markdown 代码块：
{"overall": "整体评价，3~5 句话",
 "highlights": [{"point": "亮点", "evidence": "候选人原话或简述"}],
 "concerns": [{"point": "顾虑/不足", "evidence": "候选人原话或简述"}],
 "suggestions": ["改进建议，2~4 条"],
 "score": {"technical_depth": 0, "expression": 0, "authenticity": 0}}

score 每项 0~10 的整数。evidence 必须来自候选人真实说过的内容，
**不允许编造**。如果某类为空就给空数组。"""


@router.post("/summary", summary="结束面试并生成复盘报告")
async def interview_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """根据完整对话生成复盘报告."""
    skill_ids = [str(s) for s in (payload.get("skill_ids") or [])]
    skills = _resolve_skills(skill_ids)
    skill_prompt, _ = compose_skills(skills)

    turns = payload.get("turns") or []
    if not turns:
        raise ParamInvalidError("面试还没有任何对话, 无法生成报告")

    transcript = "\n\n".join(
        f"面试官：{t.get('question', '')}\n候选人：{t.get('answer', '') or '（未回答）'}"
        for t in turns
        if isinstance(t, dict)
    )

    llm = get_llm_client()
    result = await llm.achat(
        [
            ChatMessage(role="system", content=SUMMARY_SYSTEM),
            ChatMessage(
                role="user",
                content=(
                    f"【面试风格】\n{skill_prompt}\n\n"
                    f"【完整对话记录】\n{transcript}\n\n请输出 JSON 复盘报告。"
                ),
            ),
        ],
        temperature=0.3,
        max_tokens=1500,
    )

    # 报告解析失败不报错, 把原文返回给前端展示 —— 总比丢掉一次面试的结果好
    text = (result.content or "").strip()
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        report = json.loads(cleaned.strip())
    except Exception:  # noqa: BLE001
        logger.warning("复盘报告 JSON 解析失败, 返回原文")
        report = {"overall": text, "highlights": [], "concerns": [], "suggestions": [], "score": {}}

    return ok({"report": report, "turn_count": len(turns)})


def _evaluation_hint(evaluation: dict[str, Any]) -> str:
    """把评估结果翻译成一句人话, 给前端做轻量展示."""
    if not evaluation:
        return ""
    depth_label = {
        "shallow": "偏浅",
        "medium": "中等",
        "deep": "有深度",
    }.get(str(evaluation.get("depth", "")), "")
    parts = [depth_label] if depth_label else []
    if evaluation.get("has_numbers"):
        parts.append("有具体数据")
    if evaluation.get("has_tradeoff"):
        parts.append("讲到了取舍")
    vague = evaluation.get("vague_words") or []
    if vague:
        parts.append(f"有模糊表述({'、'.join(map(str, vague))})")
    return " · ".join(parts)
