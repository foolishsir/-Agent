"""RAG 编排: 检索 → 组装 Prompt → 流式生成 → 引用校验.

以**事件流**的形式产出结果, 而不是等全部完成再返回.
这样前端可以做到:

- 检索完成就显示"找到 N 段相关资料"(用户不用盯着空白转圈)
- 逐字显示答案(感知延迟远低于等待完整响应)
- 答案结束后再补上经过服务端校验的引用列表

事件类型
--------
============  ==========================================================
``stage``     阶段进度(改写/检索/生成), 用于前端展示进度与后续做耗时分析
``sources``   检索到的候选上下文(未校验), 让用户知道"参考了哪些段落"
``token``     生成的文本增量
``citations`` 经过**服务端校验**的引用列表(已剥离编造编号)
``done``      结束, 附带耗时与 token 用量
``error``     出错
============  ==========================================================
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppException, LLMNotConfiguredError
from app.core.logging import get_logger, log_kv
from app.models.conversation import Conversation, MessageRole
from app.models.document import Document, DocumentStatus
from app.services import conversation_service
from app.services.llm import ChatMessage, get_llm_client
from app.services.rag import prompts
from app.services.rag.citation import build_citation_payload, validate_answer
from app.services.retrieval import retrieve

logger = get_logger("docmind.rag")


@dataclass
class RagEvent:
    """一个 SSE 事件."""

    event: str
    data: dict[str, Any] = field(default_factory=dict)


#: 这些词出现时说明当前问题很可能依赖上文, 需要做改写
_FOLLOWUP_HINTS = (
    "它",
    "他",
    "她",
    "这个",
    "那个",
    "这些",
    "那些",
    "该",
    "此",
    "上面",
    "上述",
    "前面",
    "刚才",
    "呢",
    "还有",
    "继续",
    "那么",
    "然后",
)


async def resolve_doc_ids(
    session: AsyncSession,
    user_id: str,
    requested: list[str] | None = None,
) -> list[str]:
    """确定本次检索范围.

    默认只检索**已就绪(READY)**的文档:
    - 处理中的文档还没进向量库, 检索不到, 但纳入范围会让人以为"应该能搜到"
    - 失败的文档更不该参与检索
    - 已删除的文档必须排除, 这是"幽灵数据"的第一道防线
    """
    stmt = select(Document.id).where(
        Document.user_id == user_id,
        Document.status == DocumentStatus.READY.value,
    )
    if requested:
        stmt = stmt.where(Document.id.in_(requested))

    return list((await session.execute(stmt)).scalars().all())


async def answer_stream(
    session: AsyncSession,
    question: str,
    *,
    user_id: str,
    doc_ids: list[str] | None = None,
    history: list[tuple[str, str]] | None = None,
    conversation_id: str | None = None,
) -> AsyncIterator[RagEvent]:
    """执行完整的问答链路, 产出事件流.

    对话历史的两种模式
    ------------------
    - 传 ``conversation_id``: 从数据库读取历史, 并把本轮问答**落库**.
      Web 界面用这种模式 —— 刷新页面后对话还在.
    - 不传: 使用调用方传入的 ``history``, 完全不落库.
      脚本、自动化测试、评测批跑用这种模式 —— 它们不需要持久化,
      强行落库只会在数据库里堆一堆垃圾会话.
    """
    started = time.perf_counter()
    history = history or []
    conversation: Conversation | None = None

    # ---------------- ⓪ 会话绑定 ----------------
    if conversation_id:
        conversation = await conversation_service.get_conversation(
            session, conversation_id, user_id=user_id
        )
        # 会话没有指定文档范围时, 沿用请求里的; 会话指定了则以会话为准.
        # 以会话为准是为了让"这个会话只问这份文档"的意图在多轮之间保持一致 ——
        # 否则用户换了文档范围, 同一个会话前后两轮检索的范围就不一样了.
        if conversation.doc_ids:
            doc_ids = conversation.doc_ids

        if not history:
            history = await conversation_service.load_history_pairs(session, conversation_id)

        yield RagEvent(
            "conversation", {"conversation_id": conversation.id, "title": conversation.title}
        )

    # ---------------- ① 解析检索范围 ----------------
    scope_started = time.perf_counter()
    resolved_ids = await resolve_doc_ids(session, user_id, doc_ids)
    yield RagEvent(
        "stage",
        {
            "stage": "scope",
            "detail": f"检索范围内有 {len(resolved_ids)} 份就绪文档",
            "doc_count": len(resolved_ids),
            "cost_ms": round((time.perf_counter() - scope_started) * 1000, 1),
        },
    )

    if not resolved_ids:
        answer = "当前没有已完成处理的文档，请先在「文档管理」上传 PDF 并等待处理完成。"
        await _persist_turn(
            session,
            conversation,
            question=question,
            answer=answer,
            refused=True,
            total_ms=int((time.perf_counter() - started) * 1000),
        )
        yield RagEvent(
            "done",
            {
                "answer": answer,
                "refused": True,
                "reason": "no_documents",
                "citations": [],
                "total_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return

    # ---------------- ② 多轮改写 ----------------
    search_query = question
    rewritten = False
    if history and _needs_rewrite(question):
        rewrite_started = time.perf_counter()
        try:
            search_query = await _rewrite_query(question, history)
            rewritten = search_query.strip() != question.strip()
        except Exception:  # noqa: BLE001 - 改写失败不该中断问答, 退回原问题即可
            logger.exception("Query 改写失败, 使用原问题检索")
            search_query = question

        yield RagEvent(
            "stage",
            {
                "stage": "rewrite",
                "detail": f"检索查询: {search_query}" if rewritten else "问题已自包含, 无需改写",
                "rewritten": rewritten,
                "search_query": search_query,
                "cost_ms": round((time.perf_counter() - rewrite_started) * 1000, 1),
            },
        )

    # ---------------- ③ 检索 ----------------
    retrieval_started = time.perf_counter()
    try:
        result = await retrieve(session, search_query, user_id=user_id, doc_ids=resolved_ids)
    except AppException as exc:
        yield RagEvent("error", {"code": exc.code.value, "message": exc.message})
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("检索失败")
        yield RagEvent("error", {"code": "RETRIEVAL_FAILED", "message": f"检索失败: {exc}"})
        return

    yield RagEvent(
        "stage",
        {
            "stage": "retrieval",
            "detail": f"检索到 {len(result.contexts)} 段相关内容",
            "contexts": len(result.contexts),
            "top_score": round(result.top_score, 4),
            "trace": result.trace.to_dict(),
            "cost_ms": round((time.perf_counter() - retrieval_started) * 1000, 1),
        },
    )

    # ---------------- ④ 拒答 ----------------
    if result.refused or not result.contexts:
        answer = prompts.REFUSAL_ANSWER
        yield RagEvent("token", {"text": answer})
        await _persist_turn(
            session,
            conversation,
            question=question,
            answer=answer,
            refused=True,
            search_query=search_query,
            total_ms=int((time.perf_counter() - started) * 1000),
        )
        yield RagEvent(
            "done",
            {
                "answer": answer,
                "refused": True,
                "reason": result.refuse_reason,
                "citations": [],
                "total_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return

    # 先把候选来源推给前端, 用户立刻知道"参考了哪些段落", 不用等答案生成完
    yield RagEvent("sources", {"sources": [c.to_dict() for c in result.contexts]})

    # ---------------- ⑤ 流式生成 ----------------
    llm = get_llm_client()
    if not llm.configured:
        exc = LLMNotConfiguredError()
        await _persist_turn(
            session,
            conversation,
            question=question,
            answer="",
            refused=False,
            search_query=search_query,
            error=exc.message,
            total_ms=int((time.perf_counter() - started) * 1000),
        )
        yield RagEvent("error", {"code": exc.code.value, "message": exc.message})
        return

    messages = [
        ChatMessage(role="system", content=prompts.SYSTEM_PROMPT),
        ChatMessage(role="user", content=prompts.build_user_prompt(question, result.contexts)),
    ]

    generation_started = time.perf_counter()
    first_token_ms: float | None = None
    pieces: list[str] = []

    try:
        async for piece in llm.astream(messages):
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - generation_started) * 1000
            pieces.append(piece)
            yield RagEvent("token", {"text": piece})
    except AppException as exc:
        # 已经吐出部分内容后失败: 不能重试(会重复输出), 只能告知前端.
        # 但已生成的部分要落库 —— 否则用户刷新页面后, 这次失败的对话会凭空消失,
        # 而"刚才那条回答哪去了"是很让人困惑的.
        await _persist_turn(
            session,
            conversation,
            question=question,
            answer="".join(pieces),
            refused=False,
            search_query=search_query,
            first_token_ms=int(first_token_ms or 0),
            total_ms=int((time.perf_counter() - started) * 1000),
            error=exc.message,
        )
        yield RagEvent("error", {"code": exc.code.value, "message": exc.message})
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("生成失败")
        await _persist_turn(
            session,
            conversation,
            question=question,
            answer="".join(pieces),
            refused=False,
            search_query=search_query,
            first_token_ms=int(first_token_ms or 0),
            total_ms=int((time.perf_counter() - started) * 1000),
            error=f"生成失败: {exc}",
        )
        yield RagEvent("error", {"code": "LLM_ERROR", "message": f"生成失败: {exc}"})
        return

    raw_answer = "".join(pieces)

    # ---------------- ⑥ 引用校验(服务端硬约束) ----------------
    check = validate_answer(raw_answer, len(result.contexts))
    citations = build_citation_payload(result.contexts, check.valid)

    if check.hallucinated:
        logger.warning(
            "模型编造了引用编号, 已剥离 | mentioned=%s valid=%s invalid=%s",
            check.mentioned,
            check.valid,
            check.invalid,
        )

    total_ms = round((time.perf_counter() - started) * 1000, 1)
    generation_ms = round((time.perf_counter() - generation_started) * 1000, 1)

    log_kv(
        logger,
        "rag.done",
        question_len=len(question),
        contexts=len(result.contexts),
        answer_len=len(check.cleaned_answer),
        citations=len(citations),
        invalid_citations=check.invalid,
        first_token_ms=round(first_token_ms or 0, 1),
        generation_ms=generation_ms,
        total_ms=total_ms,
    )

    yield RagEvent(
        "citations",
        {
            "citations": citations,
            "check": check.to_dict(),
            # 前端需要知道"编号被剥离过", 才能给出诚实的提示
            "answer_corrected": check.cleaned_answer != raw_answer,
            "cleaned_answer": check.cleaned_answer,
        },
    )

    yield RagEvent(
        "done",
        {
            "answer": check.cleaned_answer,
            "refused": False,
            "citations": citations,
            "search_query": search_query,
            "first_token_ms": round(first_token_ms or 0, 1),
            "generation_ms": generation_ms,
            "total_ms": total_ms,
        },
    )

    await _persist_turn(
        session,
        conversation,
        question=question,
        answer=check.cleaned_answer,
        refused=False,
        citations=citations,
        search_query=search_query,
        first_token_ms=int(first_token_ms or 0),
        total_ms=int(total_ms),
    )


# --------------------------------------------------------------------------- #
# 内部
# --------------------------------------------------------------------------- #
async def _persist_turn(
    session: AsyncSession,
    conversation: Conversation | None,
    *,
    question: str,
    answer: str,
    refused: bool,
    citations: list[dict[str, Any]] | None = None,
    search_query: str | None = None,
    first_token_ms: int = 0,
    total_ms: int = 0,
    error: str | None = None,
) -> None:
    """把本轮问答落库(仅在绑定了会话时).

    为什么在**最后**统一写, 而不是"先写用户消息、生成完再写助手消息":

    流式生成的整个过程中, 数据库会话正被生成协程持有. 如果在生成前先 commit
    一次用户消息, 那么生成期间的任何异常回滚都会把这条消息一起带走 ——
    状态反而更不一致. 统一在终态写入, 事务边界清晰:
    这一轮要么完整落库(含 error 字段), 要么完全不落.

    失败时也写 —— 并且带上 ``error`` 字段. 用户刷新后能看到"这条回答生成失败了",
    而不是发现提问凭空消失.

    持久化失败**不能**影响已经返回给用户的答案: 用户已经看到完整回答了,
    此时再抛异常只会让前端显示"失败", 与事实不符. 所以这里只记日志.
    """
    if conversation is None:
        return

    try:
        await conversation_service.add_message(
            session,
            conversation_id=conversation.id,
            role=MessageRole.USER.value,
            content=question,
        )
        await conversation_service.add_message(
            session,
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT.value,
            content=answer,
            citations=citations,
            refused=refused,
            search_query=search_query,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            error=error,
        )
    except Exception:  # noqa: BLE001
        logger.exception("对话落库失败(不影响已返回的答案) | conversation_id=%s", conversation.id)


def _needs_rewrite(question: str) -> bool:
    """判断当前问题是否依赖上文.

    优化点: 如果问题已经自包含(没有代词、没有省略), 跳过改写能省一次 LLM 调用.
    这是很划算的: 多轮对话里大约一半的追问是自包含的.
    """
    stripped = question.strip()
    if len(stripped) <= 8:
        return True
    return any(hint in stripped for hint in _FOLLOWUP_HINTS)


async def _rewrite_query(question: str, history: list[tuple[str, str]]) -> str:
    """用轻量 LLM 调用把追问改写成自包含的检索查询."""
    llm = get_llm_client()
    if not llm.configured:
        return question

    result = await llm.achat(
        [
            ChatMessage(role="system", content=prompts.QUERY_REWRITE_SYSTEM),
            ChatMessage(role="user", content=prompts.build_rewrite_prompt(question, history)),
        ],
        temperature=0.0,
        max_tokens=128,
    )

    # 模型偶尔会带引号或"改写后："之类的前缀, 清洗掉
    text = result.content.strip().strip("\"\u201c\u201d'")
    text = re.sub(r"^(改写后|检索查询|查询)[:：]\s*", "", text)
    return text.strip() or question


__all__ = ["RagEvent", "answer_stream", "resolve_doc_ids"]
