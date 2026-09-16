"""会话与消息的业务逻辑.

本模块不依赖 FastAPI —— 与文档/配置服务保持一致的架构约定,
这样同一份逻辑可以被 HTTP 接口、评测脚本、数据导出工具复用.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError, ParamInvalidError
from app.core.logging import get_logger, log_kv
from app.models.conversation import Conversation, Message, MessageRole

logger = get_logger("docmind.conversation")

#: 会话默认标题的最大长度
_TITLE_MAX = 40

#: 新建会话的占位标题. 只有标题还是占位符时才会被第一条消息自动替换 ——
#: 用户显式命名过的会话不能被覆盖.
DEFAULT_TITLE = "新对话"

#: 生成标题时要去掉的噪音
_TITLE_NOISE_RE = re.compile(r"[\r\n\t]+")


def make_title(question: str) -> str:
    """由第一个问题生成会话标题.

    不额外调用 LLM 生成摘要, 原因:
    那会给"发出第一条消息"增加一次几百毫秒的等待, 而用户此刻正在等答案 ——
    多等一次 LLM 调用来美化一个标题, 收益和成本完全不成比例.
    截断原问题已经足够让用户认出这是哪次对话.
    """
    text = _TITLE_NOISE_RE.sub(" ", question).strip()
    text = re.sub(r"\s{2,}", " ", text)
    if not text:
        return DEFAULT_TITLE
    return text[:_TITLE_MAX] + ("…" if len(text) > _TITLE_MAX else "")


# --------------------------------------------------------------------------- #
# 会话
# --------------------------------------------------------------------------- #
async def create_conversation(
    session: AsyncSession,
    *,
    user_id: str,
    title: str | None = None,
    doc_ids: list[str] | None = None,
) -> Conversation:
    """创建一个新会话."""
    conversation = Conversation(
        user_id=user_id,
        title=(title or DEFAULT_TITLE).strip()[:200] or DEFAULT_TITLE,
        doc_ids=doc_ids or None,
    )
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)

    log_kv(logger, "conversation.created", conversation_id=conversation.id, user_id=user_id)
    return conversation


async def list_conversations(
    session: AsyncSession,
    *,
    user_id: str,
    page: int = 1,
    page_size: int = 30,
    keyword: str | None = None,
) -> tuple[int, list[Conversation]]:
    """列出会话, 最近更新的排在前面."""
    conditions = [Conversation.user_id == user_id, Conversation.is_deleted.is_(False)]
    if keyword:
        conditions.append(Conversation.title.ilike(f"%{keyword}%"))

    total = int(
        (
            await session.execute(select(func.count()).select_from(Conversation).where(*conditions))
        ).scalar_one()
    )

    rows = (
        (
            await session.execute(
                select(Conversation)
                .where(*conditions)
                # 用 updated_at 排序而不是 created_at: 一个三天前创建但刚刚还在用的会话,
                # 应该排在最前面. 这是聊天类产品的通用预期.
                .order_by(Conversation.updated_at.desc())
                .offset(max(0, (page - 1) * page_size))
                .limit(page_size)
            )
        )
        .scalars()
        .all()
    )

    return total, list(rows)


async def get_conversation(
    session: AsyncSession, conversation_id: str, *, user_id: str | None = None
) -> Conversation:
    """按 id 取会话.

    不存在、已删除、不属于该用户 —— 三种情况都返回 404.
    统一返回 404 而不是 403 是为了防止**越权探测**:
    如果对"别人的会话"返回 403, 攻击者就能靠状态码差异枚举出系统里有哪些 id.
    """
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None or conversation.is_deleted:
        raise NotFoundError(f"会话不存在: {conversation_id}")
    if user_id is not None and conversation.user_id != user_id:
        raise NotFoundError(f"会话不存在: {conversation_id}")
    return conversation


async def rename_conversation(
    session: AsyncSession, conversation_id: str, *, user_id: str, title: str
) -> Conversation:
    conversation = await get_conversation(session, conversation_id, user_id=user_id)
    cleaned = title.strip()[:200]
    if not cleaned:
        # 空标题是**参数错误(400)**, 不是"资源不存在(404)".
        # 用错状态码会让前端把它当成"会话被删了"从而清空界面 —— 明明会话还在.
        raise ParamInvalidError("标题不能为空")
    conversation.title = cleaned
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def delete_conversation(
    session: AsyncSession, conversation_id: str, *, user_id: str
) -> dict[str, Any]:
    """软删会话.

    软删而不是物理删除: 用户误删时还能恢复, 而且历史问答对后续做评测、
    统计、挖掘高频问题仍然有价值. 真正需要物理清除时再提供单独的"彻底删除".
    """
    conversation = await get_conversation(session, conversation_id, user_id=user_id)
    conversation.is_deleted = True
    await session.commit()

    log_kv(
        logger,
        "conversation.deleted",
        conversation_id=conversation_id,
        messages=conversation.message_count,
    )
    return {"conversation_id": conversation_id, "deleted_messages": conversation.message_count}


async def clear_messages(session: AsyncSession, conversation_id: str, *, user_id: str) -> int:
    """清空会话里的消息, 但保留会话本身(标题不变)."""
    conversation = await get_conversation(session, conversation_id, user_id=user_id)
    result = await session.execute(
        delete(Message).where(Message.conversation_id == conversation_id)
    )
    deleted = result.rowcount or 0

    conversation.message_count = 0
    conversation.last_message_at = None
    await session.commit()
    return deleted


# --------------------------------------------------------------------------- #
# 消息
# --------------------------------------------------------------------------- #
async def add_message(
    session: AsyncSession,
    *,
    conversation_id: str,
    role: str,
    content: str,
    citations: list[dict[str, Any]] | None = None,
    refused: bool = False,
    search_query: str | None = None,
    first_token_ms: int = 0,
    total_ms: int = 0,
    error: str | None = None,
    update_title: bool = True,
) -> Message:
    """追加一条消息, 并同步会话的冗余统计字段.

    冗余字段必须**在同一个事务里**更新, 否则会出现"消息数对不上"的脏状态.
    """
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError(f"会话不存在: {conversation_id}")

    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        citations=citations,
        refused=refused,
        search_query=(search_query or None),
        first_token_ms=first_token_ms,
        total_ms=total_ms,
        error=error,
    )
    session.add(message)

    # 会话标题还是占位符时, 用第一条用户消息自动命名.
    #
    # 判据是"标题是否仍为占位符", 而不是"消息数是否为 0" ——
    # 后者会把**用户显式命名的会话**在第一次提问时覆盖掉:
    # 用户建了个叫"钢刀维护记录"的会话, 一问"这个多少钱"就变成了"这个多少钱".
    # 这类"用户输入被系统悄悄改掉"的问题在真实产品里非常招人烦.
    if update_title and role == MessageRole.USER.value and conversation.title == DEFAULT_TITLE:
        conversation.title = make_title(content)

    now = datetime.now(UTC)
    conversation.message_count += 1
    conversation.last_message_at = now
    # 显式刷新 updated_at: TimestampMixin 的 onupdate 只在"行本身被 UPDATE"时触发,
    # 而这里确实更新了 message_count, 所以会触发. 但为了语义明确仍然显式赋值.
    conversation.updated_at = now

    await session.commit()
    await session.refresh(message)
    return message


async def list_messages(
    session: AsyncSession,
    conversation_id: str,
    *,
    limit: int = 200,
    ascending: bool = True,
) -> list[Message]:
    """读取会话内的消息."""
    order = Message.created_at.asc() if ascending else Message.created_at.desc()
    rows = (
        (
            await session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(order)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return list(rows)


async def load_history_pairs(
    session: AsyncSession,
    conversation_id: str,
    *,
    max_turns: int = 6,
) -> list[tuple[str, str]]:
    """加载最近的对话历史, 用于多轮 Query 改写.

    只取最近 ``max_turns`` 轮: 指代消解只依赖最近的一两轮,
    更早的对话对理解"它指的是什么"没有帮助, 反而会浪费 token 并干扰改写模型.

    跳过失败的消息(有 error 的助手消息内容为空或不可靠),
    避免把 "生成失败" 当成上下文喂给改写模型.
    """
    messages = await list_messages(session, conversation_id, limit=max_turns * 2 + 4)

    pairs: list[tuple[str, str]] = []
    for message in messages:
        if message.error:
            continue
        content = message.content.strip()
        if not content:
            continue
        pairs.append((message.role, content))

    return pairs[-(max_turns * 2) :]


async def stats(session: AsyncSession, *, user_id: str) -> dict[str, Any]:
    """会话统计, 用于界面展示与后续评测集挖掘."""
    total = int(
        (
            await session.execute(
                select(func.count())
                .select_from(Conversation)
                .where(Conversation.user_id == user_id, Conversation.is_deleted.is_(False))
            )
        ).scalar_one()
    )
    messages = int(
        (
            await session.execute(
                select(func.count())
                .select_from(Message)
                .join(Conversation, Conversation.id == Message.conversation_id)
                .where(Conversation.user_id == user_id, Conversation.is_deleted.is_(False))
            )
        ).scalar_one()
    )
    refused = int(
        (
            await session.execute(
                select(func.count())
                .select_from(Message)
                .join(Conversation, Conversation.id == Message.conversation_id)
                .where(
                    Conversation.user_id == user_id,
                    Conversation.is_deleted.is_(False),
                    Message.refused.is_(True),
                )
            )
        ).scalar_one()
    )

    return {
        "conversations": total,
        "messages": messages,
        "refused": refused,
        # 拒答率是很有价值的观测指标: 持续走高通常意味着
        # 检索阈值配得太严, 或者用户问的确实超出了文档范围.
        "refused_rate": round(refused / messages, 3) if messages else 0.0,
    }
