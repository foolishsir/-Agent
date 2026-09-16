"""会话历史接口.

设计要点
--------
1. **对话历史的两种模式都支持**: 传 ``conversation_id`` 走服务端持久化,
   不传则走无状态模式(由调用方自己带 ``history``).
   脚本和评测不需要持久化, Web 界面需要 —— 强行统一成一种都会别扭.
2. **软删**: 会话删除只打标记, 消息保留. 用户误删可恢复,
   而且历史问答对后续挖掘评测集很有价值.
3. **越权返回 404 而不是 403**: 见 ``conversation_service.get_conversation``.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Query, status

from app.api.deps import CurrentUser, SessionDep
from app.core.response import PageData, ok
from app.schemas.conversation import (
    ConversationDetailOut,
    ConversationOut,
    ConversationStatsOut,
    CreateConversationRequest,
    MessageOut,
    RenameConversationRequest,
)
from app.services import conversation_service

router = APIRouter()


@router.get("", summary="会话列表")
async def list_conversations(
    session: SessionDep,
    user_id: CurrentUser,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 30,
    keyword: Annotated[str | None, Query(description="按标题模糊搜索")] = None,
) -> dict[str, Any]:
    """列出会话, 最近更新的排在前面."""
    total, items = await conversation_service.list_conversations(
        session, user_id=user_id, page=page, page_size=page_size, keyword=keyword
    )
    page_data = PageData[ConversationOut](
        total=total,
        page=page,
        page_size=page_size,
        items=[ConversationOut.model_validate(item) for item in items],
    )
    return ok(page_data.model_dump(mode="json"))


@router.post("", summary="新建会话", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    session: SessionDep,
    user_id: CurrentUser,
    payload: Annotated[CreateConversationRequest | None, Body()] = None,
) -> dict[str, Any]:
    """创建一个空会话.

    也可以在第一次提问时由 ``/chat/stream`` 自动创建(传 ``conversation_id=null``
    且带 ``create_conversation=true``). 显式创建适合"用户点了新建对话"的场景.
    """
    request = payload or CreateConversationRequest()
    conversation = await conversation_service.create_conversation(
        session, user_id=user_id, title=request.title, doc_ids=request.doc_ids
    )
    return ok(ConversationOut.model_validate(conversation).model_dump(mode="json"))


@router.get("/stats", summary="对话统计")
async def conversation_stats(session: SessionDep, user_id: CurrentUser) -> dict[str, Any]:
    """会话数 / 消息数 / 拒答率.

    拒答率是很值得盯的指标: 它持续走高通常意味着检索阈值配得太严,
    或者用户问的内容确实超出了文档范围 —— 两者的应对方式完全不同.
    """
    return ok(
        ConversationStatsOut(
            **await conversation_service.stats(session, user_id=user_id)
        ).model_dump()
    )


@router.get("/{conversation_id}", summary="会话详情(含消息)")
async def get_conversation(
    session: SessionDep,
    user_id: CurrentUser,
    conversation_id: str,
    message_limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> dict[str, Any]:
    """读取会话及其全部消息, 用于前端刷新后恢复对话."""
    conversation = await conversation_service.get_conversation(
        session, conversation_id, user_id=user_id
    )
    messages = await conversation_service.list_messages(
        session, conversation_id, limit=message_limit
    )

    detail = ConversationDetailOut.model_validate(
        {
            **ConversationOut.model_validate(conversation).model_dump(),
            "messages": [MessageOut.model_validate(m) for m in messages],
        }
    )
    return ok(detail.model_dump(mode="json"))


@router.patch("/{conversation_id}", summary="重命名会话")
async def rename_conversation(
    session: SessionDep,
    user_id: CurrentUser,
    conversation_id: str,
    payload: Annotated[RenameConversationRequest, Body()],
) -> dict[str, Any]:
    conversation = await conversation_service.rename_conversation(
        session, conversation_id, user_id=user_id, title=payload.title
    )
    return ok(ConversationOut.model_validate(conversation).model_dump(mode="json"))


@router.delete("/{conversation_id}", summary="删除会话")
async def delete_conversation(
    session: SessionDep,
    user_id: CurrentUser,
    conversation_id: str,
) -> dict[str, Any]:
    """软删会话(消息保留, 便于误删恢复与后续挖掘评测集)."""
    return ok(
        await conversation_service.delete_conversation(session, conversation_id, user_id=user_id)
    )


@router.delete("/{conversation_id}/messages", summary="清空会话消息")
async def clear_messages(
    session: SessionDep,
    user_id: CurrentUser,
    conversation_id: str,
) -> dict[str, Any]:
    """清空消息但保留会话本身(标题不变)."""
    deleted = await conversation_service.clear_messages(session, conversation_id, user_id=user_id)
    return ok({"conversation_id": conversation_id, "deleted": deleted})
