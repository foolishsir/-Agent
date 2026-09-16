"""会话与消息的请求/响应模型."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ConversationOut(BaseModel):
    """会话摘要(列表用)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    message_count: int = 0
    doc_ids: list[str] | None = None
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None = None


class MessageOut(BaseModel):
    """一条消息."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str = Field(description="user 或 assistant")
    content: str
    citations: list[dict[str, Any]] | None = None
    refused: bool = False
    search_query: str | None = Field(
        default=None, description="多轮改写后的实际检索查询(排查检索问题时看这个)"
    )
    first_token_ms: int = 0
    total_ms: int = 0
    error: str | None = None
    created_at: datetime


class ConversationDetailOut(ConversationOut):
    """会话详情(含消息列表)."""

    messages: list[MessageOut] = Field(default_factory=list)


class CreateConversationRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200, description="不传则自动生成")
    doc_ids: list[str] | None = Field(
        default=None, description="限定该会话的检索范围; 不传表示全部已就绪文档"
    )


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationStatsOut(BaseModel):
    conversations: int = 0
    messages: int = 0
    refused: int = 0
    refused_rate: float = Field(default=0.0, description="拒答率, 持续走高说明检索阈值可能过严")
