"""文档接口的请求/响应模型.

请求/响应模型与 ORM 模型**刻意分开**:
- ORM 模型跟着数据库表结构走, 加一个内部字段不应该影响对外接口
- 响应模型能精确控制暴露哪些字段(比如 ``file_path`` 绝不能返回给前端)
- 两者解耦后, 将来换 ORM 或换数据库不影响接口契约
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import DocumentStatus


class DocumentOut(BaseModel):
    """文档详情."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="文档 id")
    filename: str = Field(description="原始文件名")
    status: str = Field(description="处理状态: PENDING/PARSING/EMBEDDING/READY/FAILED")
    file_size: int = Field(description="文件大小(字节)")
    file_md5: str = Field(description="内容 MD5, 作为幂等键")

    page_count: int = Field(default=0, description="页数")
    char_count: int = Field(default=0, description="有效字符数")
    parent_chunk_count: int = Field(default=0, description="父块数量")
    child_chunk_count: int = Field(default=0, description="子块数量(入向量库的数量)")
    chunks_indexed: int = Field(default=0, description="已写入向量库的分块数")

    parse_cost_ms: int = Field(default=0, description="解析耗时(毫秒)")
    embed_cost_ms: int = Field(default=0, description="向量化耗时(毫秒)")

    error_msg: str | None = Field(default=None, description="失败原因")
    created_at: datetime
    updated_at: datetime

    @property
    def is_ready(self) -> bool:
        return self.status == DocumentStatus.READY.value


class DocumentStatusOut(BaseModel):
    """精简状态, 供前端高频轮询使用.

    轮询接口不应该返回完整详情 —— 前端每 2 秒拉一次, 传输量和序列化开销都要省.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    progress: int = Field(ge=0, le=100, description="处理进度百分比")
    error_msg: str | None = None


class UploadResponse(BaseModel):
    """上传接口响应."""

    document: DocumentOut
    created: bool = Field(
        description="是否为新文档. false 表示命中内容 MD5 幂等, 复用了已有文档(秒传)"
    )
    message: str


class DeleteResponse(BaseModel):
    """删除接口响应."""

    doc_id: str
    deleted_chunks: int = Field(description="删除的关系库分块数")
    deleted_vectors: int = Field(description="删除的向量条数")
    file_removed: bool = Field(description="原文件是否已删除")


class IngestStatsOut(BaseModel):
    """入库统计(调试与性能分析用)."""

    doc_id: str
    status: str
    page_count: int
    parent_chunks: int
    child_chunks: int
    char_count: int
    parse_cost_ms: int
    embed_cost_ms: int
    error: str | None = None


def build_status_out(document: Any) -> DocumentStatusOut:
    """由文档记录推导带进度的状态响应."""
    progress_map = {
        DocumentStatus.PENDING.value: 5,
        DocumentStatus.PARSING.value: 35,
        DocumentStatus.EMBEDDING.value: 75,
        DocumentStatus.READY.value: 100,
        DocumentStatus.FAILED.value: 0,
        DocumentStatus.DELETED.value: 0,
    }
    return DocumentStatusOut(
        id=document.id,
        status=document.status,
        progress=progress_map.get(document.status, 0),
        error_msg=document.error_msg,
    )
