"""统一响应体.

约定: 所有 HTTP 接口(除 SSE 流式接口外)统一返回 ::

    {"code": "OK", "message": "success", "data": {...}, "trace_id": "a1b2c3d4e5f6"}

这样前端只需要一套拦截器逻辑, 不用为每个接口写不同的错误判断.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from app.core.exceptions import ErrorCode
from app.core.logging import get_trace_id

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """统一响应结构."""

    code: str = Field(default=ErrorCode.OK.value, description="业务状态码, OK 表示成功")
    message: str = Field(default="success", description="给人看的提示信息")
    data: T | None = Field(default=None, description="业务数据")
    trace_id: str = Field(default="-", description="链路 id, 排查问题时报给后端")

    @property
    def success(self) -> bool:
        return self.code == ErrorCode.OK.value


class PageData(BaseModel, Generic[T]):
    """分页数据结构."""

    total: int = Field(default=0, description="总条数")
    page: int = Field(default=1, description="当前页码, 从 1 开始")
    page_size: int = Field(default=20, description="每页条数")
    items: list[T] = Field(default_factory=list, description="当前页数据")


def ok(data: Any = None, message: str = "success") -> dict[str, Any]:
    """成功响应."""
    return {
        "code": ErrorCode.OK.value,
        "message": message,
        "data": data,
        "trace_id": get_trace_id(),
    }


def fail(
    code: ErrorCode | str = ErrorCode.INTERNAL_ERROR,
    message: str = "服务内部错误",
    detail: Any = None,
) -> dict[str, Any]:
    """失败响应."""
    payload: dict[str, Any] = {
        "code": code.value if isinstance(code, ErrorCode) else code,
        "message": message,
        "data": None,
        "trace_id": get_trace_id(),
    }
    if detail is not None:
        payload["detail"] = detail
    return payload
