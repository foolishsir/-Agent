"""智能问答接口.

提供两个版本:

- ``POST /chat/stream`` —— **SSE 流式**(Web 界面用). 边生成边推送, 首字延迟低,
  并且能把检索阶段的进度也推给前端.
- ``POST /chat`` —— 非流式(脚本 / 自动化测试 / 第三方集成用).
  内部复用同一条事件流, 只是把事件收集完再一次性返回.

为什么两个都要: 流式是为交互体验服务的, 但它对调用方有要求(要能消费 SSE).
提供一个等价的非流式接口, 可以让自动化测试和批处理脚本简单很多,
也方便写评测脚本(评测不需要打字机效果, 需要的是结构化结果).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.core.response import ok
from app.services.rag import answer_stream

router = APIRouter()


class ChatTurn(BaseModel):
    """一轮历史对话."""

    role: str = Field(description="user 或 assistant")
    content: str


class ChatRequest(BaseModel):
    """问答请求."""

    question: str = Field(min_length=1, max_length=1000, description="用户问题")
    doc_ids: list[str] | None = Field(
        default=None,
        description="限定检索范围。不传则检索当前用户所有已就绪的文档",
    )
    history: list[ChatTurn] = Field(
        default_factory=list,
        description="历史对话(按时间正序)。用于多轮追问的指代消解",
    )

    def history_pairs(self) -> list[tuple[str, str]]:
        return [
            (turn.role, turn.content) for turn in self.history if turn.role in {"user", "assistant"}
        ]


def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    """把事件编码成 SSE 帧."""
    return {
        "event": event,
        # ensure_ascii=False 保证中文原样输出; SSE 本身是 UTF-8 编码
        "data": json.dumps(data, ensure_ascii=False),
    }


@router.post("/stream", summary="流式问答 (SSE)")
async def chat_stream(
    session: SessionDep,
    user_id: CurrentUser,
    payload: Annotated[ChatRequest, Body()],
) -> EventSourceResponse:
    """流式问答.

    事件类型见 ``app/services/rag/pipeline.py`` 的模块文档:
    ``stage`` / ``sources`` / ``token`` / ``citations`` / ``done`` / ``error``.

    **前端必须处理 ``error`` 事件** —— 流已经开始后无法用 HTTP 状态码表达失败,
    错误只能作为事件推过去. 漏处理的话, 用户会看到一个永远停在半句的回答.
    """

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        try:
            async for event in answer_stream(
                session,
                payload.question,
                user_id=user_id,
                doc_ids=payload.doc_ids,
                history=payload.history_pairs(),
            ):
                yield _sse(event.event, event.data)
        except Exception as exc:  # noqa: BLE001 - 生成器内异常无法再走全局处理器
            # 走到这里说明连 answer_stream 自己都没兜住(例如会话失效).
            # 必须显式转成 error 事件, 否则连接会静默断开, 前端只能靠超时发现.
            yield _sse("error", {"code": "INTERNAL_ERROR", "message": f"服务内部错误: {exc}"})

    return EventSourceResponse(
        event_generator(),
        # 心跳: 检索阶段可能耗时较久, 没有心跳时中间的代理/浏览器
        # 可能判定连接空闲而主动断开
        ping=15,
        headers={
            # 关掉 Nginx 等反向代理的缓冲, 否则流式会被攒成一次输出,
            # 打字机效果完全失效 —— 这是 SSE 部署最经典的坑
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


@router.post("", summary="问答 (非流式)")
async def chat(
    session: SessionDep,
    user_id: CurrentUser,
    payload: Annotated[ChatRequest, Body()],
) -> dict[str, Any]:
    """非流式问答: 收集完整事件流后一次性返回.

    适合脚本调用、自动化测试与评测集批跑 —— 它们不需要打字机效果,
    需要的是结构化的最终结果.
    """
    final: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    error: dict[str, Any] | None = None

    async for event in answer_stream(
        session,
        payload.question,
        user_id=user_id,
        doc_ids=payload.doc_ids,
        history=payload.history_pairs(),
    ):
        if event.event == "stage":
            stages.append(event.data)
        elif event.event == "sources":
            sources = event.data.get("sources", [])
        elif event.event == "citations":
            citations = event.data.get("citations", [])
        elif event.event == "done":
            final = event.data
        elif event.event == "error":
            error = event.data

    if error is not None:
        return ok(
            {
                "answer": "",
                "refused": False,
                "error": error,
                "citations": [],
                "sources": sources,
                "stages": stages,
            }
        )

    return ok(
        {
            "answer": final.get("answer", ""),
            "refused": final.get("refused", False),
            "reason": final.get("reason", ""),
            "search_query": final.get("search_query", payload.question),
            "citations": citations,
            "sources": sources,
            "stages": stages,
            "timing": {
                "first_token_ms": final.get("first_token_ms"),
                "generation_ms": final.get("generation_ms"),
                "total_ms": final.get("total_ms"),
            },
        }
    )


@router.get("/config", summary="问答链路当前配置")
async def chat_config() -> dict[str, Any]:
    """暴露当前检索参数, 便于前端展示与排查.

    把参数显式暴露出来是有意的: 用户问"为什么没搜到"时,
    第一件事就是确认当前的召回条数、是否开启重排、拒答阈值是多少.
    """
    return ok(
        {
            "vector_top_k": settings.vector_top_k,
            "bm25_top_k": settings.bm25_top_k,
            "rrf_k": settings.rrf_k,
            "final_top_k": settings.final_top_k,
            "rerank_enabled": settings.rerank_enabled,
            "rerank_model": settings.rerank_model if settings.rerank_enabled else None,
            "rerank_min_score": settings.rerank_min_score,
            "llm_configured": settings.llm_configured,
            "llm_model": settings.llm_model,
        }
    )
