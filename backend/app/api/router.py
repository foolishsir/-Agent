"""聚合路由. 后续新增的模块(文档/问答/会话)都在这里挂载."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import health

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["健康检查"])

# P1 阶段接入: api_router.include_router(documents.router, prefix="/documents", tags=["文档管理"])
# P2 阶段接入: api_router.include_router(chat.router, prefix="/chat", tags=["智能问答"])
