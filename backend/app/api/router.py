"""聚合路由. 后续新增的模块(文档/问答/会话)都在这里挂载."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import chat, conversations, documents, health, interview, skills, speech
from app.api.v1 import settings as settings_api

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["健康检查"])
api_router.include_router(documents.router, prefix="/documents", tags=["文档管理"])
api_router.include_router(chat.router, prefix="/chat", tags=["智能问答"])
api_router.include_router(conversations.router, prefix="/conversations", tags=["会话历史"])
api_router.include_router(skills.router, prefix="/skills", tags=["SKILL 管理"])
api_router.include_router(interview.router, prefix="/interview", tags=["面试官 Agent"])
api_router.include_router(speech.router, prefix="/speech", tags=["语音交互"])
api_router.include_router(settings_api.router, prefix="/settings", tags=["配置管理"])
