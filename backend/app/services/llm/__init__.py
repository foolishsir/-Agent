"""LLM 服务入口.

对外暴露 ``get_llm_client()`` / ``reset_llm_client()`` / ``test_connection()``.
"""

from __future__ import annotations

import threading
import time

from app.core.config import settings
from app.core.exceptions import LLMNotConfiguredError
from app.core.logging import get_logger
from app.services.llm.base import ChatMessage, ChatResult, ChatUsage, LLMProvider
from app.services.llm.openai_compatible import OpenAICompatibleLLM

logger = get_logger("docmind.llm")

_client: LLMProvider | None = None
_lock = threading.Lock()


def get_llm_client() -> LLMProvider:
    """获取全局 LLM 客户端.

    每次调用都**重新读取配置**来构造, 但因为做了单例缓存,
    只有配置变化时才会真正重建(由 ``reset_llm_client`` 触发).
    这样"在界面上改了 Key 立即生效"才能成立.
    """
    global _client
    if _client is not None:
        return _client

    with _lock:
        if _client is not None:
            return _client
        _client = OpenAICompatibleLLM(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        logger.info(
            "LLM 客户端已初始化 | base_url=%s model=%s", settings.llm_base_url, settings.llm_model
        )
        return _client


def reset_llm_client() -> None:
    """丢弃缓存的客户端, 让下次调用按最新配置重建.

    配置变更后必须调用, 否则会出现"界面上改了却没用"的假象.
    """
    global _client
    with _lock:
        _client = None
    logger.info("LLM 客户端已重置, 下次调用将使用最新配置")


async def test_connection() -> dict[str, object]:
    """真实调用一次模型, 验证配置是否可用.

    只做格式校验是不够的 —— "Key 看起来对"和"Key 真的能用"是两回事.
    余额不足、模型名写错、地址不通, 都只有真实调用才会暴露.
    """
    client = get_llm_client()
    if not client.configured:
        raise LLMNotConfiguredError()

    started = time.perf_counter()
    result: ChatResult = await client.achat(
        [
            ChatMessage(role="system", content="你是一个测试助手, 只回一个字。"),
            ChatMessage(role="user", content="回复「好」这一个字即可。"),
        ],
        temperature=0.0,
        max_tokens=16,
    )
    cost_ms = int((time.perf_counter() - started) * 1000)

    return {
        "ok": True,
        "model": result.model or settings.llm_model,
        "base_url": settings.llm_base_url,
        "reply": result.content.strip()[:50],
        "cost_ms": cost_ms,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        },
        "message": f"连接成功, 耗时 {cost_ms} ms",
    }


__all__ = [
    "ChatMessage",
    "ChatResult",
    "ChatUsage",
    "LLMProvider",
    "OpenAICompatibleLLM",
    "get_llm_client",
    "reset_llm_client",
    "test_connection",
]
