"""Embedding 服务入口.

对外只暴露 ``get_embedding_provider()`` 和 ``release_models()``,
上层不关心用的是本地模型还是云端 API.
"""

from __future__ import annotations

import threading

from app.core.config import settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger
from app.services.embedding.base import EmbeddingProvider
from app.services.embedding.cloud import OpenAICompatibleEmbedding
from app.services.embedding.local_bge import LocalBGEEmbedding

logger = get_logger("docmind.embedding")

_provider: EmbeddingProvider | None = None
_lock = threading.Lock()


def get_embedding_provider() -> EmbeddingProvider:
    """获取全局单例 Embedding 提供方.

    用模块级变量 + 锁而不是 ``lru_cache``, 原因:
    ``release_models()`` 需要**判断是否已创建**再决定是否释放.
    用 ``lru_cache`` 的话, 检查缓存状态要靠 ``cache_info()`` 这种间接手段,
    而且释放后无法优雅地重新创建.
    """
    global _provider
    if _provider is not None:
        return _provider

    with _lock:
        if _provider is not None:
            return _provider
        _provider = _build_provider()
        logger.info("Embedding 提供方已初始化 | %s", _provider.name)
        return _provider


def _build_provider() -> EmbeddingProvider:
    provider = settings.embedding_provider

    if provider == "local":
        return LocalBGEEmbedding(
            settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            query_instruction=settings.embedding_query_instruction,
            max_length=settings.embedding_max_length,
            expected_dim=settings.embedding_dim,
        )

    if provider == "openai":
        if not settings.llm_api_key:
            raise EmbeddingError("embedding_provider=openai 需要配置 DOCMIND_LLM_API_KEY")
        return OpenAICompatibleEmbedding(
            settings.embedding_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            dim=settings.embedding_dim,
            batch_size=max(settings.embedding_batch_size, 32),
        )

    raise EmbeddingError(f"不支持的 embedding_provider: {provider!r} (可选: local / openai)")


def release_models() -> None:
    """释放本地模型占用的显存/内存.

    应用关闭时由 lifespan 调用. 注意**不要在未初始化时主动创建再释放** ——
    那会让"关闭服务"变成"加载一遍模型再关掉".
    """
    global _provider
    if _provider is None:
        return

    release = getattr(_provider, "release", None)
    if callable(release):
        try:
            release()
        except Exception:  # noqa: BLE001 - 关闭阶段不应因清理失败而中断
            logger.exception("释放 Embedding 模型时发生异常")
    _provider = None


def reset_provider_for_test() -> None:
    """测试用: 清空单例, 让下一次调用重新按当前配置构建."""
    global _provider
    with _lock:
        _provider = None


def warmup() -> None:
    """预热: 加载模型并跑一次真实推理.

    不做预热的话, 第一个真实请求要等模型加载(本地 BGE 在 CPU 上约 3~10 秒),
    用户看到的就是一次莫名其妙的超时. 预热把这份开销移到启动阶段.
    """
    provider = get_embedding_provider()
    try:
        provider.encode_query("预热")
    except Exception:  # noqa: BLE001 - 预热失败不阻塞启动, 但必须告警
        logger.exception("Embedding 预热失败, 首次请求可能会很慢")
        return
    logger.info("Embedding 预热完成 | %s dim=%s", provider.name, provider.dim)


__all__ = [
    "EmbeddingProvider",
    "LocalBGEEmbedding",
    "OpenAICompatibleEmbedding",
    "get_embedding_provider",
    "release_models",
    "reset_provider_for_test",
    "warmup",
]
