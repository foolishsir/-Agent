"""Cross-Encoder 精排.

双塔 vs 交叉编码器
------------------
**双塔(Embedding 检索)**: query 和 document 分别独立编码成向量, 最后算内积.
两者在编码时**互不可见**, 直到最后一刻才交互. 优点是可以预计算文档向量,
检索是 O(1) 的向量近邻查找; 缺点是精度有上限 —— 它只能表达"整体语义相近".

**交叉编码器(Rerank)**: 把 (query, document) 拼成一个序列送进模型,
模型内部做 token 级注意力交互. 精度高得多, 但**无法预计算** ——
n 个候选就要跑 n 次完整前向.

所以标准做法是两段式::

    全量(万级) --双塔召回--> 候选(20) --交叉编码器精排--> Top-5 进 Prompt
                 快、宽                              准、窄

本项目里精排还有一个额外作用: **它的分数可以用来拒答**.
精排分是 (query, passage) 的相关性打分, 在同一个 query 内可比,
所以"最高分低于阈值"是判断"文档里到底有没有料"的可靠依据 ——
比让 LLM 自己说"我不知道"可靠得多, 因为 LLM 在被要求回答时天然倾向编造.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from app.core.config import settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger, log_kv
from app.services.retrieval.base import RetrievedChunk

logger = get_logger("docmind.retrieval.reranker")


class CrossEncoderReranker:
    """基于 sentence-transformers CrossEncoder 的精排器."""

    def __init__(self, model_name: str, *, device: str = "cpu", max_length: int = 512) -> None:
        self._model_name = model_name
        self._device = self._resolve_device(device)
        self._max_length = max_length
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"rerank:{self._model_name}"

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        """对候选集精排, 返回 Top-N.

        注意送入模型的是 **query 与子块正文**, 而不是父块.
        原因: 子块是实际被检索匹配的粒度, 用它打分才能得到"这个匹配有多强"的
        准确判断; 父块是生成用的上下文单位, 拿它打分会被无关内容稀释.
        这也意味着精排分数与拒答阈值是同一个尺度, 可以直接比较.
        """
        if not candidates:
            return []

        started = time.perf_counter()
        model = self._ensure_loaded()

        pairs = [[query, chunk.content] for chunk in candidates]
        try:
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001 - 第三方库异常类型不稳定
            raise RetrievalError(f"重排失败: {exc}") from exc

        reranked = [
            chunk.with_score(float(score), "rerank")
            for chunk, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda c: c.score, reverse=True)

        log_kv(
            logger,
            "rerank.done",
            candidates=len(candidates),
            top_n=top_n,
            top_score=round(reranked[0].score, 4) if reranked else None,
            min_score=round(reranked[-1].score, 4) if reranked else None,
            cost_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return reranked[:top_n]

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover
                raise RetrievalError("未安装 sentence-transformers, 无法使用重排功能") from exc

            # 与 Embedding 同样的离线处理: 缓存完整时不联网
            from app.services.embedding.local_bge import (
                _activate_offline_if_cached,  # noqa: PLC0415
            )

            offline = _activate_offline_if_cached(self._model_name)

            log_kv(
                logger,
                "rerank.loading",
                model=self._model_name,
                device=self._device,
                offline=offline,
            )
            try:
                model = CrossEncoder(
                    self._model_name, device=self._device, max_length=self._max_length
                )
            except Exception as exc:  # noqa: BLE001
                raise RetrievalError(f"加载重排模型失败: {exc}") from exc

            self._model = model
            logger.info("重排模型已加载 | %s device=%s", self._model_name, self._device)
            return self._model

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if not requested.startswith("cuda"):
            return requested
        try:
            import torch
        except ImportError:  # pragma: no cover
            return "cpu"
        if torch.cuda.is_available():
            return requested
        logger.warning("配置了 %s 但 CUDA 不可用, 重排自动回退 cpu", requested)
        return "cpu"

    def release(self) -> None:
        with self._lock:
            self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass


_reranker: CrossEncoderReranker | None = None
_lock = threading.Lock()


def get_reranker() -> CrossEncoderReranker | None:
    """获取全局精排器. 未启用时返回 None."""
    if not settings.rerank_enabled or settings.rerank_provider == "none":
        return None

    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = CrossEncoderReranker(
                    settings.rerank_model, device=settings.rerank_device
                )
    return _reranker


def reset_reranker() -> None:
    global _reranker
    with _lock:
        _reranker = None
