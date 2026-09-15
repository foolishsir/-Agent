"""本地 BGE 系列 Embedding 实现.

关键实现点
----------
1. **进程级单例**: 模型加载要几十秒, 必须只加载一次
2. **线程安全**: 用锁保护惰性加载, 避免并发请求同时触发加载导致显存翻倍
3. **GPU 降级**: 配置成 cuda 但机器没有 GPU 时自动回退 cpu, 而不是启动就崩
4. **``torch.inference_mode()``**: 关闭梯度计算, 省显存也更快
5. **query / passage 分开编码**: BGE 中文系列的指令前缀只加在 query 侧
6. **维度 API 版本兼容**: sentence-transformers 5.x 把
   ``get_sentence_embedding_dimension`` 改名为 ``get_embedding_dimension``
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger, log_kv

logger = get_logger("docmind.embedding.local")


class LocalBGEEmbedding:
    """基于 sentence-transformers 的本地 Embedding."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        query_instruction: str = "",
        max_length: int | None = None,
        expected_dim: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._query_instruction = query_instruction
        self._max_length = max_length
        self._expected_dim = expected_dim

        self._device = self._resolve_device(device)
        self._model: Any | None = None
        self._dim: int | None = None
        # 惰性加载必须加锁: 并发首次调用会同时进入加载分支,
        # 结果是显存里出现两份模型(甚至 OOM)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 属性
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return f"local:{self._model_name}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._ensure_loaded()
        return self._dim or 0

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------ #
    # 编码
    # ------------------------------------------------------------------ #
    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """编码文档片段(**不加**指令前缀)."""
        return self._encode(list(texts))

    def encode_query(self, text: str) -> list[float]:
        """编码查询(**加上**指令前缀).

        注意: 不是所有模型都需要指令. bge-m3、text-embedding-3-* 是对称的,
        换模型时要把 ``embedding_query_instruction`` 配成空串.
        """
        prepared = f"{self._query_instruction}{text}" if self._query_instruction else text
        vectors = self._encode([prepared])
        return vectors[0] if vectors else []

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._ensure_loaded()
        try:
            import torch

            # inference_mode 比 no_grad 更彻底: 连 autograd 的元信息都不记录
            with torch.inference_mode():
                vectors = model.encode(
                    texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=True,  # 归一化后内积 == 余弦相似度
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except Exception as exc:  # noqa: BLE001 - 第三方库异常类型不稳定
            raise EmbeddingError(f"向量化失败: {exc}") from exc

        return [[float(x) for x in row] for row in vectors]

    # ------------------------------------------------------------------ #
    # 模型加载
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model

        with self._lock:
            # 双重检查: 等锁期间可能已被其他线程加载完成
            if self._model is not None:
                return self._model

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise EmbeddingError(
                    "未安装 sentence-transformers, 无法使用本地 Embedding 模型. "
                    "请执行: pip install sentence-transformers"
                ) from exc

            log_kv(
                logger,
                "embedding.loading",
                model=self._model_name,
                device=self._device,
            )
            try:
                model = SentenceTransformer(self._model_name, device=self._device)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    f"加载本地模型 {self._model_name} 失败: {exc}. "
                    "国内网络请确认已设置 HF_ENDPOINT=https://hf-mirror.com"
                ) from exc

            if self._max_length:
                # 截断长度影响显存占用与吞吐; 中文 512 通常够用
                model.max_seq_length = self._max_length

            self._model = model
            self._dim = self._resolve_dimension(model)

            if self._expected_dim and self._dim != self._expected_dim:
                # 不抛异常, 只告警: 配置里的 dim 只是声明, 以模型实际输出为准.
                # 但如果向量库集合已经按旧维度建好了, 写入会失败 —— 必须让用户看见.
                logger.warning(
                    "配置的 embedding_dim=%s 与模型实际维度=%s 不一致, "
                    "请确认向量库集合是否需要用新维度重建",
                    self._expected_dim,
                    self._dim,
                )

            log_kv(
                logger,
                "embedding.loaded",
                model=self._model_name,
                device=self._device,
                dim=self._dim,
            )
            return self._model

    @staticmethod
    def _resolve_dimension(model: Any) -> int:
        """读取向量维度, 兼容不同版本的 sentence-transformers.

        sentence-transformers 5.x 把 ``get_sentence_embedding_dimension``
        改名为 ``get_embedding_dimension``. 两个名字都试一遍,
        这样别人 clone 项目时装了 3.x 或 5.x 都能跑.
        """
        for method_name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            getter = getattr(model, method_name, None)
            if callable(getter):
                try:
                    return int(getter())
                except Exception:  # noqa: BLE001, S112 - 换下一个方法名继续尝试
                    continue

        # 两个名字都不存在时, 用一次真实编码兜底探测
        logger.warning("无法通过 API 获取向量维度, 改用试编码探测")
        return int(model.encode(["探测"], normalize_embeddings=True).shape[1])

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """解析设备配置, 无 GPU 时自动降级."""
        if not requested.startswith("cuda"):
            return requested
        try:
            import torch
        except ImportError:
            logger.warning("配置了 %s 但未安装 torch, 回退到 cpu", requested)
            return "cpu"

        if torch.cuda.is_available():
            return requested
        logger.warning("配置了 %s 但 CUDA 不可用, 自动回退到 cpu", requested)
        return "cpu"

    def release(self) -> None:
        """释放模型占用的显存/内存."""
        with self._lock:
            self._model = None
            self._dim = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
        logger.info("已释放本地 Embedding 模型 | model=%s", self._model_name)
