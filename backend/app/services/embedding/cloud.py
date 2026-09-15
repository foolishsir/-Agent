"""云端 Embedding 实现(OpenAI 兼容协议).

存在的意义
----------
本地模型适合"数据不出域 + 无 API 成本"的场景, 云端 API 适合"零部署 + 效果更好"的场景.
两种都要支持, 否则 "EmbeddingProvider 抽象" 就只是一句空话.

注意: 云端模型(``text-embedding-3-*``)是**对称**的, query 与 passage 编码方式相同,
不需要指令前缀. 所以配置里 ``embedding_query_instruction`` 要留空.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger, log_kv

logger = get_logger("docmind.embedding.openai")


class OpenAICompatibleEmbedding:
    """通过 OpenAI 兼容接口调用云端 Embedding."""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str,
        base_url: str | None = None,
        dim: int = 1536,
        batch_size: int = 64,
        timeout: float = 60.0,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._dim = dim
        self._batch_size = batch_size
        self._timeout = timeout
        self._client: Any | None = None

    @property
    def name(self) -> str:
        return f"openai:{self._model_name}"

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise EmbeddingError("未安装 openai 包") from exc
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._ensure_client()
        vectors: list[list[float]] = []

        # 分批: 云端接口对单次请求的条目数有限制, 且大批量失败重试代价高
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            try:
                response = client.embeddings.create(model=self._model_name, input=batch)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(f"云端向量化失败: {exc}") from exc

            # 不依赖返回顺序: 按 index 字段重排, 防止服务端乱序导致向量与文本错位
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([list(item.embedding) for item in ordered])

            log_kv(
                logger,
                "embedding.batch_done",
                model=self._model_name,
                batch=len(batch),
                done=min(start + self._batch_size, len(texts)),
                total=len(texts),
            )

        return vectors

    def encode_query(self, text: str) -> list[float]:
        vectors = self.encode_passages([text])
        return vectors[0] if vectors else []

    def release(self) -> None:
        self._client = None
