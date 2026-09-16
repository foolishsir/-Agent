"""BM25 关键词检索 —— 补齐向量检索对"型号 / 数字 / 专有名词"的短板.

为什么必须有这一路
------------------
Embedding 把文本压成语义向量时会**丢失精确的词面信息**.
在语义空间里 "1333 机种" 和 "1335 机种" 几乎一样近, 但对业务来说是天壤之别.

实测场景: 问 "1333 机种钢刀的更换周期", 纯向量检索召回的全是
"设备维护""刀具管理" 这类语义相关但机种不对的段落.

BM25 基于词频统计, 对这类**精确 token 匹配**天然敏感, 与向量形成互补.

实现要点
--------
1. **中文必须分词**: BM25 是词袋模型, 不分词的话整个句子就是一个 token,
   匹配率极低. 用 jieba 的 ``lcut_for_search`` —— 它会把长词额外切出子词,
   对检索场景的召回率比精确模式更好(例如"中华人民共和国"也会产出"中华""人民").
2. **索引要缓存**: 每次查询都重建 BM25 索引是灾难性的. 但缓存必须能被
   正确失效 —— 文档增删后索引就是脏的, 会检索到已删除的内容.
3. **分数不可跨查询比较**: BM25 的绝对值取决于语料统计, 加一份文档就会整体漂移.
   所以融合时只用**排名**不用分数(见 ``fusion.py``).
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter
from typing import Any

from app.core.logging import get_logger, log_kv
from app.services.retrieval.base import RetrievedChunk

logger = get_logger("docmind.retrieval.bm25")

#: 只保留有意义的 token: 中英文数字, 长度 >= 1
_TOKEN_KEEP_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")

#: 索引缓存有效期(秒). 即使没有显式失效, 也会过期重建 ——
#: 这是防止"漏调用失效函数导致长期脏读"的兜底.
_CACHE_TTL = 300.0


def tokenize(text: str) -> list[str]:
    """中文分词 + 归一化.

    用 ``lcut_for_search`` 而不是 ``lcut``:
    前者会为长词额外产出子词, 检索场景下召回率更高.
    例如 "知识图谱" → ["知识", "图谱", "知识图谱"], 用户只搜"图谱"时也能命中.
    """
    if not text:
        return []

    import jieba  # 惰性导入: jieba 首次加载词典要 1~2 秒, 不该拖慢服务启动

    tokens: list[str] = []
    for raw in jieba.lcut_for_search(text):
        token = raw.strip().lower()
        # 丢掉纯标点和空白分词结果
        if token and _TOKEN_KEEP_RE.fullmatch(token):
            tokens.append(token)
    return tokens


class _BM25Index:
    """一份语料的 BM25 索引快照."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self._ids = [c.id for c in chunks]
        self._bm25: Any | None = None
        self._token_counts: list[Counter[str]] = []

    def build(self) -> None:
        from rank_bm25 import BM25Okapi

        corpus_tokens: list[list[str]] = []
        for chunk in self.chunks:
            tokens = tokenize(chunk.content)
            corpus_tokens.append(tokens)
            self._token_counts.append(Counter(tokens))

        if not corpus_tokens:
            return
        try:
            self._bm25 = BM25Okapi(corpus_tokens)
        except Exception:  # noqa: BLE001 - 语料为空或全为停用词时可能抛异常
            logger.warning("BM25 索引构建失败, 已退化为无关键词检索")
            self._bm25 = None

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._bm25 is None or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # 只保留正分: BM25Okapi 会给完全不相关的文档负分,
        # 把它们带进融合结果会稀释真正的命中项
        ranked = sorted(
            ((idx, float(score)) for idx, score in enumerate(scores) if score > 0),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]

        return [self.chunks[idx].with_score(score, "bm25") for idx, score in ranked]


class BM25Retriever:
    """带缓存的 BM25 检索器."""

    def __init__(self, ttl: float = _CACHE_TTL) -> None:
        self._cache: dict[str, tuple[float, _BM25Index]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl
        # 全局版本号: 文档增删时自增, 用来让缓存键失效.
        # 用"键里带版本号"而不是"遍历删除", 并发下更安全.
        self._version = 0

    def invalidate(self) -> None:
        """文档内容变化后调用(入库完成 / 删除文档 / 重新处理).

        不调用的话会检索到**已删除文档**的内容 —— 这正是"幽灵数据"问题
        在检索层的表现, 比向量库那边的残留更隐蔽, 因为它不出现在任何存储里.
        """
        with self._lock:
            self._version += 1
            self._cache.clear()
        logger.debug("BM25 索引缓存已失效 | version=%s", self._version)

    def search(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int,
        doc_ids: list[str] | None = None,
        loader: Any = None,
    ) -> list[RetrievedChunk]:
        """执行关键词检索.

        Args:
            loader: 无参可调用对象, 返回该用户/文档范围内的全部分块.
                    由调用方注入, 避免检索层直接依赖向量库实现.
        """
        if top_k <= 0 or loader is None:
            return []

        started = time.perf_counter()
        index = self._get_index(user_id=user_id, doc_ids=doc_ids, loader=loader)

        if index is None or not index.chunks:
            return []

        hits = index.search(query, top_k)
        log_kv(
            logger,
            "bm25.search",
            query_len=len(query),
            corpus=len(index.chunks),
            hits=len(hits),
            cost_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return hits

    def _get_index(
        self, *, user_id: str, doc_ids: list[str] | None, loader: Any
    ) -> _BM25Index | None:
        key = f"v{self._version}:{user_id}:{','.join(sorted(doc_ids)) if doc_ids else '*'}"

        with self._lock:
            cached = self._cache.get(key)
            if cached and (time.time() - cached[0]) < self._ttl:
                return cached[1]

        # 构建索引放在锁外 —— 它可能耗时几百毫秒, 占着锁会把所有并发查询堵死
        chunks = loader()
        index = _BM25Index(list(chunks))
        index.build()

        with self._lock:
            self._cache[key] = (time.time(), index)
            # 简单清理: 缓存项过多时丢掉最旧的, 防止内存无限增长
            if len(self._cache) > 16:
                oldest = min(self._cache, key=lambda k: self._cache[k][0])
                self._cache.pop(oldest, None)

        log_kv(
            logger,
            "bm25.index_built",
            user_id=user_id,
            chunks=len(index.chunks),
            avg_tokens=round(
                sum(len(c) for c in index._token_counts) / max(len(index._token_counts), 1), 1
            ),
        )
        return index

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": self._version,
                "cached_indexes": len(self._cache),
                "ttl_seconds": self._ttl,
            }


def _token_entropy(tokens: list[str]) -> float:
    """分词结果的词频熵 —— 仅用于调试观察分词质量."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((c / total) * math.log(c / total) for c in counts.values())


#: 全局单例
_retriever: BM25Retriever | None = None


def get_bm25_retriever() -> BM25Retriever:
    global _retriever
    if _retriever is None:
        _retriever = BM25Retriever()
    return _retriever
