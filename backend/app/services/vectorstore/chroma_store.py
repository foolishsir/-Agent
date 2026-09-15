"""Chroma 向量库实现.

选型理由与取舍见 ``docs/02-技术选型与权衡.md``. 这里只记实现要点:

1. **距离度量固定为 cosine**. 建集合时通过 ``hnsw:space`` 指定.
   默认的 L2 距离对归一化向量等价, 但对未归一化的向量不等价 ——
   显式声明比依赖默认值安全.
2. **距离换算成相似度**. Chroma 返回的是距离(越小越近). 上层如果还要
   记住"这个库是越小越好、那个库是越大越好", 迟早写出反向排序的 bug.
   在适配层统一换成"越大越相关".
3. **批量分片写入**. 一次性 upsert 上万条会撑爆请求体;
   按固定大小分片, 也让失败重试的粒度更细.
4. **单例客户端**. Chroma 的 PersistentClient 在同一进程内重复用不同参数创建会告警,
   而且各自持有一份索引, 数据不互通.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger, log_kv
from app.services.vectorstore.base import SearchFilter, SearchHit, VectorRecord

logger = get_logger("docmind.vectorstore.chroma")

#: 单次写入的分片大小
_UPSERT_CHUNK = 512
#: get() 分页大小 —— Chroma 单次返回有上限, 大文档必须翻页取全
_PAGE_SIZE = 500


class ChromaVectorStore:
    """基于 Chroma 的向量库实现."""

    def __init__(
        self,
        *,
        collection_name: str,
        mode: str = "embedded",
        persist_dir: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._mode = mode
        self._persist_dir = persist_dir
        self._host = host
        self._port = port
        self._client: Any | None = None
        self._collection: Any | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        if self._mode == "http":
            return f"chroma-http:{self._host}:{self._port}/{self._collection_name}"
        return f"chroma-embedded:{self._persist_dir}/{self._collection_name}"

    def _ensure_collection(self) -> Any:
        if self._collection is not None:
            return self._collection

        with self._lock:
            if self._collection is not None:
                return self._collection

            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings
            except ImportError as exc:  # pragma: no cover
                raise VectorStoreError("未安装 chromadb, 请执行: pip install chromadb") from exc

            try:
                if self._mode == "http":
                    client = chromadb.HttpClient(host=self._host, port=self._port)
                else:
                    client = chromadb.PersistentClient(
                        path=self._persist_dir or "./data/chroma",
                        settings=ChromaSettings(anonymized_telemetry=False),
                    )
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"连接向量库失败: {exc}") from exc

            self._client = client
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                # cosine 是文本检索的默认选择: embedding 已归一化时,
                # 余弦相似度只关心方向不关心模长, 对文本长度差异更鲁棒
                metadata={"hnsw:space": "cosine"},
            )
            log_kv(
                logger,
                "vectorstore.ready",
                store=self.name,
                count=self._collection.count(),
            )
            return self._collection

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0

        collection = self._ensure_collection()
        written = 0

        for start in range(0, len(records), _UPSERT_CHUNK):
            batch = list(records[start : start + _UPSERT_CHUNK])
            try:
                collection.upsert(
                    ids=[r.id for r in batch],
                    embeddings=[r.embedding for r in batch],
                    documents=[r.document for r in batch],
                    metadatas=[r.clean_metadata() for r in batch],
                )
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(
                    f"写入向量库失败(第 {start}~{start + len(batch)} 条): {exc}"
                ) from exc
            written += len(batch)

        log_kv(logger, "vectorstore.upsert", count=written, store=self.name)
        return written

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def query(
        self,
        embedding: list[float],
        *,
        top_k: int = 20,
        filters: SearchFilter | None = None,
    ) -> list[SearchHit]:
        if not embedding:
            return []

        collection = self._ensure_collection()
        where = filters.to_where() if filters else None

        # 空 doc_ids 会生成一个永不满足的条件, 这里直接短路, 省一次无意义的查询
        if where is not None and where.get("doc_id", {}).get("$eq") == "__none__":
            return []

        try:
            raw = collection.query(
                query_embeddings=[embedding],
                n_results=max(1, top_k),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"向量检索失败: {exc}") from exc

        return _hits_from_query(raw)

    def get_by_ids(self, ids: Sequence[str]) -> list[SearchHit]:
        if not ids:
            return []
        collection = self._ensure_collection()
        try:
            raw = collection.get(
                ids=list(ids),
                include=["documents", "metadatas"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"按 id 取回分块失败: {exc}") from exc
        return _hits_from_get(raw)

    def list_by_doc(self, doc_id: str) -> list[SearchHit]:
        """取回一份文档的全部分块.

        必须**翻页**: Chroma 的单次返回有条数上限, 直接一次 get 会静默截断,
        表现为"大文档的 BM25 语料不全", 检索时部分内容永远搜不到.
        """
        collection = self._ensure_collection()
        hits: list[SearchHit] = []
        offset = 0

        while True:
            try:
                raw = collection.get(
                    where={"doc_id": {"$eq": doc_id}},
                    include=["documents", "metadatas"],
                    limit=_PAGE_SIZE,
                    offset=offset,
                )
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"按文档取回分块失败: {exc}") from exc

            page = _hits_from_get(raw)
            hits.extend(page)
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE

        return hits

    # ------------------------------------------------------------------ #
    # 删除与统计
    # ------------------------------------------------------------------ #
    def delete_by_doc(self, doc_id: str) -> int:
        collection = self._ensure_collection()
        before = self.count(SearchFilter(doc_ids=[doc_id]))
        try:
            collection.delete(where={"doc_id": {"$eq": doc_id}})
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"删除文档向量失败: {exc}") from exc
        log_kv(logger, "vectorstore.delete", doc_id=doc_id, deleted=before)
        return before

    def count(self, filters: SearchFilter | None = None) -> int:
        collection = self._ensure_collection()
        where = filters.to_where() if filters else None
        try:
            if where is None:
                return int(collection.count())
            return len(collection.get(where=where, include=[])["ids"])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"统计向量条数失败: {exc}") from exc

    def health(self) -> tuple[bool, str]:
        try:
            count = self._ensure_collection().count()
            return True, f"{self.name} (count={count})"
        except Exception as exc:  # noqa: BLE001 - 探针需要兜住一切
            return False, f"{self.name} 不可用: {exc}"


# --------------------------------------------------------------------------- #
# 结果转换
# --------------------------------------------------------------------------- #
def _hits_from_query(raw: dict[str, Any]) -> list[SearchHit]:
    """把 Chroma query 的嵌套返回结构拍平成 SearchHit 列表.

    Chroma 的返回是"批量查询"结构: 即使只查一个向量, 结果也在
    ``ids[0]`` / ``documents[0]`` 这种二层嵌套里. 在这里拍平,
    上层就不用到处写 ``[0]``.
    """
    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    hits: list[SearchHit] = []
    for index, chunk_id in enumerate(ids):
        distance = float(distances[index]) if index < len(distances) else 1.0
        hits.append(
            SearchHit(
                id=str(chunk_id),
                document=str(documents[index]) if index < len(documents) else "",
                metadata=dict(metadatas[index]) if index < len(metadatas) else {},
                # cosine 距离 = 1 - 余弦相似度 → 相似度 = 1 - 距离
                score=1.0 - distance,
            )
        )
    return hits


def _hits_from_get(raw: dict[str, Any]) -> list[SearchHit]:
    """把 Chroma get 的返回结构转成 SearchHit 列表(无分数)."""
    ids = raw.get("ids") or []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    hits: list[SearchHit] = []
    for index, chunk_id in enumerate(ids):
        hits.append(
            SearchHit(
                id=str(chunk_id),
                document=str(documents[index]) if index < len(documents) else "",
                metadata=dict(metadatas[index]) if index < len(metadatas) else {},
                score=0.0,  # get 不计算相似度
            )
        )
    return hits


# --------------------------------------------------------------------------- #
# 单例
# --------------------------------------------------------------------------- #
_store: ChromaVectorStore | None = None
_store_lock = threading.Lock()


def get_vector_store() -> ChromaVectorStore:
    """获取全局单例向量库客户端."""
    global _store
    if _store is not None:
        return _store

    with _store_lock:
        if _store is not None:
            return _store
        _store = ChromaVectorStore(
            collection_name=settings.chroma_collection,
            mode=settings.chroma_mode,
            persist_dir=str(settings.chroma_persist_dir),
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        return _store


def reset_vector_store_for_test() -> None:
    """测试用: 清空单例."""
    global _store
    with _store_lock:
        _store = None
