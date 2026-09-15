"""向量库抽象.

为什么要做这层抽象
------------------
Chroma 的 embedded 模式把索引放在**应用进程内**: 零部署成本, 但无法多实例共享.
数据量上来或需要水平扩容时, 就必须换成 Chroma Server / Milvus / pgvector.

如果业务代码直接调 ``chromadb`` 的 API, 换库就是全量重构.
抽象成 ``VectorStore`` 协议后, 换库只需要写一个新的实现类 + 改一行配置.

**过滤条件下沉**
----------------
``query()`` 的 ``where`` 参数不是可选项而是核心能力.
多用户隔离必须靠它实现, 理由见 ``SearchFilter`` 的注释.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """一条待写入的向量记录."""

    id: str
    embedding: list[float]
    document: str
    metadata: dict[str, Any]

    def clean_metadata(self) -> dict[str, Any]:
        """去掉 None 值.

        Chroma 的 metadata 不接受 ``None`` —— 传进去会直接抛异常.
        而我们的 ``section_path`` 在文档开头(还没有标题)时确实是 None.
        在写入边界统一清洗, 比在每个调用点做判断更可靠.
        """
        return {k: v for k, v in self.metadata.items() if v is not None}


@dataclass(frozen=True, slots=True)
class SearchHit:
    """一条检索结果."""

    id: str
    document: str
    metadata: dict[str, Any]
    #: 相似度, **越大越相关**(已从距离换算). 统一方向避免上层写反排序.
    score: float

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id", ""))

    @property
    def parent_id(self) -> str | None:
        value = self.metadata.get("parent_id")
        return str(value) if value else None

    @property
    def page_start(self) -> int:
        return int(self.metadata.get("page_start", 1))

    @property
    def page_end(self) -> int:
        return int(self.metadata.get("page_end", self.page_start))


@dataclass(frozen=True, slots=True)
class SearchFilter:
    """检索过滤条件.

    **为什么过滤条件必须下沉到向量库, 而不是取回结果后在应用层筛**:

    假设某用户只上传了 1 份文档, 而系统里共有 100 份. 如果先召回 Top-20
    再在应用层过滤掉别人的, 那么这个用户实际只剩不到 1 条结果 —— 召回质量直接崩塌.

    正确做法是在**候选生成阶段**就把不相关的数据排除掉, 而不是事后筛选.
    这也是为什么 ``where`` 是 ``query()`` 的必选能力而非可选优化.
    """

    user_id: str | None = None
    doc_ids: list[str] | None = None
    #: 只检索这些分块类型(默认只看子块 —— 父块不参与向量检索)
    chunk_types: list[str] | None = None

    def to_where(self) -> dict[str, Any] | None:
        """转换成 Chroma 的 where 语法; 无条件时返回 None."""
        clauses: list[dict[str, Any]] = []

        if self.user_id:
            clauses.append({"user_id": {"$eq": self.user_id}})
        if self.doc_ids is not None:
            if not self.doc_ids:
                # 空列表表示"不允许匹配任何文档".
                # 直接返回不可满足的条件, 而不是 None(那等于不限制).
                return {"doc_id": {"$eq": "__none__"}}
            clauses.append({"doc_id": {"$in": list(self.doc_ids)}})
        if self.chunk_types:
            clauses.append({"chunk_type": {"$in": list(self.chunk_types)}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}


@runtime_checkable
class VectorStore(Protocol):
    """向量库接口."""

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        """批量写入(存在则覆盖). 返回写入条数.

        必须是 **upsert 语义**: chunk id 是确定性生成的, 重复处理同一份文档时
        靠覆盖保证幂等, 而不是产生重复数据.
        """
        ...

    def query(
        self,
        embedding: list[float],
        *,
        top_k: int = 20,
        filters: SearchFilter | None = None,
    ) -> list[SearchHit]:
        """向量相似度检索."""
        ...

    def get_by_ids(self, ids: Sequence[str]) -> list[SearchHit]:
        """按 id 批量取回(父子块回查用)."""
        ...

    def list_by_doc(self, doc_id: str) -> list[SearchHit]:
        """取回某份文档的全部分块(BM25 语料构建用)."""
        ...

    def delete_by_doc(self, doc_id: str) -> int:
        """删除某份文档的全部分块, 返回删除条数."""
        ...

    def count(self, filters: SearchFilter | None = None) -> int:
        """统计条数."""
        ...

    def health(self) -> tuple[bool, str]:
        """健康检查."""
        ...
