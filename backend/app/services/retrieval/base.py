"""检索层的数据结构与通用逻辑."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedChunk:
    """一个检索候选项.

    ``score`` 的含义**取决于来源**(向量相似度 / BM25 分 / RRF 融合分 / 精排分),
    所以同时用 ``source`` 显式标注阶段 —— 只看 score 数值无法判断它能不能跨阶段比较,
    这是检索调试里最容易搞混的地方.
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    source: str = "unknown"

    # ------------------------------ 便捷访问 ------------------------------ #
    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id", ""))

    @property
    def parent_id(self) -> str | None:
        value = self.metadata.get("parent_id")
        return str(value) if value else None

    @property
    def filename(self) -> str:
        return str(self.metadata.get("filename", "未知文档"))

    @property
    def page_start(self) -> int:
        return int(self.metadata.get("page_start", 1) or 1)

    @property
    def page_end(self) -> int:
        return int(self.metadata.get("page_end", self.page_start) or self.page_start)

    @property
    def section_path(self) -> str:
        return str(self.metadata.get("section_path", "") or "")

    def with_score(self, score: float, source: str) -> RetrievedChunk:
        """派生一个改了分数的新实例(frozen 语义靠不修改原对象实现)."""
        return RetrievedChunk(
            id=self.id,
            content=self.content,
            metadata=self.metadata,
            score=score,
            source=source,
        )


@dataclass(slots=True)
class RetrievalTrace:
    """检索过程的分阶段耗时与候选数.

    为什么要专门记录: 用户问"为什么答案不对"时, 你需要能回答
    "是召回阶段就没找到, 还是找到了但排序太靠后, 还是 LLM 没用对".
    没有分阶段数据, 这三者无法区分.
    """

    stages: list[dict[str, Any]] = field(default_factory=list)

    def add(self, stage: str, *, count: int, cost_ms: float, **extra: Any) -> None:
        self.stages.append({"stage": stage, "count": count, "cost_ms": round(cost_ms, 2), **extra})

    @property
    def total_ms(self) -> float:
        return round(sum(s["cost_ms"] for s in self.stages), 2)

    def to_dict(self) -> dict[str, Any]:
        return {"stages": self.stages, "total_ms": self.total_ms}
