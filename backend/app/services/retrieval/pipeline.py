"""检索编排: 向量召回 + BM25 召回 → RRF 融合 → 父块去重 → 精排 → 父块回查.

完整链路
--------
::

    query
      ├─ 向量检索 (Chroma, cosine) ──┐
      └─ BM25 关键词检索 (jieba) ────┤
                                     ▼
                              RRF 融合(只用排名)
                                     ▼
                            按父块去重(取最高分)
                                     ▼
                        Cross-Encoder 精排 → Top-N
                                     ▼
                        父块回查(SQL) → 送进 Prompt 的上下文

两个关键设计点
--------------
1. **精排打分打在子块上, 但送给 LLM 的是父块**.
   子块是实际被匹配的粒度, 用它打分才能准确判断"这个匹配有多强";
   父块是生成用的上下文单位, 用它打分会混入无关内容.
   精排分数与拒答阈值因此处在同一尺度上, 可以直接比较.

2. **父块去重要在精排之前做**.
   否则同一个父块的多个子块会占满精排序位, 精排就白做了 ——
   最终 Top-5 可能全来自同一段内容, 上下文多样性归零.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger, log_kv
from app.models.document import Chunk as ChunkModel
from app.models.document import ChunkType
from app.services.embedding import get_embedding_provider
from app.services.retrieval.base import RetrievalTrace, RetrievedChunk
from app.services.retrieval.bm25 import get_bm25_retriever
from app.services.retrieval.fusion import dedupe_by_parent, reciprocal_rank_fusion
from app.services.retrieval.reranker import get_reranker
from app.services.vectorstore import SearchFilter, get_vector_store

logger = get_logger("docmind.retrieval")


@dataclass
class RetrievedContext:
    """最终送进 Prompt 的一段上下文(父块粒度)."""

    index: int  # 引用编号, 从 1 开始
    parent_id: str
    doc_id: str
    filename: str
    content: str
    page_start: int
    page_end: int
    section_path: str
    score: float  # 精排分数
    matched_child_id: str | None = None
    matched_child_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_path": self.section_path,
            "score": round(self.score, 4),
            "content": self.content,
            # 给前端做引用卡片的高亮片段
            "snippet": _make_snippet(self.content),
        }


@dataclass
class RetrievalDebug:
    """检索链路的**分阶段候选集**, 用于评测与线上排查.

    为什么需要它: 用户问"为什么答案不对"时, 有三种完全不同的原因 ——

    1. 召回阶段就没找到相关内容(向量和 BM25 都没命中)
    2. 召回到了但排序太靠后(被 RRF 或精排压下去了)
    3. 召回到了也排前面了, 但 LLM 没用对

    只看最终结果无法区分这三者, 而它们的优化方向完全相反:
    第 1 种要改分块或换 embedding 模型; 第 2 种要调融合/精排; 第 3 种要改 Prompt.

    所以把每一阶段的候选集都留下来. 默认不采集(有内存开销), 由调用方按需开启.
    """

    vector_hits: list[RetrievedChunk] = field(default_factory=list)
    bm25_hits: list[RetrievedChunk] = field(default_factory=list)
    fused: list[RetrievedChunk] = field(default_factory=list)
    deduped: list[RetrievedChunk] = field(default_factory=list)
    reranked: list[RetrievedChunk] = field(default_factory=list)

    def stage(self, name: str) -> list[RetrievedChunk]:
        return {
            "vector": self.vector_hits,
            "bm25": self.bm25_hits,
            "fused": self.fused,
            "deduped": self.deduped,
            "reranked": self.reranked,
        }.get(name, [])

    def sizes(self) -> dict[str, int]:
        return {
            "vector": len(self.vector_hits),
            "bm25": len(self.bm25_hits),
            "fused": len(self.fused),
            "deduped": len(self.deduped),
            "reranked": len(self.reranked),
        }


@dataclass
class RetrievalResult:
    """一次检索的完整产出."""

    contexts: list[RetrievedContext] = field(default_factory=list)
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)
    #: 最高精排分. 用于判断"文档里到底有没有料"
    top_score: float = 0.0
    #: 是否判定为"文档中无相关内容"
    refused: bool = False
    refuse_reason: str = ""
    #: 分阶段候选集(默认不采集)
    debug: RetrievalDebug | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contexts": [c.to_dict() for c in self.contexts],
            "trace": self.trace.to_dict(),
            "top_score": round(self.top_score, 4),
            "refused": self.refused,
            "refuse_reason": self.refuse_reason,
        }


async def retrieve(
    session: AsyncSession,
    query: str,
    *,
    user_id: str,
    doc_ids: list[str],
    debug: bool = False,
) -> RetrievalResult:
    """执行完整检索链路.

    Args:
        debug: 采集分阶段候选集(见 ``RetrievalDebug``).
               评测时必须开启, 否则候选被截断到 ``final_top_k``,
               Recall@10 这类指标算不出来. 线上排查问题也可以临时打开,
               代价是每阶段多留一份引用(不是深拷贝, 开销很小).
    """
    result = RetrievalResult()

    if not doc_ids:
        result.refused = True
        result.refuse_reason = "没有可用于检索的文档, 请先上传并等待处理完成"
        return result

    store = get_vector_store()

    # ---------------- ① Query 向量化 ----------------
    started = time.perf_counter()
    provider = get_embedding_provider()
    query_vector = await asyncio.to_thread(provider.encode_query, query)
    result.trace.add("embed_query", count=1, cost_ms=(time.perf_counter() - started) * 1000)

    # ---------------- ② 双路召回 (并发) ----------------
    vector_task = _vector_search(store, query_vector, user_id=user_id, doc_ids=doc_ids)
    bm25_task = _bm25_search(user_id=user_id, doc_ids=doc_ids, query=query)
    (vector_hits, vector_ms), (bm25_hits, bm25_ms) = await asyncio.gather(vector_task, bm25_task)

    result.trace.add("vector_recall", count=len(vector_hits), cost_ms=vector_ms)
    result.trace.add("bm25_recall", count=len(bm25_hits), cost_ms=bm25_ms)

    if debug:
        result.debug = RetrievalDebug(vector_hits=vector_hits, bm25_hits=bm25_hits)

    if not vector_hits and not bm25_hits:
        result.refused = True
        result.refuse_reason = "未检索到任何相关内容"
        return result

    # ---------------- ③ RRF 融合 ----------------
    started = time.perf_counter()
    fused = reciprocal_rank_fusion(
        [vector_hits, bm25_hits] if bm25_hits else [vector_hits],
        k=settings.rrf_k,
    )
    result.trace.add("rrf_fusion", count=len(fused), cost_ms=(time.perf_counter() - started) * 1000)

    # ---------------- ④ 父块去重 ----------------
    # 必须在精排之前: 否则同一父块的多个子块会占满精排序位, 上下文多样性归零
    deduped = dedupe_by_parent(fused)
    result.trace.add("dedupe_parent", count=len(deduped), cost_ms=0.0)

    if debug and result.debug is not None:
        result.debug.fused = fused
        result.debug.deduped = deduped

    # ---------------- ⑤ 精排 ----------------
    # 注意: 这里只取 final_top_k 条. 评测想要 Recall@10 时这个截断会挡住,
    # 所以 debug 模式下调大取样数(评测才有意义).
    top_n = settings.final_top_k
    started = time.perf_counter()
    reranker = get_reranker()
    if reranker is not None and deduped:
        candidates = await asyncio.to_thread(
            reranker.rerank, query, deduped, top_n=(top_n if not debug else len(deduped))
        )
        result.trace.add(
            "rerank",
            count=len(candidates),
            cost_ms=(time.perf_counter() - started) * 1000,
            enabled=True,
        )
    else:
        # 未启用精排时退化为按融合分数截断.
        # 这样即使关掉重排, 链路依然可用(只是精度下降), 便于做 A/B 对比实验.
        candidates = sorted(deduped, key=lambda c: c.score, reverse=True)
        if not debug:
            candidates = candidates[:top_n]
        result.trace.add("rerank", count=len(candidates), cost_ms=0.0, enabled=False)

    if debug and result.debug is not None:
        result.debug.reranked = candidates

    # 评测模式下候选集已经完整保留, 但下游生成仍然只该看到 final_top_k 条 ——
    # 否则评测出来的上下文长度和线上不一致, 指标就没有参考价值了.
    if debug:
        candidates = candidates[: settings.final_top_k]

    if not candidates:
        result.refused = True
        result.refuse_reason = "未检索到任何相关内容"
        return result

    result.top_score = candidates[0].score

    # ---------------- ⑥ 拒答判定 ----------------
    if settings.rerank_min_score > 0 and result.top_score < settings.rerank_min_score:
        result.refused = True
        result.refuse_reason = (
            f"精排最高分 {result.top_score:.3f} 低于阈值 {settings.rerank_min_score}, "
            "判定文档中无相关内容"
        )
        log_kv(
            logger,
            "retrieval.refused",
            top_score=round(result.top_score, 4),
            threshold=settings.rerank_min_score,
        )
        return result

    # ---------------- ⑦ 回查父块 ----------------
    started = time.perf_counter()
    result.contexts = await _load_parents(session, candidates)
    result.trace.add(
        "load_parents", count=len(result.contexts), cost_ms=(time.perf_counter() - started) * 1000
    )

    log_kv(
        logger,
        "retrieval.done",
        query_len=len(query),
        vector=len(vector_hits),
        bm25=len(bm25_hits),
        fused=len(fused),
        final=len(result.contexts),
        top_score=round(result.top_score, 4),
        total_ms=result.trace.total_ms,
    )
    return result


# --------------------------------------------------------------------------- #
# 各路召回
# --------------------------------------------------------------------------- #
async def _vector_search(
    store: Any, query_vector: list[float], *, user_id: str, doc_ids: list[str]
) -> tuple[list[RetrievedChunk], float]:
    filter_ = SearchFilter(
        user_id=user_id,
        doc_ids=doc_ids,
        # 只检索子块 —— 父块不参与向量检索, 这是父子块架构的前提
        chunk_types=[ChunkType.CHILD.value],
    )
    started = time.perf_counter()
    hits = await asyncio.to_thread(
        store.query, query_vector, top_k=settings.vector_top_k, filters=filter_
    )
    cost_ms = (time.perf_counter() - started) * 1000

    return [
        RetrievedChunk(
            id=hit.id, content=hit.document, metadata=hit.metadata, score=hit.score, source="vector"
        )
        for hit in hits
    ], cost_ms


async def _bm25_search(
    *, user_id: str, doc_ids: list[str], query: str
) -> tuple[list[RetrievedChunk], float]:
    if settings.bm25_top_k <= 0:
        return [], 0.0

    store = get_vector_store()
    retriever = get_bm25_retriever()

    def loader() -> list[RetrievedChunk]:
        """从向量库拉取该用户范围内的全部子块作为 BM25 语料."""
        chunks: list[RetrievedChunk] = []
        for doc_id in doc_ids:
            for hit in store.list_by_doc(doc_id):
                if hit.metadata.get("chunk_type") != ChunkType.CHILD.value:
                    continue
                chunks.append(
                    RetrievedChunk(
                        id=hit.id,
                        content=hit.document,
                        metadata=hit.metadata,
                        score=0.0,
                        source="bm25",
                    )
                )
        return chunks

    started = time.perf_counter()
    hits = await asyncio.to_thread(
        retriever.search,
        query,
        user_id=user_id,
        top_k=settings.bm25_top_k,
        doc_ids=doc_ids,
        loader=loader,
    )
    return hits, (time.perf_counter() - started) * 1000


# --------------------------------------------------------------------------- #
# 父块回查
# --------------------------------------------------------------------------- #
async def _load_parents(
    session: AsyncSession, candidates: list[RetrievedChunk]
) -> list[RetrievedContext]:
    """按 parent_id 从关系库取回父块正文.

    父块只存在关系库里, 不在向量库 —— 它们不需要 embedding,
    放进向量库只会浪费空间和一次 embedding 计算.
    """
    parent_ids = [c.parent_id for c in candidates if c.parent_id]
    if not parent_ids:
        return []

    rows = (
        (await session.execute(select(ChunkModel).where(ChunkModel.id.in_(parent_ids))))
        .scalars()
        .all()
    )
    parents = {row.id: row for row in rows}

    contexts: list[RetrievedContext] = []
    for candidate in candidates:
        parent = parents.get(candidate.parent_id or "")
        if parent is None:
            # 父块缺失(理论上不该发生) → 退回用子块内容, 保证链路不断
            logger.warning("父块缺失, 回退使用子块 | parent_id=%s", candidate.parent_id)
            contexts.append(
                RetrievedContext(
                    index=len(contexts) + 1,
                    parent_id=candidate.parent_id or candidate.id,
                    doc_id=candidate.doc_id,
                    filename=candidate.filename,
                    content=candidate.content,
                    page_start=candidate.page_start,
                    page_end=candidate.page_end,
                    section_path=candidate.section_path,
                    score=candidate.score,
                    matched_child_id=candidate.id,
                    matched_child_text=candidate.content,
                )
            )
            continue

        contexts.append(
            RetrievedContext(
                index=len(contexts) + 1,
                parent_id=parent.id,
                doc_id=parent.doc_id,
                filename=candidate.filename,
                content=parent.content,
                page_start=parent.page_start,
                page_end=parent.page_end,
                section_path=parent.section_path or "",
                score=candidate.score,
                matched_child_id=candidate.id,
                matched_child_text=candidate.content,
            )
        )

    return contexts


def _make_snippet(content: str, limit: int = 120) -> str:
    """截取用于引用卡片展示的片段."""
    text = " ".join(content.split())
    return text[:limit] + ("…" if len(text) > limit else "")
