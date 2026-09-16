"""RRF(Reciprocal Rank Fusion)融合.

公式
----
::

    RRF(d) = Σ_i  1 / (k + rank_i(d))

``k`` 默认取 60(原论文经验值), ``rank`` 从 1 开始计数.

为什么用 RRF 而不是加权求和
---------------------------
向量的余弦相似度在 0~1 之间, BM25 的分数取决于语料统计(可能 0~20, 也可能 0~200),
**两者量纲完全不同, 直接加权求和是没有意义的**.

要做加权求和就必须先归一化, 而归一化的方式本身就是个坑:

- min-max 归一化对离群值极度敏感, 一条异常高分就能把所有其他分数压到 0 附近
- z-score 归一化假设分布近似正态, 而检索分数分布通常长尾
- 而且归一化参数**每个查询都不同**, 导致同一个文档在不同查询下的贡献不可比

RRF 只用**排名**不用分数, 因此:

1. 对两路分数量纲完全不敏感
2. 天然抗离群值(第一名永远是第一名, 分多高都一样)
3. 只需调一个参数 ``k``, 且鲁棒性很好

k 的作用
--------
- ``k`` 越小 → 头部排名权重越极端(第 1 名 1/1=1.0, 第 2 名 1/2=0.5)
- ``k`` 越大 → 各排名权重越平均

工程建议: **不要在这个参数上花太多时间**. 先调分块策略和重排的收益要大得多.
"""

from __future__ import annotations

from app.services.retrieval.base import RetrievedChunk


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    """融合多路检索结果.

    Args:
        result_lists: 多路已排序的检索结果(各自按自己的相关性从高到低)
        k: RRF 常数, 默认 60
        weights: 每路的权重, 默认等权. 若某路被认为更可靠可以调高.

    Returns:
        融合后的结果, 按融合分数从高到低排序, 且**已按 chunk id 去重**.
        同一条内容被多路同时召回时分数会叠加 —— 这正是融合的价值:
        "两路都认为是相关的"应该比"只有一路认为是相关的"排得更前.
    """
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    if len(weights) != len(result_lists):
        raise ValueError("weights 长度必须与 result_lists 一致")

    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    #: 记录每条内容被哪几路召回, 用于调试"为什么它排这么前"
    sources: dict[str, list[str]] = {}

    for results, weight in zip(result_lists, weights, strict=True):
        for rank, item in enumerate(results, start=1):
            scores[item.id] = scores.get(item.id, 0.0) + weight / (k + rank)
            sources.setdefault(item.id, []).append(f"{item.source}#{rank}")
            # 保留首次出现的版本(内容相同, 只是来源不同)
            best.setdefault(item.id, item)

    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)

    fused: list[RetrievedChunk] = []
    for chunk_id, score in ordered:
        item = best[chunk_id].with_score(score, "fused")
        item.metadata["_rrf_sources"] = sources.get(chunk_id, [])
        fused.append(item)

    return fused


def dedupe_by_parent(
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """按父块去重, 保留分数最高的那个子块代表该父块.

    这是父子块架构在检索侧的收尾动作:
    一个父块可能被多个子块命中(甚至被两路同时命中),
    但送给 LLM 的上下文里**同一个父块只能出现一次**, 否则白白浪费 token.

    保留最高分而不是求平均: 我们的目标是"这个父块值不值得进 Prompt",
    只要它有一个子块高度相关就够了. 求平均会让"一个强命中 + 三个弱命中"
    的父块输给"三个中等命中"的父块, 这不符合直觉.
    """
    best_by_parent: dict[str, RetrievedChunk] = {}
    order: list[str] = []

    for chunk in chunks:
        key = chunk.parent_id or chunk.id
        current = best_by_parent.get(key)
        if current is None:
            best_by_parent[key] = chunk
            order.append(key)
        elif chunk.score > current.score:
            best_by_parent[key] = chunk

    return [best_by_parent[key] for key in order]
