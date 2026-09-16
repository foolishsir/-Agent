"""分块层的数据结构.

为什么 chunk id 要**确定性生成**
--------------------------------
``{doc_id}_p0003_c002`` 这样的 id 只由"文档 + 位置"决定, 与处理时间无关.
带来的好处是**整个入库流程天然幂等**:

- 同一份文档重复处理 → 生成完全相同的 id → 向量库 upsert 覆盖而不是新增
- 处理到一半失败后重试 → 已写入的部分被覆盖, 不会产生重复数据

如果用随机 UUID, 每次重试都会在向量库里堆一份新数据, 而"脏数据"比"没数据"更难排查.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.models.document import ChunkType

# 分块 id 的前缀/序号宽度. 固定宽度保证 id 长度一致, 也让按 id 排序等价于按顺序排序.
_PARENT_PAD = 4
_CHILD_PAD = 3


def build_embedding_text(content: str, section_path: str | None) -> str:
    """构造实际送去计算向量的文本: 在正文前拼接章节路径.

    **与存库的 content 故意不同**. 原因: 一个孤立的子块常常缺少语境 ——
    正文只有 "更换周期为 20000 次或 3 个月", 单独看不知道说的是什么.
    拼上章节路径变成 "第三章 设备维护 > 3.2 钢刀 > 更换周期为 20000 次或 3 个月",
    向量就能正确落在"设备维护"的语义空间里.

    这是低成本的检索质量提升: 不改模型、不改分块, 只是给文本补语境.
    存库的 content 保持干净, 是为了让引用展示和 BM25 检索用原文.

    **去重**: 父块的第一个子块通常已经包含了章节标题本身(标题段与正文同属一个父块),
    此时再拼一遍会产生 "标题 > 标题 + 正文" 的重复文本, 白占 token 还稀释向量.

    抽成模块级函数而不是留在 ``Chunk`` 的 property 里, 是因为
    **读取已落库分块的接口也必须用同一份逻辑** ——
    否则界面上显示的"送入向量的文本"与实际入库时用的不一致,
    用户会基于错误信息做调参决策. 这是本项目真实踩过的坑.
    """
    if not section_path:
        return content

    leaf = section_path.split(" > ")[-1].strip()
    if leaf and content.lstrip().startswith(leaf):
        return content
    return f"{section_path} > {content}"


@dataclass(frozen=True, slots=True)
class Chunk:
    """一个分块(父块或子块)."""

    id: str
    doc_id: str
    parent_id: str | None
    chunk_type: ChunkType
    content: str
    page_start: int
    page_end: int
    section_path: str | None
    order_index: int

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def embedding_text(self) -> str:
        """实际送去计算向量的文本(见 ``build_embedding_text``)."""
        return build_embedding_text(self.content, self.section_path)


@dataclass
class ChunkingResult:
    """一次分块的完整产出."""

    parents: list[Chunk]
    children: list[Chunk]

    @property
    def total(self) -> int:
        return len(self.parents) + len(self.children)


def make_parent_id(doc_id: str, index: int) -> str:
    return f"{doc_id}_p{index:0{_PARENT_PAD}d}"


def make_child_id(doc_id: str, parent_index: int, child_index: int) -> str:
    return f"{doc_id}_p{parent_index:0{_PARENT_PAD}d}_c{child_index:0{_CHILD_PAD}d}"


def coalesce_short_chunks(chunks: list[Chunk], min_size: int) -> list[Chunk]:
    """把过短的块并入相邻块.

    为什么必须处理: 段落末尾常会剩下一两个短句(如 "已验证。"), 单独成块后
    会变成一条 5~15 字的向量记录. 这种块的 embedding 毫无信息量, 却会:

    1. **污染检索结果** —— 语义模糊的短块容易意外匹配到很多无关查询
    2. **拉低指标可解释性** —— Recall@K 命中了一个几乎没内容的块

    做法是**合并**而不是丢弃: 丢弃会丢信息, 合并只是改变切分边界.
    合并时保留前一个块的 id, 因此 id 依然是确定性生成的, 不影响幂等性.

    三种分块策略共用这一份实现 —— 各写一遍必然会各自出 bug,
    而且修的时候容易漏掉其中一处.
    """
    if len(chunks) <= 1:
        return chunks

    merged: list[Chunk] = []
    for chunk in chunks:
        if merged and chunk.char_count < min_size:
            previous = merged[-1]
            merged[-1] = replace(
                previous,
                content=previous.content + chunk.content,
                page_start=min(previous.page_start, chunk.page_start),
                page_end=max(previous.page_end, chunk.page_end),
            )
        else:
            merged.append(chunk)

    # 首块过短时没有"前一块"可合并, 只能并入后一块
    if len(merged) >= 2 and merged[0].char_count < min_size:
        first, second = merged[0], merged[1]
        merged[1] = replace(
            second,
            content=first.content + second.content,
            page_start=min(first.page_start, second.page_start),
            page_end=max(first.page_end, second.page_end),
        )
        merged.pop(0)

    return merged
