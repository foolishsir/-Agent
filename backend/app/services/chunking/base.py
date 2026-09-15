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

from dataclasses import dataclass

from app.models.document import ChunkType

# 分块 id 的前缀/序号宽度. 固定宽度保证 id 长度一致, 也让按 id 排序等价于按顺序排序.
_PARENT_PAD = 4
_CHILD_PAD = 3


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
        """实际送去计算向量的文本.

        **与 ``content`` 故意不同**: 在正文前拼接章节路径.

        原因: 一个孤立的子块常常缺少语境. 例如正文只有
        "更换周期为 20000 次或 3 个月", 单独看不知道说的是什么.
        拼上章节路径变成 "第三章 设备维护 > 3.2 钢刀 > 更换周期为 20000 次或 3 个月",
        向量就能正确落在"设备维护"的语义空间里.

        这是低成本的检索质量提升手段: 不改模型、不改分块, 只是给文本补语境.
        把它和 ``content`` 分开, 是为了让存库的原文保持干净
        (引用展示、BM25 检索用的都是 ``content``).

        **去重**: 父块的第一个子块通常已经包含了章节标题本身(标题段与正文同属一个父块),
        此时再拼一遍会产生 "标题 > 标题 + 正文" 的重复文本, 白占 token 且稀释向量.
        所以先判断正文是否已经以标题开头.
        """
        if not self.section_path:
            return self.content

        leaf = self.section_path.split(" > ")[-1].strip()
        if leaf and self.content.lstrip().startswith(leaf):
            return self.content
        return f"{self.section_path} > {self.content}"


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
